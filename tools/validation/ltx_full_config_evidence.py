#!/usr/bin/env python3
"""Independent full-media configuration audit; never hash model payloads."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

from ltx_base_benchmark_evidence import portable
from ltx_full_config_screen import CASES, STDOUT_POLICY, make_request, parse_native_stdout, probe_media, request_seconds, validate_rows
from runtime_ane_common import sha256_file

ROOT = Path(__file__).resolve().parents[2]
MEMORY_KEYS = ("complete", "tree_peak_phys_footprint_bytes", "tree_peak_rss_bytes", "compression_bytes",
               "decompression_bytes", "swap_in_bytes", "swap_out_bytes")


def frame_metrics(reference, candidate):
    if not reference or len(reference) != len(candidate):
        raise ValueError("paired full RGB frame payloads required")
    a = np.frombuffer(reference, dtype=np.uint8).astype(np.float64)
    b = np.frombuffer(candidate, dtype=np.uint8).astype(np.float64)
    delta = a - b
    dot, aa, bb = float(np.sum(a * b)), float(np.sum(a * a)), float(np.sum(b * b))
    sa, sb, count = float(np.sum(a)), float(np.sum(b)), a.size
    covariance = dot - sa * sb / count
    va, vb = max(0.0, aa - sa * sa / count), max(0.0, bb - sb * sb / count)
    return dict(correlation=covariance / math.sqrt(va * vb) if va and vb else 1.0,
        cosine=dot / math.sqrt(aa * bb) if aa and bb else 1.0,
        mae_255=float(np.sum(np.abs(delta))) / count, rmse_255=math.sqrt(float(np.sum(delta * delta)) / count))


def motion_energy(previous, current):
    if not previous or len(previous) != len(current):
        raise ValueError("paired complete motion frames required")
    a = np.frombuffer(previous, dtype=np.uint8).astype(np.int16)
    b = np.frombuffer(current, dtype=np.uint8).astype(np.int16)
    return float(np.sum(np.abs(a - b))) / a.size


def compare_media(reference, candidate):
    """Same full-frame gate/decoder; vectorize only offline sufficient stats."""
    sys.path.insert(0, str(ROOT / "tools/native"))
    import video_quality_gate as gate
    original_frame, original_motion = gate._rgb_metrics, gate._motion_energy
    try:
        gate._rgb_metrics, gate._motion_energy = frame_metrics, motion_energy
        return gate.compare(reference, candidate)
    finally:
        gate._rgb_metrics, gate._motion_energy = original_frame, original_motion


def checked_cohort(data):
    order = data.get("order", [])
    if (data.get("status") != "complete_diagnostic" or not order or order != list(reversed(order)) or
            any(name not in CASES or order.count(name) != 2 for name in order) or len(data.get("trials", [])) != len(order)):
        raise ValueError("terminal exact two-process-per-configuration palindrome required")
    for index, trial in enumerate(data["trials"]):
        if trial["index"] != index or trial["name"] != order[index] or trial["configured"] != CASES[trial["name"]]:
            raise ValueError("actual order/configuration differs")
        validate_rows(trial["rows"], trial["name"])
        for row in trial["rows"]:
            request_seconds(row, trial["configured"]["single"])
    return order


def portable_diagnostics(values):
    result = []
    for item in values:
        value = item.copy()
        # The raw stdout binding preserves the exact local cache path. Publish
        # cache-relative identity, not the developer's account/home directory.
        path = value["resource_path"]
        marker = "/Library/Caches/turbocider/"
        value["resource_path"] = "<USER_CACHE>/turbocider/" + path.split(marker, 1)[1] if marker in path else "<LOCAL_E5_RESOURCE>/" + Path(path).name
        result.append(value)
    return result


def audit(screen, build_manifest, with_quality=False):
    data = json.loads((screen / "summary.json").read_text())
    order = checked_cohort(data)
    build = json.loads(build_manifest.read_text())
    if any(sha256_file(ROOT / item["path"]) != item["sha256"] for item in build["inputs"]["sources"]):
        raise ValueError("native build/source mismatch")
    library = build_manifest.parent.parent / "libturbocider.dylib"
    if data["source_sha256"].get(str(library.resolve())) != sha256_file(library):
        raise ValueError("actual library artifact differs")
    for path, digest in data["source_sha256"].items():
        if Path(path).suffix != ".safetensors" and sha256_file(Path(path)) != digest:
            raise ValueError("recorded helper/config/profile/collector source changed")
    result = dict(schema="tc-ltx-full-config-validation-v1", status="complete_diagnostic", date="2026-10-11",
        qualification_passed=False, goal_complete=False, hardware="M4 Max64GiB/macOS26.6.2",
        evidence=str((screen / "summary.json").relative_to(ROOT)), evidence_sha256=sha256_file(screen / "summary.json"),
        native_inputs_matched=len(build["inputs"]["sources"]), source_sha256=portable(data["source_sha256"]),
        order=order, scope=data["scope"], trials=[], routes=[], quality=[])
    for trial in data["trials"]:
        name, case = trial["name"], trial["configured"]
        directory = screen / f'{trial["index"]}-{name}'
        logs, memories = trial["stdout_logs"], trial["memory"]
        expected_processes = 2 if case["single"] else 1
        if len(logs) != expected_processes or len(memories) != expected_processes:
            raise ValueError("complete batch versus clean-exec process boundaries required")
        replayed, published_logs = [], []
        for iteration, (log, memory) in enumerate(zip(logs, memories)):
            stdout = directory / f"stdout-{iteration}.jsonl"
            if log["iteration"] != iteration or log["path"] != stdout.name or log["parser_policy"] != STDOUT_POLICY or sha256_file(stdout) != log["sha256"]:
                raise ValueError("actual complete raw stdout changed")
            rows, diagnostics = parse_native_stdout(stdout.read_text())
            if diagnostics != log["diagnostics"]:
                raise ValueError("raw diagnostic messages were hidden/changed")
            replayed += rows
            checked = memory["verified"]
            if checked["complete"] is not True or checked["command_exit_code"] != 0 or checked["reasons"]:
                raise ValueError("independent complete successful memory evidence required")
            for key in ("evidence", "report"):
                if sha256_file(directory / memory[key]) != memory[key + "_sha256"]:
                    raise ValueError("raw memory evidence changed")
            if json.loads((directory / memory["report"]).read_text()) != checked:
                raise ValueError("verified memory receipt differs from bound independent report")
            published_logs.append(dict(iteration=iteration, sha256=log["sha256"], parser_policy=STDOUT_POLICY,
                diagnostics=portable_diagnostics(diagnostics), diagnostic_scope=log["diagnostic_scope"]))
        if replayed != trial["rows"]:
            raise ValueError("actual decoded native receipts changed")
        if len(trial["media"]) != 2:
            raise ValueError("two complete MP4 receipts required")
        for iteration in range(2):
            request = json.loads((directory / f"request-{iteration}.json").read_text())
            if case["hybrid"] and Path(request["ane_manifest"]).resolve() != screen / "int8-profile.json":
                raise ValueError("actual request uses a different source-bound INT8 profile")
            expected = make_request(name, directory / f"{iteration}.mp4", Path(request["ane_manifest"]) if case["hybrid"] else None)
            if request != expected:
                raise ValueError("actual full-source request controls differ")
            media = probe_media(directory / f"{iteration}.mp4")
            if media["sha256"] != trial["media"][iteration]["sha256"]:
                raise ValueError("complete MP4 changed")
        result["trials"].append(dict(index=trial["index"], name=name, configured=case, rows=portable(trial["rows"]),
            media=portable(trial["media"]), stdout_logs=published_logs,
            memory=[{key: item["verified"][key] for key in MEMORY_KEYS} for item in memories],
            memory_bindings=[dict(evidence_sha256=item["evidence_sha256"], report_sha256=item["report_sha256"]) for item in memories],
            native_timing_boundary=trial["native_timing_boundary"], warm_scope=trial["warm_scope"],
            parent_wall_including_sampler_and_verification=trial["parent_wall_including_sampler_and_verification"],
            hybrid_evidence_scope=trial["hybrid_evidence_scope"]))
    for name in sorted(set(order)):
        trials = [trial for trial in data["trials"] if trial["name"] == name]
        case = CASES[name]
        cold, warm = ([request_seconds(trial["rows"][index], case["single"]) for trial in trials] for index in (0, 1))
        declared = next(item for item in data["routes"] if item["name"] == name)
        if declared["warm_seconds"] != warm or declared["warm_median"] != statistics.median(warm):
            raise ValueError("all actual warm samples must match declared median")
        result["routes"].append(dict(name=name, configured=case, cold_seconds=cold, warm_seconds=warm,
            warm_median=statistics.median(warm), warm_scope=trials[0]["warm_scope"],
            native_timing_boundary=trials[0]["native_timing_boundary"],
            physical_peaks=[max(item["verified"]["tree_peak_phys_footprint_bytes"] for item in trial["memory"]) for trial in trials]))
    # Never pool clean-exec estimates/new-process load with persistent request_wall.
    result["best_measured"] = []
    for single in (False, True):
        candidates = [route for route in result["routes"] if route["configured"]["single"] == single]
        if candidates:
            fastest = min(candidates, key=lambda route: route["warm_median"])
            result["best_measured"].append(dict(single=single, name=fastest["name"], seconds=fastest["warm_median"],
                timing_boundary=fastest["native_timing_boundary"], scope="tested configuration median only; attention/INT8 approximations separate; not global optimum or broad quality qualification"))
    if with_quality:
        reference = next((trial for trial in data["trials"] if trial["name"] == "gpu-quality-batch"), None)
        if reference is None:
            raise ValueError("full quality GPU reference required for media comparison")
        baseline = screen / f'{reference["index"]}-{reference["name"]}' / "1.mp4"
        for name in sorted(set(order)):
            trial = next(trial for trial in data["trials"] if trial["name"] == name)
            candidate = screen / f'{trial["index"]}-{name}' / "1.mp4"
            metrics = compare_media(baseline, candidate)
            if metrics["frame_count_compared"] != 97 or metrics["finite"] is not True or not all(metrics[key] for key in ("shape_equal", "frame_count_equal", "fps_equal")):
                raise ValueError("complete original-resolution 97-frame finite media comparison required")
            result["quality"].append(dict(name=name, reference_sha256=sha256_file(baseline), candidate_sha256=sha256_file(candidate), metrics=portable(metrics)))
    result["quality_scope"] = "full97-frame decoded RGB/motion when explicitly requested; original gate/ffmpeg decoder with equivalent float64 sufficient statistics vectorized offline; no resize or frame sampling; configured profile not physical NE trace; no broad perceptual or latent quality promotion"
    if "/Users/" in json.dumps(result):
        raise ValueError("portable evidence contains machine-specific paths")
    return result


def compact(full, full_path):
    if full["status"] != "complete_diagnostic" or len(full["quality"]) != len(set(full["order"])):
        raise ValueError("complete full-media quality audit required for publication")
    result = {key: value for key, value in full.items() if key not in ("trials", "quality")}
    result.update(full_audit=str(full_path.relative_to(ROOT)), full_audit_sha256=sha256_file(full_path), trials=[], quality=[])
    row_keys = ("model", "operation", "width", "height", "frames", "fps", "audio", "execution",
        "conditioning_cache_hit", "video_vae_isolation", "ane_profile", "denoiser_implementation",
        "gpu_parallel_av", "gpu_batch_audio_commands", "timings_seconds", "validation")
    plan_keys = ("algorithm_approximations", "execution", "ltx_sol_stage1", "ltx_sol_stage2", "ltx_sol_tau",
        "ltx_sol_dense_edge_blocks", "ltx_sol_dense_edge_steps", "ltx_stage2_text_rows", "stages")
    for trial in full["trials"]:
        item = {key: value for key, value in trial.items() if key not in ("rows", "media")}
        item["rows"] = []
        for row in trial["rows"]:
            retained = {key: row[key] for key in row_keys if key in row}
            if "plan" in row:
                retained["plan"] = {key: row["plan"][key] for key in plan_keys if key in row["plan"]}
            item["rows"].append(retained)
        item["media"] = [dict(sha256=media["sha256"], width=media["video"]["width"], height=media["video"]["height"],
            frames=int(media["video"]["nb_frames"]), fps=media["video"]["r_frame_rate"],
            video_codec=media["video"]["codec_name"], channels=media["audio"]["channels"],
            sample_rate=int(media["audio"]["sample_rate"]), audio_codec=media["audio"]["codec_name"]) for media in trial["media"]]
        result["trials"].append(item)
    for row in full["quality"]:
        if row["metrics"]["frame_count_compared"] != 97 or len(row["metrics"]["frame_metrics"]) != 97:
            raise ValueError("all original97-frame metrics must be bound before compacting")
        result["quality"].append({**row, "metrics": {key: value for key, value in row["metrics"].items() if key != "frame_metrics"}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("screen", "build-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--with-quality", action="store_true", help="full97-frame RGB/motion; run only after hardware experiments are idle")
    parser.add_argument("--compact-output", type=Path, help="optional portable concise publication bound to the complete frame-level audit")
    args = parser.parse_args()
    if args.output.exists() or args.compact_output and args.compact_output.exists():
        parser.error("fresh evidence output required")
    result = audit(args.screen.resolve(), args.build_manifest.resolve(), args.with_quality)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    if args.compact_output:
        publication = compact(result, args.output.resolve())
        args.compact_output.parent.mkdir(parents=True, exist_ok=True)
        with args.compact_output.open("x") as output:
            json.dump(publication, output, indent=2, allow_nan=False)
            output.write("\n")
    print(json.dumps(dict(status=result["status"], best_measured=result["best_measured"])))


if __name__ == "__main__":
    main()
