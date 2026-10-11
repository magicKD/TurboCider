#!/usr/bin/env python3
"""Portable actual samples, memory and RGB comparison; never loads model weights."""
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from PIL import Image

from flux9_bf16_framework_screen import requests, validate_native
from runtime_ane_common import sha256_file
from runtime_ane_image_compare import rgb_metrics

ROOT = Path(__file__).resolve().parents[2]


def portable(value):
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str):
        return value.replace(str(ROOT) + "/", "").replace(str(ROOT.parent / "references") + "/", "../references/")
    return value


def rgb(path):
    with Image.open(path) as image:
        value = np.array(image)
    if value.shape == (512, 512, 4):
        if not np.all(value[:, :, 3] == 255):
            raise ValueError("non-opaque native alpha cannot be silently dropped")
        value = value[:, :, :3]
    if value.shape != (512, 512, 3) or value.dtype != np.uint8:
        raise ValueError("same-size unresized 8-bit RGB required")
    return value


def audit(screen, mlxserve, build_manifest, evidence_date="2026-10-11"):
    date.fromisoformat(evidence_date)
    data = json.loads((screen / "summary.json").read_text())
    if (data["status"] != "complete_diagnostic" or data.get("stage_probe") is not False or
            data["order"] not in (["native", "mflux", "mflux", "native"],
                ["native", "native-compiled", "mflux", "mflux", "native-compiled", "native"],
                ["native", "native-compiled", "native-hybrid", "mflux", "mflux", "native-hybrid", "native-compiled", "native"]) or
            len(data["trials"]) != len(data["order"])):
        raise ValueError("complete original BF16 native/reference palindrome required")
    build = json.loads(build_manifest.read_text())
    if any(sha256_file(ROOT / row["path"]) != row["sha256"] for row in build["inputs"]["sources"]):
        raise ValueError("native build/source mismatch")
    if sha256_file(build_manifest.parent.parent / "libturbocider.dylib") != data["source_sha256"]["libturbocider.dylib"]:
        raise ValueError("native library artifact mismatch")
    output = dict(schema="tc-flux9-bf16-framework-validation-v1", status="complete_diagnostic", date=evidence_date,
                  qualification_passed=False, goal_complete=False, hardware="M4 Max 64 GiB/macOS26.6.2",
                  task=data["task"], model_sha256=data["model_sha256"], source_sha256=data["source_sha256"],
                  native_inputs_matched=len(build["inputs"]["sources"]), mflux_commit=data["mflux_commit"],
                  mflux_source_files=data["mflux_source_files"], mflux_source_snapshot=data["mflux_source_snapshot"],
                  evidence=str((screen / "summary.json").relative_to(ROOT)),
                  evidence_sha256=sha256_file(screen / "summary.json"), order=data["order"], trials=[],
                  timing_scope=data["timing_scope"], source_policy=data["source_policy"],
                  scope="same BF16 files and native/reference templates; fresh prompt latency, not equal allocator/residency policy or strict numerical equivalence")
    output["reference_cache_mib"] = data.get("reference_cache_mib")
    output["native_encoder_eval_interval"] = data.get("native_encoder_eval_interval")
    output["native_encoder_fused_attention"] = data.get("native_encoder_fused_attention", False)
    has_hybrid = "native-hybrid" in data["order"]
    if has_hybrid:
        bindings = data.get("ane_source_sha256", {})
        manifest = Path(data["ane_manifest"])
        if not bindings or bindings.get(str(manifest.relative_to(ROOT))) != sha256_file(manifest):
            raise ValueError("retained experimental manifest identity required")
        for name, digest in bindings.items():
            if sha256_file(ROOT / name) != digest:
                raise ValueError("retained compiled CoreML partition changed")
        metadata = json.loads(manifest.read_text())
        source = metadata["source"]
        index = ROOT / "models/FLUX.2-klein-9B/transformer/diffusion_pytorch_model.safetensors.index.json"
        if (source.get("checkpoint") != str(index) or source.get("blocks") != list(range(24)) or
                source.get("checkpoint_sha256") != data["model_sha256"].get(str(index.relative_to(ROOT))) or
                len(source.get("checkpoint_shards", {})) != 2):
            raise ValueError("complete original9B checkpoint/index identity required")
        for name, identity in source["checkpoint_shards"].items():
            if identity["sha256"] != data["model_sha256"].get(str((index.parent / name).relative_to(ROOT))):
                raise ValueError("original9B complete shard identities differ")
        output["ane_manifest"] = str(manifest.relative_to(ROOT))
        output["ane_source_sha256"] = bindings
        output["representation_scope"] = "original BF16 source; hybrid uses CoreML INT8 per-channel weights/FP16 I/O prefix plus BF16 GPU8192-channel complement; not activation8/native INT8 MAC or equal-precision comparison"
    reference_coverage_complete = True
    for expected_index, trial in enumerate(data["trials"]):
        if trial["index"] != expected_index or trial["engine"] != data["order"][expected_index]:
            raise ValueError("actual complete palindrome order differs")
        report, memory = trial["report"], trial["memory"]
        if memory["verified"]["complete"] is not True or memory["verified"]["command_exit_code"] != 0:
            raise ValueError("complete successful independent process-tree proof required")
        directory = screen / f'{trial["index"]}-{trial["engine"]}'
        if json.loads((directory / "report.json").read_text()) != report:
            raise ValueError("raw native/reference report differs from collected summary")
        for key in ("evidence", "report"):
            if sha256_file(directory / memory[key]) != memory[key + "_sha256"]:
                raise ValueError("memory evidence changed")
        if report["engine"].startswith("native"):
            hybrid = report["engine"] == "native-hybrid"
            validate_native(report["runs"], hybrid)
            if report["engine"] == "native-compiled" and any(row.get("actual_compiled_single_blocks") != 96 for row in report["runs"]):
                raise ValueError("actual complete9B compiled single blocks required")
            if hybrid:
                if report.get("ane_manifest_sha256") != sha256_file(Path(data["ane_manifest"])):
                    raise ValueError("actual loaded hybrid manifest identity differs")
                for index, expected in enumerate(requests(directory, ane_manifest=data["ane_manifest"])):
                    if json.loads((directory / f"request-{index}.json").read_text()) != expected:
                        raise ValueError("actual experimental base-only9B hybrid request differs")
        elif (report.get("instrumentation") != "host-step-counter-only; no extra GPU sync or encoder wrapper" or
              report["quantization"] is not None or report["precision"] != "mlx.core.bfloat16" or
              len(report["runs"]) != 4 or any(row["actual_steps"] != [0, 1, 2, 3] or row["loras"] or
                  row["actual_quantization"] is not None or row["text_rows"] != 512 for row in report["runs"])):
            raise ValueError("wrong actual original mflux base precision/geometry/work")
        if report["engine"] == "mflux":
            if data.get("reference_cache_mib") is not None and report.get("reference_cache_limit_bytes") != data["reference_cache_mib"] << 20:
                raise ValueError("actual reference process-local cache candidate differs")
            coverage = report.get("required_weight_coverage")
            if coverage is None:
                reference_coverage_complete = False  # Historical cohort; never invent its receipt.
            elif not isinstance(coverage, dict) or set(coverage) != {"vae", "transformer", "text_encoder"} or any(
                    type(row.get("required_targets")) is not int or row["required_targets"] <= 0 or
                    row.get("covered_targets") != row["required_targets"] or row.get("missing") != [] for row in coverage.values()):
                raise ValueError("incomplete actual reference source coverage")
        if trial["pngs"][2]["rgb_sha256"] != trial["pngs"][3]["rgb_sha256"]:
            raise ValueError("repeat changed output within the same engine")
        if len(trial["pngs"]) != 4:
            raise ValueError("four full original-size PNG receipts required")
        for index, identity in enumerate(trial["pngs"]):
            path = directory / f"{index}.png"
            if (sha256_file(path) != identity["file_sha256"] or
                    hashlib.sha256(rgb(path).tobytes()).hexdigest() != identity["rgb_sha256"]):
                raise ValueError("full PNG or raw RGB identity changed")
        output["trials"].append(dict(index=trial["index"], engine=trial["engine"], report=portable(report),
            memory={key: memory["verified"][key] for key in ("complete", "tree_peak_phys_footprint_bytes",
                "compression_bytes", "decompression_bytes", "swap_in_bytes", "swap_out_bytes")},
            memory_evidence_sha256=memory["evidence_sha256"], memory_report_sha256=memory["report_sha256"],
            pngs=trial["pngs"]))
    output["reference_required_weight_coverage_verified"] = reference_coverage_complete
    if has_hybrid and not reference_coverage_complete:
        raise ValueError("new experimental cohort requires complete actual reference coverage")
    output["medians"] = data["medians"]
    for engine, declared in data["medians"].items():
        actual = [row["request_wall_including_export"] for trial in output["trials"] if trial["engine"] == engine
                  for row in trial["report"]["runs"][1:3]]
        if declared["samples"] != actual or declared["fresh_warm_seconds"] != statistics.median(actual):
            raise ValueError("median is not actual pooled fresh-request samples")
    a, b = (data["medians"][engine]["fresh_warm_seconds"] for engine in ("native", "mflux"))
    output["fresh_warm_speedup"] = b / a
    output["fresh_warm_reduction_percent"] = 100 * (1 - a / b)
    if "native-compiled" in data["medians"]:
        compiled = data["medians"]["native-compiled"]["fresh_warm_seconds"]
        output["compiled_gpu_comparison"] = dict(native_eager_over_compiled=a / compiled,
            reference_over_compiled=b / compiled, compiled_reduction_vs_eager_percent=100 * (1 - compiled / a),
            scope="same BF16 source/four-step whole requests;96 actual host-submitted compiled single-block branches per request, no added fences; not physical GPU occupancy trace")
    if has_hybrid:
        hybrid = data["medians"]["native-hybrid"]["fresh_warm_seconds"]
        output["hybrid_comparison"] = dict(native_eager_over_hybrid=a / hybrid,
            native_compiled_over_hybrid=compiled / hybrid, reference_over_hybrid=b / hybrid,
            hybrid_reduction_vs_compiled_percent=100 * (1 - hybrid / compiled),
            ane_channels=4096, gpu_complement_channels=8192, actual_runtime_calls_per_request=96,
            fallback=False, scope="complete four-step requests,24 single blocks each step; host async GPU complement and blocking CoreML prefix; physical hardware overlap/ANE occupancy not measured")
    output["rgb_comparisons"] = []
    pairs = []
    for route in ("native-compiled", "native-hybrid", "mflux"):
        positions = [index for index, engine in enumerate(data["order"]) if engine == route]
        if positions:
            pairs.extend((("0-native", f"{positions[0]}-{route}"),
                          (f'{len(data["order"])-1}-native', f"{positions[1]}-{route}")))
    for native, reference in pairs:
        for index in range(4):
            left, right = screen / native / f"{index}.png", screen / reference / f"{index}.png"
            x, y = rgb(left), rgb(right)
            measured = rgb_metrics(x, y)
            measured["mae_u8"] = float(np.mean(np.abs(x.astype(np.float64) - y)))
            output["rgb_comparisons"].append(dict(native=native, reference=reference, index=index,
                native_png_sha256=sha256_file(left), reference_png_sha256=sha256_file(right), **measured))
    output["forward_reverse_rgb_determinism"] = []
    for route in sorted(set(data["order"])):
        trials = [trial for trial in data["trials"] if trial["engine"] == route]
        output["forward_reverse_rgb_determinism"].append(dict(engine=route, exact_requests=[
            trials[0]["pngs"][index]["rgb_sha256"] == trials[1]["pngs"][index]["rgb_sha256"] for index in range(4)]))
    output["rgb_comparison_scope"] = "native opaque alpha verified; compare original RGB without resize; same seed not proof of identical noise; not broad perceptual qualification"
    probe = json.loads((mlxserve / "summary.json").read_text())
    if (probe["status"] != "startup_rejected" or probe["server_terminal"] is not True or probe["requests"] or
            probe["latency_comparison_valid"] is not False or probe["exit_code"] != 1):
        raise ValueError("explicit unmodified original BF16 rejection proof required")
    logs = (mlxserve / "stderr.txt").read_text()
    if "MISSING WEIGHT: embed_tokens.weight" not in logs or "MissingFluxWeight" not in logs:
        raise ValueError("probe did not prove original layout/key rejection")
    output["mlxserve"] = dict(evidence=str((mlxserve / "summary.json").relative_to(ROOT)),
        evidence_sha256=sha256_file(mlxserve / "summary.json"), result=portable(probe),
        rejection="original HF text-encoder keys do not match expected mflux-converted keys; affine planes also required by pinned loader",
        release_commit="02bee553f48cd3bc7d82aba0f8073820bd924738",
        release_flux_source_sha256=sha256_file(ROOT / "outputs/mlxserve-v26.10.1-flux.zig"))
    output["retained_failures"] = [
        dict(evidence="outputs/flux9-bf16-framework-v1-precheck-20261011.log", reason="empty Python __init__.py rejected by model-only source guard"),
        dict(evidence="outputs/flux9-bf16-framework-v2-precheck-20261011/summary.json",
             reason="all images completed but unknown Python child made independent memory evidence inconclusive; no performance promotion")]
    output["retained_stage_observer"] = dict(evidence="outputs/flux9-bf16-framework-v3-abba-20261011/summary.json",
        purpose="auxiliary phase observer with eval boundaries and pre-call token checks; not the primary stock whole-request denominator")
    if "/Users/" in json.dumps(output):
        raise ValueError("machine-specific paths in portable evidence")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--mlxserve-probe", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--date", default="2026-10-11", help="actual publication/completion local date; raw campaign directory labels may predate completion")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh portable evidence output required")
    result = audit(args.screen.resolve(), args.mlxserve_probe.resolve(), args.build_manifest.resolve(), args.date)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as target:
        json.dump(result, target, indent=2, allow_nan=False); target.write("\n")
    print(json.dumps(dict(status=result["status"], fresh_warm_speedup=result["fresh_warm_speedup"],
                          native_inputs_matched=result["native_inputs_matched"])))


if __name__ == "__main__":
    main()
