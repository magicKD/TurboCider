#!/usr/bin/env python3
"""Portable current LTX full-media timing and qualified-scope comparison."""
import argparse
import json
import math
from pathlib import Path
import statistics

from runtime_ane_common import sha256_file
from ltx_base_benchmark_screen import validate_rows

ROOT = Path(__file__).resolve().parents[2]


def portable(value):
    if isinstance(value, dict):
        return {portable(key): portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str):
        return value.replace(str(ROOT) + "/", "")
    return value


def validate_sources(first, second):
    """Compare recorded inputs, without re-reading multi-GB model payloads."""
    def inputs(raw):
        profiles = {f'{raw["directory"]}/{mode}-profile.json' for mode in raw["order"] if mode != "gpu"}
        sources = {portable(name): digest for name, digest in raw["source_sha256"].items()}
        if not profiles <= sources.keys():
            raise ValueError("configured profile source hashes missing")
        return {name: digest for name, digest in sources.items() if name not in profiles}

    a, b = inputs(first), inputs(second)
    fp16_prefix = "models/coreml/ltx_mlp_a6912_dual_fp16_20260913/"
    fp16 = {name: digest for name, digest in a.items() if name.startswith(fp16_prefix)}
    shared = {name: digest for name, digest in a.items() if not name.startswith(fp16_prefix)}
    int8_prefix = "models/coreml/ltx_mlp_a6912_dual_int8_20260913/"
    if (len(fp16) != 48 or sum(name.startswith(int8_prefix) for name in shared) != 48 or
            not any(name.endswith("/libturbocider.dylib") for name in shared) or
            not any(name.endswith("/ltx-video-vae-decode") for name in shared) or
            not any(name.endswith(".safetensors") for name in shared) or shared != b):
        raise ValueError("same complete original source/runtime inputs required")


def validate_request(request, row, mode, index):
    if (request.get("model"), request.get("operation"), request.get("width"), request.get("height"),
            request.get("frames"), request.get("fps"), request.get("steps"), request.get("seed"), request.get("audio")) != (
                "ltx-2.5-distilled", "video.generate", 704, 448, 97, 24, 11, 42, True) or request.get("loras"):
        raise ValueError("original no-adapter fixed full-media schedule required")
    execution = "gpu" if mode == "gpu" else "gpu_ane"
    if (request.get("execution") != execution or row.get("execution") != execution or
            row.get("conditioning_cache_hit") is not (index == 1) or row.get("denoiser_implementation") != "c_metal"):
        raise ValueError("actual C/Metal cold/warm lifecycle or execution differs")
    wall = row["timings_seconds"]["request_wall"]
    if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
        raise ValueError("finite positive full-media request wall required")


def validate_quality(quality):
    if set(quality) != {"int8", "fp16"}:
        raise ValueError("both full-media route comparisons required")
    for comparison in quality.values():
        if (any(comparison.get(name) is not True for name in ("shape_equal", "frame_count_equal", "fps_equal", "finite")) or
                comparison.get("frame_count_compared") != 97 or len(comparison.get("frame_metrics", [])) != 97):
            raise ValueError("complete finite full-resolution quality evidence required")
        for name in ("reference", "candidate"):
            media = comparison[name]
            if (media["width"], media["height"], media["fps"], media["decoded_frames"]) != (704, 448, 24, 97):
                raise ValueError("quality comparison uses a different video task")


def collect(path, expected):
    raw = json.loads((path / "summary.json").read_text())
    if raw["status"] != "complete_diagnostic" or raw["order"] != expected or len(raw["trials"]) != len(expected):
        raise ValueError("complete exact full-media order required")
    trials = []
    for trial_index, trial in enumerate(raw["trials"]):
        mode = trial["mode"]
        if mode != expected[trial_index] or trial["index"] != trial_index:
            raise ValueError("actual trial order differs")
        validate_rows(trial["rows"], mode)
        memory = trial["memory"]
        if memory["verified"]["complete"] is not True or memory["verified"]["command_exit_code"] != 0:
            raise ValueError("independent complete memory evidence required")
        directory = path / f'{trial["index"]}-{mode}'
        for key in ("evidence", "report"):
            if sha256_file(directory / memory[key]) != memory[key + "_sha256"]:
                raise ValueError("memory evidence changed")
        rows = []
        for index, row in enumerate(trial["rows"]):
            request = json.loads((directory / f"request-{index}.json").read_text())
            validate_request(request, row, mode, index)
            media = trial["media"][index]
            if sha256_file(directory / f"{index}.mp4") != media["sha256"]:
                raise ValueError("media changed")
            video, audio = media["video"], media["audio"]
            if (video["width"], video["height"], int(video["nb_frames"]), video["r_frame_rate"]) != (704, 448, 97, "24/1") or audio["channels"] != 2 or int(audio["sample_rate"]) != 48000:
                raise ValueError("complete 97-frame stereo media required")
            rows.append(dict(index=index, timings_seconds=row["timings_seconds"],
                conditioning_cache_hit=row["conditioning_cache_hit"], execution=row["execution"],
                gpu_parallel_av=row["gpu_parallel_av"], gpu_batch_audio_commands=row["gpu_batch_audio_commands"],
                ane_profile=portable(row["ane_profile"]), video_vae_isolation=row["video_vae_isolation"],
                request=portable(request), media=dict(sha256=media["sha256"], width=video["width"], height=video["height"],
                    frames=int(video["nb_frames"]), fps=video["r_frame_rate"], video_codec=video["codec_name"],
                    audio_codec=audio["codec_name"], sample_rate=int(audio["sample_rate"]), channels=audio["channels"],
                    video_duration=video["duration"], audio_duration=audio["duration"])))
        trials.append(dict(mode=mode, rows=rows, memory={key: memory["verified"][key] for key in (
            "complete", "tree_peak_phys_footprint_bytes", "compression_bytes", "decompression_bytes", "swap_in_bytes", "swap_out_bytes")},
            memory_evidence_sha256=memory["evidence_sha256"], memory_report_sha256=memory["report_sha256"]))
    raw["directory"] = str(path.relative_to(ROOT))
    return raw, dict(evidence=str((path / "summary.json").relative_to(ROOT)),
                     evidence_sha256=sha256_file(path / "summary.json"), order=raw["order"], trials=trials)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--reverse", type=Path, required=True)
    parser.add_argument("--media-quality", type=Path, required=True)
    parser.add_argument("--mlxserve-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh portable evidence required")
    a, first = collect(args.forward.resolve(), ["gpu", "int8", "fp16"])
    b, second = collect(args.reverse.resolve(), ["int8", "gpu"])
    # Only generated profile destinations and the forward-only FP16 graphs
    # may differ; an intersection alone would hide missing source inputs.
    validate_sources(a, b)
    result = dict(schema="tc-ltx-base-current-validation-v1", status="complete_diagnostic", date="2026-10-11",
        qualification_passed=False, goal_complete=False, hardware="M4 Max64 GiB/macOS26.6.2", sweeps=[first, second],
        source_sha256=portable(a["source_sha256"]), routes=[],
        scope="current original local distilled ConvRot base; 704x448x97/24fps/11 steps/stereo; configured C/Metal+CoreML artifact route, not physical ANE-residency/call-count trace")
    for mode in ("gpu", "int8", "fp16"):
        trials = [trial for sweep in result["sweeps"] for trial in sweep["trials"] if trial["mode"] == mode]
        cold = [trial["rows"][0]["timings_seconds"]["request_wall"] for trial in trials]
        warm = [trial["rows"][1]["timings_seconds"]["request_wall"] for trial in trials]
        result["routes"].append(dict(mode=mode, cold_seconds=cold, warm_seconds=warm, warm_median=statistics.median(warm),
                                   physical_peaks=[trial["memory"]["tree_peak_phys_footprint_bytes"] for trial in trials]))
    result["video_quality"] = portable(json.loads(args.media_quality.read_text()))
    validate_quality(result["video_quality"])
    result["video_quality_scope"] = "full decoded RGB/motion comparison of the first sweep's warm MP4s; no resize; not broad perceptual/semantic or latent equivalence"
    result["video_quality_evidence_sha256"] = sha256_file(args.media_quality)
    probe = json.loads((args.mlxserve_probe / "summary.json").read_text())
    if (probe["server_terminal"] is not True or probe["status"] != "startup_rejected" or probe["requests"] or
            probe["exit_code"] != 1 or probe["latency_comparison_valid"] is not False):
        raise ValueError("actual terminal rejected original-layout framework probe required")
    result["mlxserve"] = dict(result=portable(probe), reason="stock media-memory preflight rejected original local LTX pack before generation",
        evidence=str((args.mlxserve_probe / "summary.json").resolve().relative_to(ROOT)),
        evidence_sha256=sha256_file(args.mlxserve_probe / "summary.json"))
    result["retained_failed_window"] = dict(evidence="outputs/ltx-base-current-v1-gpu-20261011/summary.json",
        reason="native produced two complete clips but decoder child was unknown to sampler; inconclusive, not promoted")
    if "/Users/" in json.dumps(result):
        raise ValueError("portable evidence contains machine-specific paths")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        json.dump(result, destination, indent=2, allow_nan=False); destination.write("\n")
    print(json.dumps(result["routes"]))


if __name__ == "__main__":
    main()
