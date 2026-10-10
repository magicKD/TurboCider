#!/usr/bin/env python3
"""Audit two complete serial sweeps; portable samples and per-format winners."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np
from PIL import Image

from runtime_ane_common import sha256_file
from runtime_ane_image_compare import rgb_metrics
from z_base_formats_screen import cases, validate_rows

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_KEYS = ("executor_backend", "data_path", "source_recipe", "partition_axis", "ane_channels", "gpu_channels",
                "channel_blocks_session_total", "hybrid_blocks_session_total", "fallback_blocks_session_total",
                "overflow_retries_session_total", "headroom_scale", "device_io_calls_session_total",
                "fp32_channel_join_enabled", "io_path", "stage_seconds_session_total", "gpu_ffn_seconds_session_total")
IMPORT_KEYS = ("recipe", "gpu_graph_recipe", "source_sha256", "dense_scope", "dense_weight_capacity_upper_bytes",
               "packed_capacity_bytes", "allocator_cache_limit_bytes", "source_read_bytes", "request_source_read_bytes",
               "request_verification_bytes", "request_load_seconds", "session_packed_retention", "reused_packed_bank")


def relative(path):
    return str(path.resolve().relative_to(ROOT))


def controls(values):
    return {name: value.replace(str(ROOT) + "/", "") for name, value in values.items()}


def samples(directory):
    data = json.loads((directory / "summary.json").read_text())
    if data["status"] != "complete_diagnostic" or data["steps"] != 8 or set(data["names"]) != set(cases()) or len(data["trials"]) != len(cases()):
        raise ValueError("complete all-format 8-step sweep required")
    output = []
    for trial in data["trials"]:
        case = cases()[trial["name"]]
        if trial["case"] != case:
            raise ValueError("wrong declared source/recipe")
        rows = trial["rows"]
        validate_rows(rows, case, 8)
        checked = trial["memory"]["verified"]
        if checked["complete"] is not True or checked["command_exit_code"] != 0:
            raise ValueError("successful complete memory evidence required")
        raw = directory / trial["name"]
        for key in ("evidence", "report"):
            if sha256_file(raw / trial["memory"][key]) != trial["memory"][key + "_sha256"]:
                raise ValueError("memory evidence changed")
        actual = []
        previous_calls = previous_retries = previous_blocks = 0
        for index, row in enumerate(rows):
            if row["actual_denoise_steps"] != 8:
                raise ValueError("actual work was shortened")
            hybrid = row.get("hybrid") or {}
            runtime = hybrid.get("runtime_weight") or {}
            if case["mode"] == "private":
                calls, retries, blocks = (hybrid["runtime_calls_session_total"],
                    runtime["overflow_retries_session_total"], runtime["channel_blocks_session_total"])
                if blocks - previous_blocks != 256 or calls - previous_calls != 256 + retries - previous_retries:
                    raise ValueError("actual channel work/retry count differs")
                if index and retries != previous_retries:
                    raise ValueError("warm execution changed headroom/retry policy")
                previous_calls, previous_retries, previous_blocks = calls, retries, blocks
            location = raw / f"{index}.png"
            if sha256_file(location) != trial["png_sha256"][index]:
                raise ValueError("output changed")
            actual.append(dict(index=index, prompt_cache_hit=row["prompt_cache_hit"], text_tokens=row["text_tokens"],
                runtime_backend=row["runtime_backend"], runtime_precision=row["runtime_precision"], gpu_graph=row["gpu_graph"],
                timings_seconds=row["timings_seconds"], png_sha256=trial["png_sha256"][index],
                import_recipe={key: row["gguf_import"][key] for key in IMPORT_KEYS if key in (row.get("gguf_import") or {})},
                runtime_calls=hybrid.get("runtime_calls_session_total", 0),
                runtime_weight={key: runtime[key] for key in RUNTIME_KEYS if key in runtime}))
        output.append(dict(name=trial["name"], format=case["format"], mode=case["mode"],
            controls=controls(trial["controls"]), runs=actual,
            memory={key: checked[key] for key in ("complete", "tree_peak_phys_footprint_bytes", "tree_peak_rss_bytes",
                "compression_bytes", "decompression_bytes", "swap_in_bytes", "swap_out_bytes")},
            memory_evidence_sha256=trial["memory"]["evidence_sha256"], memory_report_sha256=trial["memory"]["report_sha256"]))
    return data, dict(evidence=relative(directory / "summary.json"),
                     evidence_sha256=sha256_file(directory / "summary.json"), order=data["names"], trials=output)


def rgb(path):
    with Image.open(path) as image:
        array = np.array(image)
    if array.shape == (512, 512, 4):
        if not np.all(array[:, :, 3] == 255):
            raise ValueError("non-opaque alpha must not be discarded")
        array = array[:, :, :3]
    if array.shape != (512, 512, 3) or array.dtype != np.uint8:
        raise ValueError("original unresized RGB required")
    return array


def best_routes(routes, source):
    selected = [row for row in routes if row["format"] == source]
    gpu = min((row for row in selected if row["mode"] in ("gpu", "fast")), key=lambda row: row["fresh_warm_median"])
    hybrid = [row for row in selected if row["mode"] not in ("gpu", "fast")]
    best = min(hybrid, key=lambda row: row["fresh_warm_median"]) if hybrid else None
    return dict(format=source, gpu=gpu["name"], gpu_fresh_seconds=gpu["fresh_warm_median"],
                hybrid=best["name"] if best else None, hybrid_fresh_seconds=best["fresh_warm_median"] if best else None,
                gpu_over_hybrid=gpu["fresh_warm_median"] / best["fresh_warm_median"] if best else None,
                hybrid_reduction_percent=100*(1-best["fresh_warm_median"]/gpu["fresh_warm_median"]) if best else None)


def audit(forward, reverse, manifest):
    a, first = samples(forward)
    b, second = samples(reverse)
    if a["source_sha256"] != b["source_sha256"] or a["names"] != list(reversed(b["names"])):
        raise ValueError("exact same sources, library and reverse route order required")
    build = json.loads(manifest.read_text())
    if build["inputs"]["policy"]["experimental_probes"] != "1" or any(
            sha256_file(ROOT / row["path"]) != row["sha256"] for row in build["inputs"]["sources"]):
        raise ValueError("matching explicit experimental build required")
    result = dict(schema="tc-z-base-format-validation-v1", status="complete_diagnostic", date="2026-10-11",
        qualification_passed=False, goal_complete=False, hardware="M4 Max64 GiB/macOS26.6.2",
        task="512 square/8 steps/seed42; cold + two fresh warm + repeated warm; no LoRA",
        native_inputs_matched=len(build["inputs"]["sources"]),
        source_sha256={str(Path(name).relative_to(ROOT)): digest for name, digest in a["source_sha256"].items()},
        sweeps=[first, second], routes=[], best_per_format=[], output_checks=[],
        scope="same-library/components/task; representations and opt-in arithmetic differ; timing and memory diagnostics, no hard cap/physical-overlap/general visual qualification")
    for name, case in cases().items():
        trials = [next(trial for trial in sweep["trials"] if trial["name"] == name) for sweep in result["sweeps"]]
        values = [row["timings_seconds"]["request_wall"] for trial in trials for row in trial["runs"][1:3]]
        repeat = [trial["runs"][3]["timings_seconds"]["request_wall"] for trial in trials]
        result["routes"].append(dict(name=name, **case, fresh_warm_seconds=values, fresh_warm_median=statistics.median(values),
            repeat_seconds=repeat, repeat_median=statistics.median(repeat),
            cold_seconds=[trial["runs"][0]["timings_seconds"]["request_wall"] for trial in trials],
            physical_peaks=[trial["memory"]["tree_peak_phys_footprint_bytes"] for trial in trials]))
        for index in range(4):
            x, y = rgb(forward / name / f"{index}.png"), rgb(reverse / name / f"{index}.png")
            result["output_checks"].append(dict(name=name, index=index, forward_reverse_rgb_exact=bool(np.array_equal(x, y))))
    for source in ("bf16", "nvfp4", "convrot", "q4", "q8"):
        result["best_per_format"].append(best_routes(result["routes"], source))
    result["rgb_source_comparisons"] = []
    for best in result["best_per_format"]:
        if not best["hybrid"]:
            continue
        x, y = rgb(forward / best["gpu"] / "2.png"), rgb(forward / best["hybrid"] / "2.png")
        measured = rgb_metrics(x, y)
        measured["mae_u8"] = float(np.mean(np.abs(x.astype(np.float64)-y)))
        result["rgb_source_comparisons"].append(dict(format=best["format"], gpu=best["gpu"], hybrid=best["hybrid"], **measured))
    result["limitations"] = ["NVFP4 runtime explicitly GPU-only; no invented ANE route", "no per-format equal arithmetic promise",
        "best among tested complete configurations, not exhaustive global optimum", "cold may include source hashing/native import and artifact setup",
        "initial whole-window model SHA; no parent per-inference payload hash; native import counters retain actual verification scope"]
    result["retained_failed_precheck"] = dict(evidence="outputs/z-base-formats-v1-precheck-20261011/summary.json",
        reason="ConvRot Public used Sylvester template; actual runtime failure/GPU fallback rejected; correct Comfy H256 template rechecked separately")
    if "/Users/" in json.dumps(result):
        raise ValueError("portable evidence leaked machine-specific paths")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--reverse", type=Path, required=True)
    parser.add_argument("--build-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh audit output required")
    value = audit(args.forward.resolve(), args.reverse.resolve(), args.build_manifest.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps(value["best_per_format"]))


if __name__ == "__main__":
    main()
