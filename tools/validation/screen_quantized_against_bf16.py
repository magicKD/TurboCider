#!/usr/bin/env python3
"""Matched warm speed/memory screen against optimized BF16; never qualification."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from screen_gguf_memory_budgets import positive, sha


REVISION = "fastest-bf16-low-memory-screen-v2"
PRIVATE_CONTROLS = {"TURBOCIDER_Z_GGUF_IMPORT", "TURBOCIDER_Z_GGUF_PACKED_WEIGHT_LIMIT_BYTES",
                    "TURBOCIDER_Z_GGUF_COMPILE_PACKED", "TURBOCIDER_Z_GGUF_RETAIN_PACKED"}


def content_identity(report):
    components = report.get("component_binding")
    if not isinstance(components, dict) or not all(components.get(role) for role in ("encoder", "vae", "tokenizer")):
        raise ValueError("requires actual content-bound encoder/VAE/tokenizer")
    for files in components.values():
        for entry in files:
            sha(entry["sha256"]); positive(entry["bytes"], "component bytes")
    libraries = {name: sha(value["sha256"]) for name, value in report["loaded_libraries"].items()}
    if libraries.get("libturbocider.dylib") != sha(report["library_sha256"]) or "libmlx.dylib" not in libraries:
        raise ValueError("actual library binding missing/mismatched")
    environment = report.get("runtime_environment")
    if not isinstance(environment, dict):
        raise ValueError("missing runtime environment")
    hardware = report.get("hardware")
    device = None
    if hardware is not None:
        if not isinstance(hardware, dict) or not all(hardware.get(k) for k in ("gpu", "os", "mlx_version")) or hardware["gpu"]=="unavailable":
            raise ValueError("invalid actual device binding")
        positive(hardware["physical_memory_bytes"], "physical memory")
        device = {k: hardware[k] for k in ("gpu", "physical_memory_bytes", "os", "mlx_version")}
    return {"components": components, "libraries": libraries, "device": device,
            "cache": report["prompt_cache_policy"],
            "gpu_controls": {k: v for k, v in environment.items() if k not in PRIVATE_CONTROLS}}


def workload(row):
    r, m = row["request"], row["metrics"]
    if r["execution"]["policy"] != "gpu" or m.get("hybrid"):
        raise ValueError("this screen requires pure GPU arms")
    if "dump_tensors" in r or row.get("cancellation_triggered"):
        raise ValueError("diagnostics/cancellation cannot be a speed sample")
    return {"inputs": r["inputs"], "sampling": r["sampling"], "operation": r["operation"],
            "parameters": {k: v for k, v in r["parameters"].items() if k != "compile_gpu"},
            "outputs": [{k: v for k, v in out.items() if k != "path"} for out in r["outputs"]],
            "tokens": m["text_tokens"]}


def collect(timing, memory, baseline, min_samples):
    shared, fixed_workload, recipe, png = None, None, None, None
    times, diffusion = [], []
    peak, gap = 0, 0.
    source = None
    for mode, report in (("timing", timing), ("memory", memory)):
        if report.get("status") != "completed" or report.get("measurement") != mode or not report.get("runs"):
            raise ValueError("requires completed separate timing/memory reports")
        if report.get("baseline_bf16") is not baseline:
            raise ValueError("wrong BF16/candidate role")
        identity = content_identity(report)
        if shared is not None and identity != shared:
            raise ValueError("component/library/cache/GPU tuning changed")
        shared = identity
        report_source = sha(report.get("dit_source_sha256"))
        if source is not None and source != report_source:
            raise ValueError("arm source changed between timing/memory reports")
        source = report_source
        for row in report["runs"]:
            if row.get("status") != 0 or row.get("error") or row.get("measurement") != mode:
                raise ValueError("failed/mislabeled sample")
            if type(row.get("warmup")) is not bool:
                raise ValueError("invalid warmup marker")
            w = workload(row)
            if fixed_workload is not None and w != fixed_workload:
                raise ValueError("workload/token geometry changed")
            fixed_workload = w
            m = row["metrics"]
            if baseline and (m.get("runtime_precision") != "bf16" or
                             m.get("runtime_backend") != "mlx_cpp_metal" or
                             m.get("plan", {}).get("gpu_graph") != "compiled_fused_blocks"):
                raise ValueError("baseline is not compiled pure GPU BF16")
            q = m.get("quantized_execution") or m.get("gguf_import")
            current = {"config": row["request"]["execution"].get("quantized_execution"),
                       "import": report.get("native_import"), "controls": report.get("native_import_environment"),
                       "layout": q.get("layout_digest", q.get("plan_digest")) if q else None,
                       "consumer": q.get("consumer_revision") if q else None,
                       "graph": q.get("gpu_graph_recipe") if q else None,
                       "retention": q.get("session_packed_retention") if q else None}
            if not baseline and not q:
                raise ValueError("candidate has no bound quantized execution/import")
            if recipe is not None and current != recipe:
                raise ValueError("candidate math/layout/lifecycle changed")
            recipe = current
            if q:
                sha(current["layout"])
                if source != sha(q["source_sha256"]):
                    raise ValueError("candidate source changed")
            if identity["cache"] == "miss" and m.get("prompt_cache_hit") is not False:
                raise ValueError("miss sample reused conditioning")
            if not row["warmup"] and identity["cache"] == "hit" and m.get("prompt_cache_hit") is not True:
                raise ValueError("hit samples require explicit warmup")
            current_png = sha(row["png_sha256"])
            if png is not None and png != current_png:
                raise ValueError("same-arm PNG changed")
            png = current_png
            if mode == "timing":
                if row.get("memory_samples"):
                    raise ValueError("timing contaminated by memory observer")
                if not row["warmup"]:
                    times.append(positive(row["wall_seconds"], "wall time"))
                    diffusion.append(positive(m["timings_seconds"]["denoise"], "diffusion time"))
            else:
                samples = row.get("memory_samples")
                if not samples or row.get("sampling_errors"):
                    raise ValueError("missing/failed memory observer")
                intervals = [b["time"]-a["time"] for a,b in zip(samples,samples[1:])]
                if not intervals or any(not math.isfinite(value) or value <= 0 for value in intervals):
                    raise ValueError("nonmonotonic/invalid memory sample timestamps")
                actual_gap = max(intervals)
                recorded = positive(row["memory_sample_max_gap_seconds"], "memory gap")
                if abs(recorded-actual_gap) > 1e-6 or actual_gap > .05 or actual_gap <= 0:
                    raise ValueError("invalid/inadequate memory sampling gap")
                if row.get("vm_deltas", {}).get("Swapouts") != 0:
                    raise ValueError("swap observed/unknown")
                gap = max(gap, actual_gap)
                peak = max(peak, *(positive(s[k], "process footprint") for s in samples
                           for k in ("phys_footprint_bytes", "lifetime_max_phys_footprint_bytes")))
    if len(times) < min_samples or not peak:
        raise ValueError("insufficient speed/memory evidence")
    return shared, fixed_workload, {"samples_seconds": times, "median_seconds": statistics.median(times),
        "denoise_median_seconds": statistics.median(diffusion), "observed_peak_bytes": peak,
        "maximum_memory_gap_seconds": gap, "recipe": recipe, "source_sha256": source, "png_sha256": png}


def screen(bt, bm, ct, cm, min_samples=4):
    if type(min_samples) is not int or min_samples < 2:
        raise ValueError("at least two screening samples required")
    bi, bw, b = collect(bt, bm, True, min_samples)
    ci, cw, c = collect(ct, cm, False, min_samples)
    if bi != ci or bw != cw:
        raise ValueError("BF16/candidate binary/components/workload/cache/GPU tuning differ")
    ratio = c["median_seconds"] / b["median_seconds"]
    memory_ratio = c["observed_peak_bytes"] / b["observed_peak_bytes"]
    device_bound = bi["device"] is not None
    return {"schema": REVISION, "production_qualified": False, "whole_request_memory": "unknown",
        "formal_performance": "not_run", "quality": "not_run", "baseline_fastest_qualification": "screen_only",
        "identity": bi, "workload": bw, "baseline": b, "candidate": c,
        "warm_wall_ratio": ratio, "observed_process_memory_ratio": memory_ratio,
        "preferred_10pct_latency_pass": ratio <= 1.10, "hard_20pct_latency_pass": ratio <= 1.20,
        "observed_lower_memory": memory_ratio < 1,
        "device_binding": "recorded" if device_bound else "unknown_legacy_reports",
        "empirical_speed_memory_target_pass": device_bound and ratio <= 1.20 and memory_ratio < 1,
        "scope": "paired local warm screen; no formal fastest/quality/cold/whole-request certification"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-timing", "baseline-memory", "candidate-timing", "candidate-memory"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--min-samples", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.output.is_symlink():
        p.error("output already exists")
    paths = [a.baseline_timing, a.baseline_memory, a.candidate_timing, a.candidate_memory]
    result = screen(*(json.loads(path.read_text()) for path in paths), min_samples=a.min_samples)
    result["raw_reports"] = [{"id": path.parent.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
    result["verifier_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print("wall ratio", result["warm_wall_ratio"], "memory ratio", result["observed_process_memory_ratio"],
          "screen target", result["empirical_speed_memory_target_pass"])


if __name__ == "__main__":
    main()
