#!/usr/bin/env python3
"""Read explicit local FFN reports into portable evidence; never run a workload.

Preserves every sample, separates metric provenance and withholds statistics
for failed numerical arms. Bootstrap intervals with four pairs are exploratory,
not proof of product quality, physical placement or request acceleration.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_stats(document, expected_repeats=8):
    if document["status"] != "component_numeric_pass" or not all(sample.get("component_numeric_pass") is True for sample in document["samples"]):
        return None
    baseline = document.get("baseline", "gpu")
    warm = [sample for sample in document["samples"] if not sample["warmup"]]
    if len(warm) != expected_repeats or {sample["trial"] for sample in warm} != set(range(expected_repeats)):
        raise ValueError("incomplete or duplicated warm trials")
    paired = {}
    values = {baseline: [], "candidate": []}
    for sample in warm:
        value, variant = sample["wall_seconds"], sample["variant"]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0 or variant not in values:
            raise ValueError("invalid measured wall span or arm")
        arms = paired.setdefault(sample["pair"], {})
        if variant in arms:
            raise ValueError("duplicate paired arm")
        arms[variant] = sample
        values[variant].append(value)
    ratios = []
    for pair in sorted(paired):
        arms = paired[pair]
        if set(arms) != set(values) or arms[baseline]["point"] != arms["candidate"]["point"]:
            raise ValueError("incomplete or mismatched input pair")
        ratios.append(arms[baseline]["wall_seconds"] / arms["candidate"]["wall_seconds"])
    rng = random.Random(29)
    bootstrap = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(10000))
    return {"source_kind": "derived", "baseline": baseline, "warm_pairs": len(ratios),
            "median_wall_seconds": {key: statistics.median(value) for key, value in values.items()},
            "paired_speedups": ratios, "median_paired_speedup": statistics.median(ratios),
            "exploratory_bootstrap_95_percentile_interval": [bootstrap[249], bootstrap[9749]],
            "interval_method": "paired ratio median bootstrap, seed29, 10000 resamples; small n, not qualification"}


def local(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError("missing or escaping campaign asset")
    return path


def curate(root, runs, fixture, capture, environment=None, diagnostics=()):
    root = root.resolve()
    fixture_path, capture_path = local(root, fixture), local(root, capture)
    fixture_data = json.loads(fixture_path.read_text())
    capture_data = json.loads(capture_path.read_text())
    if capture_data.get("status") != "captured" or fixture_data["capture_receipt_sha256"] != digest(capture_path):
        raise ValueError("fixture/capture provenance mismatch")
    result = {"schema": "tc-real-ffn-tiling-evidence-v1", "date": "2026-10-15",
              "scope": "real full-shape FFN component and GPU consumer; not request quality or physical ANE proof",
              "production_qualified": False, "physical_ane_placement": "unverified",
              "semantics": "numerically equivalent within the common FP16 recipe; BF16->FP16 is a separately tested approximate conversion",
              "capture": {key: capture_data[key] for key in ("runtime_build_id", "cli_sha256", "library_sha256", "native_source_files_matched", "seed", "steps", "loras", "shape", "original_dtype", "completion_sha256")},
              "capture_receipt_sha256": digest(capture_path), "fixture_sha256": digest(fixture_path),
              "fixture": {key: fixture_data[key] for key in ("shape_mhf", "source_layer", "original_gate_up_tensor_sha256", "original_down_tensor_sha256", "points", "files", "original_reference_scope")},
              "curator_sha256": digest(Path(__file__)), "runs": [], "diagnostics": [],
              "field_provenance": {
                  "wall_seconds": {"source_kind": "measured", "unit": "seconds", "scope": "complete declared consumer"},
                  "prediction_host_seconds": {"source_kind": "measured", "unit": "seconds", "scope": "public prediction + actual output copy, not per-op device time"},
                  "input_staging_host_seconds": {"source_kind": "measured", "unit": "seconds"},
                  "post_prediction_restore_join_host_seconds": {"source_kind": "measured", "unit": "seconds", "scope": "includes any exposed GPU tail wait"},
                  "vs_fp32_same_fp16_source": {"source_kind": "derived", "scope": "whole output against FP32 MLX reference"},
                  "vs_independent_fp64_witness": {"source_kind": "derived", "scope": "sampled rows/columns, all reduction channels, CPU FP64 einsum"},
                  "vs_original_bf16_replica": {"source_kind": "derived", "scope": "whole output against original BF16 compiled GPU replica, not request quality"},
                  "compute_plan": {"source_kind": "predicted", "scope": "anticipated device/cost, null costs retained in raw plan"},
                  "host_events": {"source_kind": "measured", "scope": "CPU host timestamps only, never GPU/ANE hardware timestamps"}},
              "limitations": ["Only layer0 real activations replayed so far; 3 depths were captured.",
                              "Resident weights/interfaces; no weight streaming or LoRA qualification.",
                              "No global cache purge, PMU, Xcode capture or physical overlap proof.",
                              "The initial CPU-only diagnostic failed during conversion for disk space; a later retry and same-bundle policy diagnostic must be interpreted separately.",
                              "Initial runs had low disk headroom; post-cleanup runs record explicit disk admission.",
                              "Raw capture wall time is instrumented, not a normal request benchmark.",
                              "Original fixture weights, inputs and full reports stay local; no media or weights embedded here."]}
    if environment:
        path = local(root, environment)
        result["supplemental_environment"] = {"sha256": digest(path), "data": json.loads(path.read_text())}
    for group, name in runs:
        path = local(root, name + "/report.json")
        value = json.loads(path.read_text())
        if value["schema"] != "tc-coreml-tiling-experiment-v1" or value.get("fixture_sha256") != digest(fixture_path):
            raise ValueError("run schema or source fixture mismatch")
        selected = {key: value[key] for key in ("status", "shape_mhf", "coreml_geometry", "layout", "route", "prefix_channels", "runtime_weights",
                    "silu_lowering", "compute_units_policy", "baseline", "diagnostic_host_timeline", "disk_admission", "runner_sha256", "graphs_sha256",
                    "weight_sha256", "environment", "gates", "points", "samples", "boundary", "excluded", "cache_state", "recorded_order", "source_snapshots",
                    "conversion_host_seconds", "compile_host_seconds", "load_host_seconds", "compute_plan", "process_peak_rss_bytes") if key in value}
        errors = [{"phase": error["phase"], "type": error["type"],
                   "classification": "disk-space-exhausted" if "No space left" in error["message"] else
                   "fixed-numerical-gate-rejected" if "fixed numerical gate" in error["message"] else "other-failure-see-raw-report"}
                  for error in value["errors"]]
        plan = path.parent / "compute-plan.json"
        for snapshot in value.get("source_snapshots", []):
            if digest(local(path.parent.resolve(), snapshot["path"])) != snapshot["sha256"]:
                raise ValueError("retained source snapshot changed")
        plan_data = json.loads(plan.read_text()) if plan.is_file() else None
        plan_summary = None if plan_data is None else {
            "source_kind": "derived-from-predicted-plan", "physical_device_timing": None,
            "missing_cost_operations": sum(op["estimated_cost_weight"] is None for op in plan_data["operations"]),
            "operations": [{key: op[key] for key in ("operator", "preferred_device", "estimated_cost_weight")}
                           for op in plan_data["operations"] if op["operator"] != "const"]}
        result["runs"].append({"id": name, "group": group, "raw_report_sha256": digest(path),
                               "raw_plan_sha256": digest(plan) if plan.is_file() else None,
                               **selected, "source_snapshot_status": "retained-and-end-checked" if value.get("source_snapshots") else "early-exploration-hash-only-not-full-source-snapshot",
                               "plan_summary": plan_summary, "errors": errors, "paired_statistics": paired_stats(value)})
    for name in diagnostics:
        path=local(root,name+"/report.json")
        value=json.loads(path.read_text())
        snapshot=value.get("source_snapshot")
        if snapshot:
            source=(path.parent/snapshot["path"]).resolve()
            if not source.is_relative_to(path.parent) or digest(source)!=snapshot["sha256"]:raise ValueError("diagnostic source changed")
        if value.get("schema")=="tc-coreml-same-bundle-policy-v1" and value.get("status")=="collected":
            if (value["compiled_assets_before"]!=value["compiled_assets_after"] or not value["compiled_assets_before"]
                    or [call["compute_units_policy"] for call in value["calls"]]!=["CPU_ONLY","CPU_AND_NE","CPU_AND_NE","CPU_ONLY"]
                    or not all(call.get("input_unchanged_after_call") is True and call["timing_qualified"] is False for call in value["calls"])):
                raise ValueError("same-bundle policy control is inconsistent")
            for call in value["calls"]:
                error=call["sampled_fp64"]["relative_l2"]
                if call["component_sample_gate_pass"] is not (error is not None and error<=.02):
                    raise ValueError("policy diagnostic gate marker inconsistent")
        result["diagnostics"].append({"id":name,"raw_report_sha256":digest(path),"data":value})
        result["continuation_dates"]=["2026-10-16"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fixture", required=True, help="relative fixture.json")
    parser.add_argument("--capture", required=True, help="relative capture receipt.json")
    parser.add_argument("--environment", help="explicit relative sanitized environment addendum")
    parser.add_argument("--diagnostic", action="append", default=[], help="explicit relative correctness diagnostic directory")
    parser.add_argument("--run", action="append", required=True, help="group:relative-run-directory")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    runs = []
    for value in args.run:
        group, separator, name = value.partition(":")
        if not separator or group not in ("discovery", "confirmation", "diagnostic", "resource-failure") or not re.fullmatch(r"[a-z0-9-]+", name):
            parser.error("use explicit group:run-name")
        runs.append((group, name))
    if len({name for _, name in runs}) != len(runs):
        parser.error("duplicate run")
    print(json.dumps(curate(args.root, runs, args.fixture, args.capture, args.environment, args.diagnostic), indent=None if args.compact else 2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
