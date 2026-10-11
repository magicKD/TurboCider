#!/usr/bin/env python3
"""Publish valid native results without using invalid reference noise as a rank."""
import argparse
import json
from pathlib import Path
import statistics

from runtime_ane_common import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def summarize(full):
    if full["status"] != "complete_native_diagnostic_invalid_reference" or full["reference_latency_ranking_valid"] is not False:
        raise ValueError("explicitly retained invalid reference cohort required")
    trials = [trial for trial in full["trials"] if trial["route"].startswith("native-")]
    if [trial["route"] for trial in trials] != ["native-gpu", "native-frozen", "native-frozen", "native-gpu"]:
        raise ValueError("actual complete native ABBA required")
    result = {key: full[key] for key in ("date", "hardware", "model_sha256", "library_sha256", "native_inputs_matched", "evidence", "evidence_sha256")}
    result.update(schema="tc-qwen-native-base-cohort-v1", status="complete_native_cohort", qualification_passed=False,
                  goal_complete=False, steps=40, geometry=[512, 512], seed=29, trials=[], routes=[],
                  scope="native original BF16 base GPU versus source-bound W8A8 Core ML W6144; request-local GPU encoder; native/API full PNG wall",
                  invalid_reference_reason=full["invalid_reference_reason"], reference_latency_ranking_valid=False)
    hybrid_keys = ("ane_mlp_range", "block_count", "bucket", "hidden", "mlp_width", "checkpoint_sha256_verified",
                   "compute_units", "observed_ane_residency", "runtime_calls_session_total", "runtime_failures_session_total",
                   "runtime_failed", "warmup_calls_session_total", "weight_variant", "load_seconds", "manifest_validation_seconds")
    for trial in trials:
        if trial["memory"]["complete"] is not True or len(trial["report"]["runs"]) != 4:
            raise ValueError("four complete native requests and memory evidence required")
        runs = []
        for index, row in enumerate(trial["report"]["runs"]):
            metrics = row["metrics"]
            if metrics["actual_denoise_steps"] != 40 or metrics["prompt_cache_hit"] is not (index == 3):
                raise ValueError("actual native steps/cache differ")
            hybrid = metrics.get("hybrid")
            if trial["route"] == "native-frozen" and (not hybrid or hybrid["runtime_calls_session_total"] != (index + 1) * 1248 or
                    hybrid["runtime_failed"] is not False or hybrid["checkpoint_sha256_verified"] is not True):
                raise ValueError("actual checkpoint-bound successful frozen work required")
            runs.append(dict(index=index, request_wall_including_export=row["request_wall_including_export"],
                timings_seconds=metrics["timings_seconds"], prompt_cache_hit=metrics["prompt_cache_hit"],
                encoder_weight_residency=metrics.get("encoder_weight_residency"),
                hybrid={key: hybrid[key] for key in hybrid_keys if key in hybrid} if hybrid else None))
        result["trials"].append(dict(route=trial["route"], index=trial["index"], constructor_seconds=trial["report"]["constructor_seconds"],
            runs=runs, memory=trial["memory"], memory_evidence_sha256=trial["memory_evidence_sha256"],
            memory_report_sha256=trial["memory_report_sha256"], pngs=trial["pngs"]))
    for route in ("native-gpu", "native-frozen"):
        selected = [trial for trial in result["trials"] if trial["route"] == route]
        fresh = [row["request_wall_including_export"] for trial in selected for row in trial["runs"][1:3]]
        result["routes"].append(dict(route=route, cold_seconds=[trial["runs"][0]["request_wall_including_export"] for trial in selected],
            fresh_warm_seconds=fresh, fresh_warm_median=statistics.median(fresh),
            repeat_seconds=[trial["runs"][3]["request_wall_including_export"] for trial in selected],
            physical_peaks=[trial["memory"]["tree_peak_phys_footprint_bytes"] for trial in selected]))
    a, b = (route["fresh_warm_median"] for route in result["routes"])
    result.update(native_gpu_over_frozen=a / b, frozen_reduction_percent=100 * (1 - b / a),
                  quality=[record for record in full["quality"] if record["route"] == "native-frozen"],
                  quality_scope=full["quality_scope"],
                  invalid_reference_diagnostics=[dict(route=route["route"], fresh_seconds=route["fresh_warm_seconds"],
                      diagnostic_median=route["fresh_warm_median"], ranking_valid=False)
                      for route in full["routes"] if route["route"].startswith("mflux-")])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh native cohort evidence required")
    result = summarize(json.loads(args.audit.read_text()))
    result.update(full_audit_evidence=str(args.audit.resolve().relative_to(ROOT)), full_audit_sha256=sha256_file(args.audit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as target:
        json.dump(result, target, indent=2, allow_nan=False)
        target.write("\n")
    print(json.dumps(dict(status=result["status"], reduction_percent=result["frozen_reduction_percent"])))


if __name__ == "__main__":
    main()
