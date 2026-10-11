#!/usr/bin/env python3
"""Compact published Qwen timing/coverage record; full raw receipts stay bound."""
import argparse
import json
from pathlib import Path
import statistics

from runtime_ane_common import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def compact(full):
    if full["status"] != "complete_diagnostic" or full.get("reference_latency_ranking_valid") is not True or full.get("invalid_reference_reason"):
        raise ValueError("only a complete valid reference cohort can enter this publication")
    expected = full["order"]
    known = {"native-gpu", "native-frozen", "mflux-bf16", "mflux-q4", "mflux-q8"}
    if (full["steps"] != 40 or full["requests_per_process"] != 4 or expected != list(reversed(expected)) or
            not {"native-gpu", "native-frozen", "mflux-bf16"}.issubset(expected) or
            any(route not in known or expected.count(route) != 2 for route in expected) or
            len(full["trials"]) != len(expected)):
        raise ValueError("complete native GPU/frozen/reference palindrome required")
    retained = ("date", "hardware", "steps", "requests_per_process", "order", "evidence", "evidence_sha256",
                "native_inputs_matched", "library_sha256", "model_sha256", "mflux_commit", "mflux_source_snapshot",
                "layout_proof", "timing_scope", "representation_scope", "source_policy", "routes", "quality", "quality_scope")
    result = {key: full[key] for key in retained}
    result.update(schema="tc-qwen-valid-framework-cohort-v1", status="complete_diagnostic", qualification_passed=False,
                  goal_complete=False, reference_latency_ranking_valid=True,
                  reference_cache_mib=full.get("reference_cache_mib"), trials=[])
    hybrid_keys = ("ane_mlp_range", "block_count", "bucket", "hidden", "mlp_width", "checkpoint_sha256_verified",
                   "compute_units", "observed_ane_residency", "runtime_calls_session_total", "runtime_failures_session_total",
                   "runtime_failed", "warmup_calls_session_total", "weight_variant", "load_seconds", "manifest_validation_seconds")
    for index, trial in enumerate(full["trials"]):
        if trial["route"] != expected[index] or trial["memory"]["complete"] is not True or len(trial["report"]["runs"]) != 4:
            raise ValueError("actual request/independent memory cohort differs")
        report = trial["report"]
        summary = {key: report[key] for key in ("engine", "constructor_seconds", "precision", "quantization", "packages",
            "required_weight_coverage", "text_embedding_dtype", "transformer_input_weight_dtype", "instrumentation",
            "reference_cache_limit_bytes", "previous_reference_cache_limit_bytes") if key in report}
        runs = []
        for request, row in enumerate(report["runs"]):
            if trial["route"].startswith("mflux-"):
                bits = {"mflux-bf16": None, "mflux-q4": 4, "mflux-q8": 8}[trial["route"]]
                if (row["actual_steps"] != list(range(40)) or row["loras"] or row["actual_quantization"] != bits or
                        row["prompt_cache_hit"] is not (request == 3)):
                    raise ValueError("actual original40-step base work required")
                coverage = report["required_weight_coverage"]
                if set(coverage) != {"vae", "text_encoder", "transformer"} or any(
                        type(value["required_targets"]) is not int or value["required_targets"] <= 0 or
                        value["missing"] or value["required_targets"] != value["covered_targets"] for value in coverage.values()):
                    raise ValueError("incomplete reference source parameters")
                runs.append(dict(index=request, seconds=row["request_wall_including_export"], actual_steps=40,
                    actual_quantization=row["actual_quantization"], prompt_cache_hit=row["prompt_cache_hit"], mlx_peak_bytes=row["mlx_peak_bytes"]))
            else:
                metrics = row["metrics"]
                hybrid = metrics.get("hybrid")
                runs.append(dict(index=request, seconds=row["request_wall_including_export"], timings_seconds=metrics["timings_seconds"],
                    actual_steps=metrics["actual_denoise_steps"], prompt_cache_hit=metrics["prompt_cache_hit"],
                    encoder_weight_residency=metrics.get("encoder_weight_residency"),
                    hybrid={key: hybrid[key] for key in hybrid_keys if key in hybrid} if hybrid else None))
        summary["runs"] = runs
        result["trials"].append(dict(route=trial["route"], index=index, report=summary, memory=trial["memory"],
            memory_evidence_sha256=trial["memory_evidence_sha256"], memory_report_sha256=trial["memory_report_sha256"], pngs=trial["pngs"]))
    medians = {row["route"]: row["fresh_warm_median"] for row in result["routes"]}
    for route in result["routes"]:
        actual = [row["seconds"] for trial in result["trials"] if trial["route"] == route["route"] for row in trial["report"]["runs"][1:3]]
        if actual != route["fresh_warm_seconds"] or statistics.median(actual) != route["fresh_warm_median"]:
            raise ValueError("actual complete pooled fresh samples differ")
    result["comparisons"] = dict(reference_bf16_over_native_gpu=medians["mflux-bf16"] / medians["native-gpu"],
        reference_bf16_over_native_frozen=medians["mflux-bf16"] / medians["native-frozen"],
        gpu_reduction_vs_reference_percent=100 * (1 - medians["native-gpu"] / medians["mflux-bf16"]),
        frozen_reduction_vs_reference_percent=100 * (1 - medians["native-frozen"] / medians["mflux-bf16"]),
        frozen_reduction_vs_native_gpu_percent=100 * (1 - medians["native-frozen"] / medians["native-gpu"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh valid-framework publication required")
    result = compact(json.loads(args.audit.read_text()))
    result.update(full_audit=str(args.audit.resolve().relative_to(ROOT)), full_audit_sha256=sha256_file(args.audit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as target:
        json.dump(result, target, indent=2, allow_nan=False); target.write("\n")
    print(json.dumps(result["comparisons"]))


if __name__ == "__main__":
    main()
