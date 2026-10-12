#!/usr/bin/env python3
"""Audit complete Z framework cohorts without opening model payloads."""
import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from flux9_framework_evidence import portable
from qwen21_mflux_framework_evidence import compare_rgba, rgba
from runtime_ane_common import sha256_file, validate_fp32_channel_join
from z_base_formats_evidence import RUNTIME_KEYS
from z_mflux_framework_screen import ROUTES, PRIVATE_SHARES, validate_runs

ROOT = Path(__file__).resolve().parents[2]
HYBRID_KEYS = ("ane_mlp_range", "bucket", "hidden", "mlp_width", "block_count",
               "compute_units", "observed_ane_residency", "weight_variant",
               "checkpoint_sha256_verified", "runtime_calls_session_total",
               "runtime_failures_session_total", "runtime_failed", "warmup_calls_session_total",
               "load_seconds", "manifest_validation_seconds")
MEMORY_KEYS = ("complete", "tree_peak_phys_footprint_bytes", "tree_peak_rss_bytes",
               "compression_bytes", "decompression_bytes", "swap_in_bytes", "swap_out_bytes")


def cohort(data):
    """Reject partial, reordered or shortened cohorts before trusting timings."""
    order = data.get("order", [])
    if (data.get("status") != "complete_diagnostic" or data.get("steps") != 8 or data.get("count") != 4 or
            not order or len(order) % 2 or order != list(reversed(order)) or
            any(route not in ROUTES or order.count(route) != 2 for route in order) or
            len(data.get("trials", [])) != len(order)):
        raise ValueError("complete eight-step two-process-per-route palindrome required")
    for index, trial in enumerate(data["trials"]):
        if trial["index"] != index or trial["route"] != order[index]:
            raise ValueError("actual route order differs")
        validate_runs(trial["report"], trial["route"], 8, 4)
        if trial["route"].startswith("mflux-"):
            if data.get("reference_cache_mib") is not None and trial["report"].get("reference_cache_limit_bytes") != data["reference_cache_mib"] << 20:
                raise ValueError("actual Z reference cache limit differs")
            coverage = trial["report"].get("required_weight_coverage")
            if not isinstance(coverage, dict) or set(coverage) != {"vae", "transformer", "text_encoder"} or any(
                    type(item.get("required_targets")) is not int or item["required_targets"] <= 0 or
                    item.get("covered_targets") != item["required_targets"] or item.get("missing") != []
                    for item in coverage.values()):
                raise ValueError("complete actual reference source parameter coverage required")
        if trial["route"] == "native-private":
            if data.get("private_channels") not in PRIVATE_SHARES or data.get("private_join") not in ("f16", "f32"):
                raise ValueError("known private channel/join candidate required")
            rows = [row["metrics"] for row in trial["report"]["runs"]]
            validate_fp32_channel_join(rows, data["private_join"] == "f32")
            previous_calls = previous_retries = previous_blocks = 0
            for row in rows:
                hybrid, runtime = row["hybrid"], row["hybrid"]["runtime_weight"]
                if (runtime["ane_channels"] != data["private_channels"] or
                        runtime["gpu_channels"] != 10240 - data["private_channels"] or
                        runtime["executor_backend"] != "private_ane" or runtime["fallback_blocks_session_total"] != 0):
                    raise ValueError("actual successful private channel complement differs")
                calls, retries, blocks = (hybrid["runtime_calls_session_total"],
                    runtime["overflow_retries_session_total"], runtime["channel_blocks_session_total"])
                if blocks - previous_blocks != 256 or calls - previous_calls != 256 + retries - previous_retries:
                    raise ValueError("actual complete channel work/retry accounting differs")
                if previous_blocks and retries != previous_retries:
                    raise ValueError("warm private headroom/retry policy changed")
                previous_calls, previous_retries, previous_blocks = calls, retries, blocks
    routes = []
    for route in sorted(set(order)):
        trials = [trial for trial in data["trials"] if trial["route"] == route]
        values = [row["request_wall_including_export"] for trial in trials for row in trial["report"]["runs"][1:3]]
        declared = data["medians"][route]
        if declared["samples"] != values or declared["fresh_warm_seconds"] != statistics.median(values):
            raise ValueError("declared median differs from all actual fresh samples")
        routes.append(dict(route=route, fresh_warm_seconds=values, fresh_warm_median=statistics.median(values),
            cold_seconds=[trial["report"]["runs"][0]["request_wall_including_export"] for trial in trials],
            repeat_seconds=[trial["report"]["runs"][3]["request_wall_including_export"] for trial in trials],
            constructor_seconds=[trial["report"]["constructor_seconds"] for trial in trials],
            physical_peaks=[trial["memory"]["verified"]["tree_peak_phys_footprint_bytes"] for trial in trials]))
    return routes


def audit(screen, layout, build_manifest, evidence_date="2026-10-11"):
    date.fromisoformat(evidence_date)
    data = json.loads((screen / "summary.json").read_text())
    routes = cohort(data)
    build = json.loads(build_manifest.read_text())
    if any(sha256_file(ROOT / row["path"]) != row["sha256"] for row in build["inputs"]["sources"]):
        raise ValueError("native build/source mismatch")
    library = build_manifest.parent.parent / "libturbocider.dylib"
    if sha256_file(library) != data["library_sha256"]:
        raise ValueError("native library artifact differs")
    receipt = json.loads((layout / "layout-receipt.json").read_text())
    if (receipt["family"] != "z" or receipt["status"] != "complete_layout_only" or
            sha256_file(layout / "layout-receipt.json") != data["layout_receipt_sha256"] or
            any(item["full_payload_equal"] is not True for item in receipt["files"])):
        raise ValueError("bound same-source complete lossless Z layout required")
    result = dict(schema="tc-z-valid-framework-cohort-v1", date=evidence_date, status="complete_diagnostic",
        goal_complete=False, qualification_passed=False, reference_latency_ranking_valid=True,
        hardware="M4 Max64GiB/macOS26.6.2", steps=8, requests_per_process=4, order=data["order"],
        native_inputs_matched=len(build["inputs"]["sources"]), library_sha256=data["library_sha256"],
        model_sha256=data["model_sha256"], mflux_commit=data["mflux_commit"],
        mflux_source_snapshot=data["mflux_source_snapshot"], private_channels=data["private_channels"],
        mflux_package_version=data.get("mflux_package_version"), reference_cache_mib=data.get("reference_cache_mib"),
        private_join=data["private_join"], source_policy=data["source_policy"], timing_scope=data["timing_scope"],
        representation_scope=data["representation_scope"],
        evidence=str((screen / "summary.json").relative_to(ROOT)), evidence_sha256=sha256_file(screen / "summary.json"),
        layout_proof=dict(receipt_sha256=data["layout_receipt_sha256"], files=[dict(kind=item["kind"],
            source_sha256=item["source_sha256"], derived_sha256=item["derived_sha256"],
            payload_sha256=item["payload_sha256"], full_payload_equal=item["full_payload_equal"],
            tensor_bindings=len(item["tensor_bindings"])) for item in receipt["files"]]),
        routes=routes, trials=[], quality=[], output_checks=[])
    images = {}
    for trial in data["trials"]:
        directory = screen / f'{trial["index"]}-{trial["route"]}'
        memory = trial["memory"]
        checked = memory["verified"]
        if checked["complete"] is not True or checked["command_exit_code"] != 0 or checked["reasons"]:
            raise ValueError("complete successful independent process-tree evidence required")
        for key in ("evidence", "report"):
            if sha256_file(directory / memory[key]) != memory[key + "_sha256"]:
                raise ValueError("memory evidence changed")
        if len(trial["pngs"]) != 4:
            raise ValueError("all four full PNG receipts required")
        for index, png in enumerate(trial["pngs"]):
            path = directory / f"{index}.png"
            value = rgba(path)
            if (sha256_file(path) != png["file_sha256"] or
                    hashlib.sha256(value[:, :, :3].tobytes()).hexdigest() != png["rgb_sha256"]):
                raise ValueError("actual PNG/RGB changed")
            images[trial["index"], index] = value
        report = trial["report"]
        published = {key: report[key] for key in ("engine", "constructor_seconds", "precision", "quantization",
            "packages", "required_weight_coverage", "text_embedding_dtype", "instrumentation", "compute_precision_name",
            "float32_stream", "transformer_floating_parameter_dtypes", "reference_cache_limit_bytes",
            "previous_reference_cache_limit_bytes") if key in report}
        published["runs"] = []
        for row in report["runs"]:
            actual = dict(index=row["index"], seconds=row["request_wall_including_export"])
            if trial["route"].startswith("mflux-"):
                actual.update(actual_steps=row["actual_steps"], actual_quantization=row["actual_quantization"],
                    loras=row["loras"], prompt_cache_hit=row["prompt_cache_hit"], mlx_peak_bytes=row["mlx_peak_bytes"])
                actual["generation_parameters"] = row.get("generation_parameters", {})
            else:
                metrics = row["metrics"]
                hybrid, runtime = metrics.get("hybrid") or {}, (metrics.get("hybrid") or {}).get("runtime_weight") or {}
                actual.update(actual_steps=metrics["actual_denoise_steps"], prompt_cache_hit=metrics["prompt_cache_hit"],
                    timings_seconds=metrics["timings_seconds"], gpu_graph=metrics["gpu_graph"],
                    hybrid={key: hybrid[key] for key in HYBRID_KEYS if key in hybrid},
                    runtime_weight={key: runtime[key] for key in (*RUNTIME_KEYS, "async_hybrid_blocks_session_total",
                        "overflow_events", "overflow_events_dropped_session_total") if key in runtime})
            published["runs"].append(actual)
        result["trials"].append(dict(index=trial["index"], route=trial["route"], report=published,
            controls=portable(trial["controls"]), memory={key: checked[key] for key in MEMORY_KEYS},
            memory_evidence_sha256=memory["evidence_sha256"], memory_report_sha256=memory["report_sha256"], pngs=trial["pngs"]))
    native = next((trial for trial in data["trials"] if trial["route"] == "native-gpu"), None)
    if native:
        for trial in data["trials"]:
            if trial is native:
                continue
            for index in range(4):
                result["quality"].append(dict(route=trial["route"], process_index=trial["index"], request_index=index,
                    comparison=compare_rgba(images[native["index"], index], images[trial["index"], index])))
    for route in sorted(set(data["order"])):
        left, right = [trial for trial in data["trials"] if trial["route"] == route]
        for index in range(4):
            result["output_checks"].append(dict(route=route, index=index,
                forward_reverse_rgba_exact=bool(np.array_equal(images[left["index"], index], images[right["index"], index]))))
    reference = next((trial for trial in data["trials"] if trial["route"] == "mflux-bf16"), None)
    result["reference_compute_quality"] = []
    if reference:
        for trial in data["trials"]:
            if trial["route"] == "mflux-fp16":
                for index in range(4):
                    result["reference_compute_quality"].append(dict(reference="mflux-bf16", candidate="mflux-fp16",
                        process_index=trial["index"], request_index=index,
                        comparison=compare_rgba(images[reference["index"], index], images[trial["index"], index])))
    medians = {route["route"]: route["fresh_warm_median"] for route in routes}
    if "native-gpu" in medians and "mflux-bf16" in medians:
        result["comparisons"] = dict(reference_bf16_over_native_gpu=medians["mflux-bf16"] / medians["native-gpu"])
        if "native-frozen" in medians:
            result["comparisons"].update(reference_bf16_over_native_frozen=medians["mflux-bf16"] / medians["native-frozen"],
                frozen_reduction_vs_native_gpu_percent=100 * (1 - medians["native-frozen"] / medians["native-gpu"]))
        if "mflux-fp16" in medians:
            result["comparisons"].update(reference_fp16_over_native_gpu=medians["mflux-fp16"] / medians["native-gpu"],
                reference_fp16_reduction_vs_bf16_percent=100 * (1 - medians["mflux-fp16"] / medians["mflux-bf16"]))
            if "native-frozen" in medians:
                result["comparisons"]["reference_fp16_over_native_frozen"] = medians["mflux-fp16"] / medians["native-frozen"]
    result["quality_scope"] = "all original-size RGBA/RGB with alpha and white-composited metrics; same seed not proof of equal noise; diagnostics, not broad perceptual or physical-NE qualification"
    if "/Users/" in json.dumps(result):
        raise ValueError("portable evidence contains machine-specific paths")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("screen", "layout", "build-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--date", default="2026-10-11", help="actual local completion/publication date")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh audit output required")
    result = audit(args.screen.resolve(), args.layout.resolve(), args.build_manifest.resolve(), args.date)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as target:
        json.dump(result, target, indent=2, allow_nan=False)
        target.write("\n")
    print(json.dumps(dict(status=result["status"], comparisons=result.get("comparisons"), routes=result["routes"])))


if __name__ == "__main__":
    main()
