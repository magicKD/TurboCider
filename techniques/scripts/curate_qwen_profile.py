#!/usr/bin/env python3
"""Read selected BF16 FFN host profiles; never mix staged spans into scores."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

PHASES = ("gate-up-projection", "silu-times-up", "down-projection")


def metric_pass(metric):
    value = metric.get("relative_l2")
    return metric.get("finite") is True and type(value) in (int,float) and math.isfinite(value) and 0 <= value <= .03


def summarize(document):
    if (document.get("schema") != "tc-qwen-bf16-host-profile-v1" or document.get("status") != "component-numeric-pass"
            or document.get("source_inputs_unchanged") is not True or document.get("checkpoint_stat_unchanged") is not True
            or document.get("physical_device_timestamps") is not None or document.get("native_counter_calibration_applied") is not False
            or document.get("synthetic") is not False or document.get("loras") != []):
        raise ValueError("source-stable real BF16 host-only report required")
    shape, abi = document.get("shape_mhf"), document.get("consumer_abi")
    if (not isinstance(shape,list) or len(shape)!=3 or any(type(value) is not int or value<=0 for value in shape)
            or not isinstance(abi,list) or any(type(value) is not int for value in abi)
            or abi != [1,shape[0],shape[1]] or document.get("source_dtype") != "bfloat16"
            or document.get("gates") != {"replica_relative_l2_max":.03,"fp64_sample_relative_l2_max":.03}):
        raise ValueError("original BF16 ABI and fixed numerical gates required")
    points = sorted({row["point"] for row in document["formal_samples"]})
    if not 1 <= len(points) <= 3 or {row["point"] for row in document["diagnostic_samples"]} != set(points):
        raise ValueError("matching explicit points required")
    result = {}
    for point in points:
        formal = [row for row in document["formal_samples"] if row["point"]==point]
        diagnostic = [row for row in document["diagnostic_samples"] if row["point"]==point]
        if [row["trial"] for row in formal] != [-2,-1,0,1,2,3] or [row["trial"] for row in diagnostic] != [0,1]:
            raise ValueError("complete ordered formal/warmup and diagnostic samples required")
        for row in formal + diagnostic:
            if row.get("component_numeric_pass") is not True or not all(metric_pass(row[key]) for key in ("replica","fp64_sample")):
                raise ValueError("failed numerical case cannot receive timing summary")
        for row in formal:
            if (row.get("formal_performance_score") is not True or row["warmup"] is not (row["trial"] < 0)
                    or row.get("source_kind") != "measured" or type(row["duration_ns"]) is not int or row["duration_ns"] <= 0):
                raise ValueError("invalid formal host output-ready sample")
        for row in diagnostic:
            events = row["host_events"]
            if row.get("formal_performance_score") is not False or tuple(event["name"] for event in events) != PHASES:
                raise ValueError("staged events must be complete and diagnostic-only")
            last = 0
            for event in events:
                if (event.get("source_kind") != "measured" or event.get("clock_domain") != "this-process-perf-counter-host-only"
                        or type(event.get("duration_ns")) is not int or event["duration_ns"] <= 0
                        or type(event.get("start_ns")) is not int or event["start_ns"] < last):
                    raise ValueError("invalid ordered same-process host event")
                last = event["start_ns"] + event["duration_ns"]
            if type(row.get("host_boundary_ns")) is not int or row["host_boundary_ns"] < last:
                raise ValueError("diagnostic events escape their observed host boundary")
        result[point] = {"source_kind": "derived", "formal_compiled_host_ready_median_ns": statistics.median(
            row["duration_ns"] for row in formal if not row["warmup"]),
            "staged_host_phase_median_ns": {phase: statistics.median(event["duration_ns"] for row in diagnostic
                for event in row["host_events"] if event["name"]==phase) for phase in PHASES},
            "staged_durations_are_not_compiled_kernel_costs": True, "speedup": None, "production_qualified": False}
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    try:
        if len({path.resolve() for path in args.reports}) != len(args.reports):
            raise ValueError("duplicate report")
        for path in args.reports:
            if path.stat().st_size > 32 * 1024 * 1024:
                raise ValueError("report exceeds adapter cap")
            document = json.loads(path.read_text(encoding="utf-8"))
            summary = summarize(document)
            for snapshot in document["source_snapshots"]:
                relative = Path(snapshot["path"])
                source = (path.parent / relative).resolve()
                if (relative.is_absolute() or ".." in relative.parts or not source.is_relative_to(path.parent.resolve())
                        or digest(source) != snapshot["sha256"]):
                    raise ValueError("missing, escaping or changed profile source snapshot")
            records.append({"id": path.parent.name, "raw_report_sha256": digest(path), "summary": summary,
                "report": {key: document[key] for key in (
                    "schema", "status", "synthetic", "production_qualified", "request_quality_qualified",
                    "physical_device_timestamps", "native_counter_calibration_applied", "loras", "source_snapshots",
                    "fixture_sha256", "formal_samples", "diagnostic_samples", "gates", "environment", "shape_mhf",
                    "source_layer", "source_dtype", "consumer_abi", "capture_receipt_sha256", "original_tensor_sha256",
                    "checkpoint_stat_unchanged", "source_inputs_unchanged")}})
        result = {"schema": "tc-qwen-bf16-host-profile-evidence-v1", "curator_sha256": digest(Path(__file__)),
                  "production_qualified": False, "runs": records}
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print("verified BF16 host profile evidence; no device timestamp or staged speedup inferred")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
