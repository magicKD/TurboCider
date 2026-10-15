#!/usr/bin/env python3
"""Export ONE explicit FFN trial to Chrome Trace/Perfetto host events.

No device timestamps are invented. Relative CPU spans within this trial share
one perf_counter domain; never merge unrelated process clocks or call a host
prediction span an ANE kernel span. No viewer, network or workload is launched.
"""
import argparse
import hashlib
import json
from pathlib import Path


def convert(document, trial):
    if document.get("schema") != "tc-coreml-tiling-experiment-v1" or type(trial) is not int:
        raise ValueError("explicit FFN experiment and integer trial required")
    samples = [sample for sample in document["samples"] if sample["trial"] == trial]
    if len(samples) != 1 or not samples[0].get("host_events"):
        raise ValueError("unique trial has no recorded diagnostic host events")
    sample = samples[0]
    trace = [{"name": "thread_name", "ph": "M", "pid": 1, "tid": 1, "args": {"name": "CPU owner host calls"}},
             {"name": "thread_name", "ph": "M", "pid": 1, "tid": 2, "args": {"name": "CPU prediction worker calls"}}]
    for event in sample["host_events"]:
        if (event.get("source_kind") != "measured" or event.get("clock_domain") != "this-process-perf-counter-host-only"
                or any(type(event.get(key)) is not int or event[key] < 0 for key in ("start_ns", "duration_ns"))):
            raise ValueError("invalid or non-host timestamp provenance")
        worker = event["name"].startswith("coreml-predict") and sample["variant"] == "candidate" and document["route"] == "hybrid"
        trace.append({"name": event["name"], "cat": "cpu-host-only", "ph": "X", "pid": 1, "tid": 2 if worker else 1,
                      "ts": event["start_ns"] / 1000, "dur": event["duration_ns"] / 1000,
                      "args": {"variant": sample["variant"], "point": sample["point"], "component_numeric_pass": sample["component_numeric_pass"]}})
    return {"traceEvents": trace, "displayTimeUnit": "ms", "metadata": {
        "scope": "one trial, CPU host events only; no GPU/ANE execution timestamps or physical overlap proof",
        "clock_domain": "selected trial perf_counter-relative; trace ts/dur in microseconds",
        "trial": trial, "warmup": sample["warmup"], "runner_sha256": document["runner_sha256"],
        "pid_and_tid": "normalised host lanes, not operating-system identifiers"}}


def convert_qwen(document, point, trial):
    if (document.get("schema") != "tc-qwen-bf16-host-profile-v1" or not isinstance(point,str) or type(trial) is not int
            or document.get("physical_device_timestamps") is not None or document.get("native_counter_calibration_applied") is not False):
        raise ValueError("explicit point/trial and host-only Qwen profile required")
    matches = [sample for sample in document["diagnostic_samples"] if sample["point"]==point and sample["trial"]==trial]
    if len(matches)!=1 or matches[0].get("formal_performance_score") is not False:
        raise ValueError("unique diagnostic host sample required")
    sample = matches[0]
    events = sample["host_events"]
    if [event["name"] for event in events] != ["gate-up-projection","silu-times-up","down-projection"]:
        raise ValueError("complete three-phase FFN host timeline required")
    trace, end = [], 0
    for event in events:
        if (event.get("source_kind") != "measured" or event.get("clock_domain") != "this-process-perf-counter-host-only"
                or type(event.get("start_ns")) is not int or type(event.get("duration_ns")) is not int
                or event["start_ns"] < end or event["duration_ns"] <= 0):
            raise ValueError("invalid ordered host events")
        end = event["start_ns"]+event["duration_ns"]
        trace.append({"name":event["name"], "cat":"cpu-host-diagnostic", "ph":"X", "pid":1, "tid":1,
            "ts":event["start_ns"]/1000, "dur":event["duration_ns"]/1000,
            "args":{"point":point,"trial":trial,"component_numeric_pass":sample["component_numeric_pass"],
                    "formal_performance_score":False}})
    if end > sample["host_boundary_ns"]:
        raise ValueError("events exceed observed host boundary")
    return {"traceEvents":trace,"displayTimeUnit":"ms","metadata":{"point":point,"trial":trial,
        "formal_performance_score":False,"production_qualified":False,
        "scope":"one staged real BF16 FFN; host calls with explicit drain, not compiled GPU kernel costs",
        "clock_domain":"selected trial perf_counter-relative; trace ts/dur in microseconds",
        "physical_device_timestamps":None,"native_counter_calibration_applied":False}}


def select_report(document, run):
    if document.get("schema") == "tc-qwen-bf16-host-profile-evidence-v1":
        if not isinstance(run,str):
            raise ValueError("curated evidence requires an explicit --run; never merge process clocks")
        selected = [record for record in document["runs"] if record["id"]==run]
        if len(selected)!=1:
            raise ValueError("unique selected profile run required")
        declared_hash = selected[0]["raw_report_sha256"]
        if not isinstance(declared_hash,str) or len(declared_hash)!=64 or any(ch not in "0123456789abcdef" for ch in declared_hash):
            raise ValueError("curated source hash must be an explicit SHA256")
        return selected[0]["report"], declared_hash
    if run is not None:
        raise ValueError("--run applies only to the curated Qwen evidence capsule")
    return document, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--point", help="required for real BF16 host profile; never infer across points")
    parser.add_argument("--run", help="required when reading a curated Qwen evidence capsule")
    args = parser.parse_args()
    try:
        raw = args.report.read_bytes()
        document, declared_source_hash = select_report(json.loads(raw), args.run)
        if document.get("schema") == "tc-qwen-bf16-host-profile-v1":
            result = convert_qwen(document, args.point, args.trial)
        else:
            if args.point is not None:
                raise ValueError("--point only applies to a Qwen BF16 host profile")
            result = convert(document, args.trial)
        if declared_source_hash:
            result["metadata"].update(curated_input_sha256=hashlib.sha256(raw).hexdigest(),
                                       declared_source_raw_report_sha256=declared_source_hash,
                                       declared_source_hash_not_recomputed=True, selected_run=args.run)
        else:
            result["metadata"]["raw_report_sha256"] = hashlib.sha256(raw).hexdigest()
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
