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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--trial", type=int, required=True)
    args = parser.parse_args()
    try:
        raw = args.report.read_bytes()
        result = convert(json.loads(raw), args.trial)
        result["metadata"]["raw_report_sha256"] = hashlib.sha256(raw).hexdigest()
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
