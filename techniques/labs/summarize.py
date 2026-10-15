#!/usr/bin/env python3
"""Describe ONE lab report at a time; never merge unmatched device campaigns."""
import argparse
import json
import math
from pathlib import Path
import statistics

from common import emit


def summarize(document):
    groups = {}
    if document.get("schema") == "turbocider-metal-teaching-lab-v1":
        if not document.get("actual_metal_dispatch") or not document.get("all_outputs_finite") or not document.get("tail_guards_intact"):
            raise ValueError("not a successful validated actual Metal report")
        samples = document["samples"]
        context = {key: document.get(key) for key in ("case", "device", "os", "shape", "threads", "source_sha256", "scope")}
        field = "wall_seconds"
    elif document.get("schema") == "turbocider-technique-lab-v1" and document.get("lab_id") == "coreml-ffn":
        details = document["details"]
        samples = details["samples"]
        context = {key: details.get(key) for key in ("kind", "shape", "compute_units_policy", "coremltools", "runner_sha256")}
        context["scope"] = document["scope"]
        field = "request_boundary_seconds"
    else:
        raise ValueError("expected actual Metal/CoreML timing report; integration process wall is not operator latency")
    if not samples:
        raise ValueError("empty timing report")
    pairs = {}
    for sample in samples:
        value = sample[field]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("invalid positive finite measured boundary")
        groups.setdefault(sample["variant"], []).append(value)
        if "pair" in sample and len(set(row["variant"] for row in samples)) == 2:
            row = pairs.setdefault(sample["pair"], {})
            if sample["variant"] in row:
                raise ValueError("duplicate paired arm")
            row[sample["variant"]] = sample.get("input_generation")
    if len(groups) == 2:
        if not pairs or any(set(row) != set(groups) or len(set(row.values())) != 1 for row in pairs.values()):
            raise ValueError("unmatched inputs or incomplete paired arms")
    return {"schema": "turbocider-lab-descriptive-summary-v1", "context": context, "measured_boundary": field,
            "variant_median_ms": {key: statistics.median(rows) * 1000 for key, rows in groups.items()},
            "variant_worst_ms": {key: max(rows) * 1000 for key, rows in groups.items()},
            "variant_samples": {key: len(rows) for key, rows in groups.items()},
            "matched_pairs": len(pairs), "production_qualified": False,
            "warnings": ["Descriptive tiny-operator observations only; no automatic significance, E2E or physical ANE claim.",
                         "Do not multiply these times/ratios with unrelated caches, kernels or historical model campaigns."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.report.open(encoding="utf-8") as stream:
        emit(summarize(json.load(stream)), args.output)


if __name__ == "__main__":
    main()
