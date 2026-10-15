#!/usr/bin/env python3
"""Summarize a pre-normalized matched campaign, never infer hardware placement."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def summarize(document):
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if document.get("scope") not in ("request", "phase", "operator"):
        raise ValueError("scope must explicitly name request, phase or operator")
    if type(document.get("synthetic", False)) is not bool:
        raise ValueError("synthetic must be a boolean when supplied")
    identity = document.get("identity", {})
    required = ("campaign", "hardware", "software", "workload", "weights", "adapter", "cache_policy", "precision_policy")
    if not isinstance(identity, dict) or any(not isinstance(identity.get(key), str) or not identity[key].strip() for key in required):
        raise ValueError("identity needs nonempty string fields: " + ", ".join(required))
    samples = document.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("samples must be a nonempty list")
    builds, seen, warm, cold, order = {}, set(), {}, {}, []
    failures = 0
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("each sample must be an object")
        if set(sample) - {"pair", "variant", "build", "phase", "status", "quality", "wall_seconds", "notes"}:
            raise ValueError("unexpected sample fields; normalize and verify one common campaign identity before comparing")
        variant, build = sample.get("variant"), sample.get("build")
        if variant not in ("baseline", "candidate") or not isinstance(build, str) or not build:
            raise ValueError("variant must be baseline/candidate with a build identity")
        if variant in builds and builds[variant] != build:
            raise ValueError("mixed builds inside one variant")
        builds[variant] = build
        pair = sample.get("pair")
        phase = sample.get("phase")
        if not isinstance(pair, str) or not pair or phase not in ("cold", "warm"):
            raise ValueError("each sample needs a nonempty pair and cold/warm phase")
        key = (pair, phase, variant)
        if key in seen:
            raise ValueError("duplicate pair/phase/variant: " + repr(key))
        seen.add(key)
        if sample.get("status") not in ("ok", "error", "cancelled") or sample.get("quality") not in ("pass", "fail", "unknown"):
            raise ValueError("explicit status and quality required")
        if not positive(sample.get("wall_seconds")):
            raise ValueError("all observed durations must be finite positive seconds")
        # Do not turn failed attempts into a faster successful-only campaign.
        if sample["status"] != "ok" or sample["quality"] != "pass":
            failures += 1
        target = warm if phase == "warm" else cold
        target.setdefault(pair, {})[variant] = sample["wall_seconds"]
        order.append("A" if variant == "baseline" else "B")
    if len(builds) != 2 or not warm or any(set(pair) != {"baseline", "candidate"} for pair in warm.values()):
        raise ValueError("both variants and complete matched warm pairs required")
    if any(set(pair) != {"baseline", "candidate"} for pair in cold.values()):
        raise ValueError("cold observations, when present, must also be paired")
    values = {variant: [pair[variant] for pair in warm.values()] for variant in builds}
    medians = {variant: statistics.median(rows) for variant, rows in values.items()}
    warnings = ["Descriptive statistics only; no automatic significance, quality or release qualification.",
                "Identity/cache/quality labels are supplied by the caller; this script does not verify their truth or source hashes."]
    if len(warm) < 4:
        warnings.append("Fewer than four warm pairs: exploratory sample size.")
    if document.get("synthetic") is True:
        warnings.append("Synthetic data: not a hardware benchmark.")
    result = {"scope": document["scope"], "synthetic": document.get("synthetic", False), "identity": identity, "builds": builds,
              "warm_pairs": len(warm), "cold_pairs": len(cold), "recorded_order": "".join(order),
              "warm_median_seconds": medians,
              "warm_p90_nearest_rank_seconds": {key: sorted(rows)[math.ceil(.9 * len(rows)) - 1] for key, rows in values.items()},
              "warm_worst_seconds": {key: max(rows) for key, rows in values.items()},
              "unsuccessful_or_unqualified_samples": failures,
              "qualification_passed": False,
              "warnings": warnings}
    if failures:
        result["warnings"].append("Failure/unknown quality present: speedup deliberately withheld; retain every attempt.")
    else:
        ratio = medians["baseline"] / medians["candidate"]
        paired = [pair["baseline"] / pair["candidate"] for pair in warm.values()]
        reduction = 100 * (1 - medians["candidate"] / medians["baseline"])
        if not all(math.isfinite(x) for x in [ratio, reduction, *paired]):
            raise ValueError("nonfinite derived statistic; timing units/ranges invalid")
        result["pooled_median_speedup"] = ratio
        result["paired_median_speedup"] = statistics.median(paired)
        result["pooled_time_reduction_percent"] = reduction
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    try:
        with args.campaign.open(encoding="utf-8") as stream:
            result = summarize(json.load(stream))
    except (OSError, ValueError, TypeError, AttributeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
