#!/usr/bin/env python3
"""Educational phase-liveness budget model, NOT a process memory admission tool."""
import argparse
import json
from pathlib import Path


def natural(value, field):
    if type(value) is not int or value < 0:
        raise ValueError(field + " must be a nonnegative integer byte/count value")
    return value


def analyze(document):
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    phases = document.get("phases")
    if not isinstance(phases, list) or not phases or any(not isinstance(x, str) or not x for x in phases) or len(set(phases)) != len(phases):
        raise ValueError("phases must be unique nonempty strings in execution order")
    claims = document.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("claims must be a nonempty list")
    indices = {name: index for index, name in enumerate(phases)}
    names, allocations = set(), {}
    per_phase = {name: {} for name in phases}
    for claim in claims:
        name, allocation = claim["name"], claim["allocation"]
        if not isinstance(name, str) or not name or name in names or not isinstance(allocation, str) or not allocation:
            raise ValueError("claims need unique names and nonempty allocation identities")
        names.add(name)
        size = natural(claim["bytes"], "bytes")
        start, end = indices[claim["first"]], indices[claim["last"]]
        if start > end:
            raise ValueError("claim lifetime is reversed")
        # Same backing can have several live aliases. Full allocation bytes,
        # not logical view bytes, must agree; count the backing once per phase.
        if allocation in allocations and allocations[allocation] != size:
            raise ValueError("aliases disagree about actual backing allocation bytes")
        allocations[allocation] = size
        for phase in phases[start:end + 1]:
            per_phase[phase][allocation] = size
    totals = {phase: sum(rows.values()) for phase, rows in per_phase.items()}
    peak_phase = max(totals, key=totals.get)
    if type(document.get("synthetic", False)) is not bool:
        raise ValueError("synthetic must be a boolean when supplied")
    result = {"scope": "serialized declared phase model; excludes undeclared/driver/system allocations",
              "synthetic": document.get("synthetic", False),
              "phase_bytes": totals, "peak_phase": peak_phase, "declared_peak_bytes": totals[peak_phase],
              "distinct_backing_bytes": sum(allocations.values()),
              "hard_memory_bound": False,
              "warnings": ["Overlapping requests need a different joint timeline; max(serial phases) would undercount.",
                           "Use full allocations for views. Do not sum MLX active and physical footprint."]}
    if "target_bytes" in document:
        target = natural(document["target_bytes"], "target_bytes")
        reserve = natural(document.get("reserve_bytes", 0), "reserve_bytes")
        result.update(target_bytes=target, reserve_bytes=reserve,
                      declared_slack_bytes=target - reserve - totals[peak_phase],
                      declared_fits=totals[peak_phase] + reserve <= target)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    try:
        with args.plan.open(encoding="utf-8") as stream:
            result = analyze(json.load(stream))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
