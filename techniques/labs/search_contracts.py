"""Independent cache/acceptance contracts; no ANE, NumPy or upstream imports."""
import hashlib
import json
import math
import random
import statistics

from common import report


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def identities(structure, environment, weights_sha256, input_sha256, limit):
    """Example application keys, NOT a reimplementation of upstream graph hashing.

    The compile key includes frozen weights and implementation/environment;
    a structural search hint cannot grant accuracy for new weights or inputs.
    The caller must bind layout, recipe and boundary in structure/environment.
    """
    for value in (weights_sha256, input_sha256):
        if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError("explicit content SHA256 required")
    if type(limit) not in (int, float) or not math.isfinite(limit) or not 0 < limit < 1:
        raise ValueError("explicit finite numerical budget required")
    if not structure or not environment:
        raise ValueError("explicit structure and environment required")
    asset = {"structure": structure, "environment": environment, "weights_sha256": weights_sha256}
    return {"structural_search_hint": digest(structure), "compile_asset": digest(asset),
            "numerical_qualification": digest({**asset, "input_sha256": input_sha256,
                                                "relative_l2_limit": limit})}


def numeric_pass(metric, limit):
    value = metric.get("relative_l2")
    return (type(limit) in (int, float) and 0 < limit < 1
            and metric.get("finite") is True and type(value) in (int, float)
            and math.isfinite(value) and 0 <= value <= limit)


def paired_result(samples, baseline, candidate, limit=.01):
    """Validate all first/warm/measured rows before returning any paired ratio.

    Timing-only samples never grant numerical qualification. All selected arms
    must have a unique row per phase/trial and share the same actual input hash.
    """
    if baseline == candidate or type(limit) not in (int, float) or not 0 < limit < 1:
        raise ValueError("two distinct arms and a numerical budget required")
    selected = [row for row in samples if row["arm"] in (baseline, candidate)]
    groups = {}
    for row in selected:
        if (row["phase"] not in ("first", "warmup", "measured") or type(row["trial"]) is not int
                or type(row["wall_ns"]) is not int or row["wall_ns"] <= 0
                or row.get("boundary") != "host-fp16-array-to-new-host-fp32-array"
                or row.get("output_dtype") != "float32"
                or not isinstance(row.get("input_sha256"), str) or len(row["input_sha256"]) != 64
                or any(ch not in "0123456789abcdef" for ch in row["input_sha256"])):
            raise ValueError("typed positive time, input identity and complete host boundary required")
        key = (row["phase"], row["trial"])
        group = groups.setdefault(key, {})
        if row["arm"] in group:
            raise ValueError("duplicate arm in a pair")
        group[row["arm"]] = row
    if not groups:
        raise ValueError("no samples")
    for group in groups.values():
        if set(group) != {baseline, candidate} or len({row["input_sha256"] for row in group.values()}) != 1:
            raise ValueError("missing arm or different input in a pair")
    measured = [group for (phase, _), group in groups.items() if phase == "measured"]
    expected = {"first": {0}, "warmup": {0, 1}, "measured": set(range(6))}
    if any({trial for phase, trial in groups if phase == name} != trials for name, trials in expected.items()):
        raise ValueError("complete first, two warmup and six measured pairs required")
    passed = all(numeric_pass(row["metrics"], limit) for row in selected)
    result = {"baseline": baseline, "candidate": candidate, "all_calls_numeric_pass": passed,
              "measured_pairs": len(measured), "median_paired_host_ratio": None,
              "bootstrap_median_ratio_95pct": None, "production_qualified": False}
    if passed:
        ratios = [group[baseline]["wall_ns"] / group[candidate]["wall_ns"] for group in measured]
        rng = random.Random(91)
        boot = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000))
        result.update(paired_ratios=ratios, median_paired_host_ratio=statistics.median(ratios),
                      bootstrap_median_ratio_95pct=[boot[49], boot[1949]])
    return result


def qualification(seed=17):
    """Small CPU identity demo; hashes synthetic metadata, never runs a device."""
    structure = {"graph": "SwiGLU", "shape": [7, 16, 32], "dtype": "fp16",
                 "layout": "row-major", "boundary": "host-output-ready"}
    environment = {"backend": "illustrative-private-runtime", "source": "fixture-v1", "os": "fixture"}
    a = identities(structure, environment, digest([seed, "weights-A"]), digest([seed, "input"]), .01)
    b = identities(structure, environment, digest([seed, "weights-B"]), digest([seed, "input"]), .01)
    assert a["structural_search_hint"] == b["structural_search_hint"]
    assert a["compile_asset"] != b["compile_asset"]
    assert a["numerical_qualification"] != b["numerical_qualification"]
    return report("search-qualification", {"same_structure_different_weights": [a, b],
                  "unmet_error_budget_rejected": not numeric_pass({"finite": True, "relative_l2": .02}, .01),
                  "upstream_algorithm_emulated": False},
                  scope="CPU application identity/acceptance demonstration; no compiler or device evidence")
