"""Standard-library-only audit/summary for the ANE survey's synthetic receipts.

Checks evidence consistency, not hardware attestation or independent rerunning
of the FP64 oracle. Explicitly keeps plans and inferred ratios non-physical.
"""
from collections import Counter
import math
import random
import statistics


ARMS = ("fp16", "w8", "lut4", "w8a8", "mlx-fp16")
GATES = {"fp16": .015, "w8": .06, "lut4": .35, "w8a8": .08, "mlx-fp16": .015}


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def paired_ratio(rows, arm, rounds):
    pairs = []
    for trial in range(rounds):
        selected = {r["arm"]: r for r in rows if r["trial"] == trial}
        if len(selected) != len(ARMS) or len({r["input_sha256"] for r in selected.values()}) != 1:
            raise ValueError("unpaired/missing input generation")
        pairs.append(selected["fp16"]["host_seconds"] / selected[arm]["host_seconds"])
    rng = random.Random(1729)
    boot = sorted(statistics.median(rng.choices(pairs, k=len(pairs))) for _ in range(2000))
    return {"source_kind": "derived", "paired_median_fp16_over_arm": statistics.median(pairs),
            "exploratory_bootstrap_95pct": [boot[49], boot[1949]],
            "pairs": len(pairs), "caveat": "ordered same-process samples; not independent long-run qualification"}


def compression(report):
    if report.get("schema") != "ane-compression-survey-v1" or report.get("status") != "finished":
        raise ValueError("not a successful complete compression protocol")
    if set(report.get("arms", {})) != set(ARMS) or report.get("gates_relative_l2") != GATES:
        raise ValueError("arms/gates changed")
    if report.get("physical_ane_placement") != "unverified" or report.get("integer_mac_observed") is not False:
        raise ValueError("physical attribution promoted")
    rounds, layers = report["rounds"], report["layers"]
    if rounds not in (10, 20) or not digest(report["source_sha256"]) or not digest(report["weight_sha256"]):
        raise ValueError("bad protocol/source identity")
    seen = set()
    for row in report["samples"]:
        trial, arm = row["trial"], row["arm"]
        if (type(trial) is not int or not -3 <= trial < rounds or arm not in ARMS
                or (trial, arm) in seen or set(row["order"]) != set(ARMS)
                or len(row["order"]) != len(ARMS)):
            raise ValueError("invalid/duplicate call or order")
        seen.add((trial, arm))
        if (not positive(row["host_seconds"]) or row["phase"] != ("warmup" if trial < 0 else "timed")
                or row["input_sha256"] != report["input_sha256"][trial + 3]
                or not digest(row["output_sha256"])):
            raise ValueError("invalid time/phase/generation")
        metric = row["metrics"]
        if any(type(metric[k]) not in (int, float) or not math.isfinite(metric[k]) or metric[k] < 0
               for k in ("relative_l2", "max_abs")) or metric["relative_l2"] > GATES[arm]:
            raise ValueError("numerical gate failed")
    if len(seen) != (rounds + 3) * len(ARMS):
        raise ValueError("missing first/warmup/timed outputs")
    summary = {}
    for arm in ARMS:
        info = report["arms"][arm]
        rows = [r for r in report["samples"] if r["arm"] == arm]
        timed = [r for r in rows if r["trial"] >= 0]
        if (info["status"] != "component-pass" or info["timed_calls"] != rounds
                or info["gate"] != GATES[arm]
                or info["median_host_seconds"] != statistics.median(r["host_seconds"] for r in timed)
                or info["first_prediction_seconds"] != rows[0]["host_seconds"]
                or info["maximum_relative_l2"] != max(r["metrics"]["relative_l2"] for r in rows)):
            raise ValueError("arm summary differs from complete samples")
        summary[arm] = {k: v for k, v in info.items() if k != "compute_plan"}
        summary[arm]["pair_comparison"] = paired_ratio(report["samples"], arm, rounds)
        if arm != "mlx-fp16":
            ops = info["mil_operator_counts"]
            required = {"conv": layers, "relu": layers}
            if arm == "w8" or arm == "w8a8":
                required["constexpr_blockwise_shift_scale"] = layers
            if arm == "lut4":
                required["constexpr_lut_to_dense"] = layers
            if arm == "w8a8":
                required.update(quantize=layers + 1, dequantize=layers + 1)
            if any(ops.get(k) != v for k, v in required.items()):
                raise ValueError("compression graph identity differs")
            plan = info["compute_plan"]
            summary[arm]["plan_preferred_counts"] = dict(Counter(
                o["preferred_device"] or "unknown" for o in plan["operations"]))
            summary[arm]["plan_kind"] = "predicted-not-physical"
    return summary


def training(report):
    if (report.get("schema") != "ane-training-probe-v1" or report.get("status") != "component-pass"
            or report.get("relative_l2_gate") != .015 or len(report.get("calls", [])) != 37
            or len(report.get("steps", [])) != 12 or report.get("optimizer") != "host-fp32-SGD"):
        raise ValueError("incomplete training protocol")
    if report["compile_count"] != (1 if report["mode"] == "public-coreml" else 0):
        raise ValueError("wrong compilation boundary")
    if report["unique_master_generations"] != 12 or len({s["master_sha256"] for s in report["steps"]}) != 12:
        raise ValueError("missing parameter generation")
    if not all(positive(c["host_seconds"]) and 0 <= c["relative_l2"] <= .015
               and digest(c["input_sha256"]) and digest(c["output_sha256"]) for c in report["calls"]):
        raise ValueError("training call failed")
    if not positive(report["final_loss"]) or report["final_loss"] >= report["initial_loss"] * .8:
        raise ValueError("training loss gate rejected")
    if [c["kind"] for c in report["calls"]] != ["forward", "dX", "dW"] * 12 + ["final-forward"]:
        raise ValueError("incorrect backprop sequence")
    summary = {k: v for k, v in report.items() if k not in ("calls", "steps", "compute_plan")}
    summary["maximum_call_relative_l2"] = max(c["relative_l2"] for c in report["calls"])
    summary["plan_preferred_counts"] = dict(Counter(o["preferred_device"] or "unknown"
        for o in report.get("compute_plan", {}).get("operations", [])))
    return summary
