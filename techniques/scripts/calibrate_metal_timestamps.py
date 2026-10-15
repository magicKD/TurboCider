#!/usr/bin/env python3
"""Calibrate explicit owned Metal reports with measured reference brackets.

Only same-process CLOCK_UPTIME_RAW/Mach brackets qualify a CPU unit hypothesis.
GPU ticks are mapped by the observed two-point slope, then cross-checked against
completed command windows. Converted values remain derived diagnostics, not
unprofiled performance scores or a device-independent clock qualification.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

MAX_BRACKET_NS = 1000000
ABS_WINDOW_TOLERANCE_NS = 2000
REL_DURATION_TOLERANCE = .05


def uint(value):
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError("raw clock values must be uint64, not boolean/float")
    return value


def positive_float(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("completed command timestamps must be finite positive seconds")
    return value


def reference_hypotheses(reference):
    if (reference.get("source_kind") != "measured" or reference.get("reference_clock") !=
            "mach_absolute_time and CLOCK_UPTIME_RAW; same process"):
        raise ValueError("measured same-process reference-clock provenance required")
    fields = ("cpu_raw", "gpu_raw", "mach_ticks_before", "mach_ticks_after", "uptime_ns_before",
              "uptime_ns_after", "mach_timebase_numer", "mach_timebase_denom")
    values = {key: uint(reference[key]) for key in fields}
    if not values["mach_timebase_numer"] or not values["mach_timebase_denom"]:
        raise ValueError("positive Mach timebase required")
    factor = Fraction(values["mach_timebase_numer"], values["mach_timebase_denom"])
    mach_low, mach_high = values["mach_ticks_before"], values["mach_ticks_after"]
    ns_low, ns_high = values["uptime_ns_before"], values["uptime_ns_after"]
    if (mach_high < mach_low or ns_high < ns_low
            or max((mach_high - mach_low) * factor, ns_high - ns_low) > MAX_BRACKET_NS):
        return set(), factor
    candidates = set()
    if ns_low <= values["cpu_raw"] <= ns_high:
        candidates.add("nanoseconds")
    if mach_low <= values["cpu_raw"] <= mach_high:
        candidates.add("mach-ticks")
    return candidates, factor


def calibrate_sample(sample):
    counter = sample.get("counter_diagnostic")
    result = {"trial": sample["trial"], "warmup": sample["warmup"], "variant": sample["variant"],
              "status": "not-calibrated", "cpu_unit": None, "gpu_ns_per_tick": None,
              "issues": [], "events": [], "formal_performance_score": False}
    if not counter or not counter.get("reference_before") or not counter.get("reference_after"):
        result["issues"].append("missing-reference-brackets; historical raw ticks not retrofitted")
        return result
    if counter.get("source_kind") != "measured" or counter.get("sampling_point") != "stage-boundary":
        result["issues"].append("counter provenance or sampling point not qualified")
        return result
    before, after = counter["reference_before"], counter["reference_after"]
    hypotheses_before, factor_before = reference_hypotheses(before)
    hypotheses_after, factor_after = reference_hypotheses(after)
    common = hypotheses_before & hypotheses_after
    if factor_before != factor_after:
        result["issues"].append("Mach timebase changed")
        return result
    if common == {"nanoseconds", "mach-ticks"} and factor_before == 1:
        unit, cpu_factor = "ns-or-Mach-equivalent-at-timebase-one", Fraction(1)
    elif len(common) == 1:
        unit = next(iter(common))
        cpu_factor = Fraction(1) if unit == "nanoseconds" else factor_before
    else:
        result["issues"].append("CPU unit not uniquely bracketed")
        return result
    cpu_begin, cpu_end = uint(before["cpu_raw"]), uint(after["cpu_raw"])
    gpu_begin, gpu_end = uint(before["gpu_raw"]), uint(after["gpu_raw"])
    if (cpu_end <= cpu_begin or gpu_end <= gpu_begin
            or counter["cpu_pair_before_raw"] != cpu_begin or counter["cpu_pair_after_raw"] != cpu_end
            or counter["gpu_pair_before_raw"] != gpu_begin or counter["gpu_pair_after_raw"] != gpu_end):
        result["issues"].append("nonmonotonic or inconsistent calibration pairs")
        return result
    scale = Fraction(cpu_end - cpu_begin) * cpu_factor / (gpu_end - gpu_begin)
    result.update(cpu_unit=unit, cpu_ns_per_cpu_unit=float(cpu_factor), gpu_ns_per_tick=float(scale),
                  scale_fraction={"numerator": scale.numerator, "denominator": scale.denominator},
                  reference_bracket_width_ns=[int(before["uptime_ns_after"] - before["uptime_ns_before"]),
                                              int(after["uptime_ns_after"] - after["uptime_ns_before"])])
    rows, commands = counter["samples"], counter.get("completed_command_windows", [])
    if not rows or not commands or len(commands) not in (1, len(rows)):
        result["issues"].append("missing or unsupported stage-to-command association")
        return result
    for index, row in enumerate(rows):
        start, end = uint(row["start_raw"]), uint(row["end_raw"])
        event = {"index": index, "label": row["label"], "raw_start": start, "raw_end": end,
                 "derived_duration_seconds": None, "qualified_duration_seconds": None,
                 "source_kind": "derived", "window_match": False, "issues": []}
        if (row.get("valid") is not True or not 0 < start < end < 2**64 - 1
                or row.get("delta_raw") != end - start):
            event["issues"].append("invalid raw counter or sentinel")
            result["events"].append(event)
            continue
        command = commands[index if len(commands) == len(rows) else 0]
        if command.get("source_kind") != "measured":
            raise ValueError("measured completed-command provenance required")
        cb_start = Fraction(str(positive_float(command["start_host_seconds"]))) * 10**9
        cb_end = Fraction(str(positive_float(command["end_host_seconds"]))) * 10**9
        if cb_end <= cb_start:
            raise ValueError("nonpositive completed command window")
        mapped_start = cpu_begin * cpu_factor + (start - gpu_begin) * scale
        mapped_end = cpu_begin * cpu_factor + (end - gpu_begin) * scale
        duration_ns = (end - start) * scale
        event.update(derived_duration_seconds=float(duration_ns / 10**9),
                     start_minus_command_start_ns=float(mapped_start - cb_start),
                     end_minus_command_end_ns=float(mapped_end - cb_end),
                     command_duration_seconds=float((cb_end - cb_start) / 10**9))
        if start < gpu_begin or end > gpu_end:
            event["issues"].append("counter outside sampled calibration interval")
        if mapped_start < cb_start - ABS_WINDOW_TOLERANCE_NS or mapped_end > cb_end + ABS_WINDOW_TOLERANCE_NS:
            event["issues"].append("mapped stage not contained in associated command window")
        if len(commands) == len(rows):
            residual = duration_ns - (cb_end - cb_start)
            event["duration_residual_ns"] = float(residual)
            tolerance = max(ABS_WINDOW_TOLERANCE_NS, float(cb_end - cb_start) * REL_DURATION_TOLERANCE)
            if abs(residual) > tolerance:
                event["issues"].append("single-stage duration disagrees with completed command")
        event["window_match"] = not event["issues"]
        if event["window_match"]:
            event["qualified_duration_seconds"] = event["derived_duration_seconds"]
        result["events"].append(event)
    result["status"] = "calibration-and-window-pass" if all(row["window_match"] for row in result["events"]) else "window-validation-failed"
    return result


def normalize(document):
    if (document.get("schema") != "turbocider-metal-teaching-lab-v1"
            or document.get("diagnostic_mode") != "counter-stage"
            or any(document.get(key) is not True for key in
                   ("actual_metal_dispatch", "all_outputs_finite", "tail_guards_intact", "source_inputs_unchanged"))):
        raise ValueError("validated source-stable native stage-counter report required")
    samples = document["warmup_samples"] + document["samples"]
    repeats = document["arguments"]["repeats"]
    if (type(repeats) is not int or not 4 <= repeats <= 64 or repeats % 4
            or [row["trial"] for row in samples] != [-4,-3,-2,-1,*range(repeats)]
            or any(row["warmup"] is not (row["trial"] < 0) for row in samples)):
        raise ValueError("complete ordered warmup and measured native trials required")
    converted = [calibrate_sample(row) for row in samples]
    return {"schema": "tc-metal-clock-calibration-v1", "status": "pass" if all(row["status"] == "calibration-and-window-pass" for row in converted) else "incomplete-or-failed",
            "case": document["case"], "device": document["device"], "os": document.get("os"), "arguments": document["arguments"],
            "collector_sources_sha256": document["source_sha256"], "binary_sha256": document["binary_sha256"],
            "numerical_max_abs_error": document["maximum_abs_error"], "samples": converted,
            "gates": {"reference_bracket_max_ns": MAX_BRACKET_NS, "command_absolute_tolerance_ns": ABS_WINDOW_TOLERANCE_NS,
                      "single_stage_command_relative_tolerance": REL_DURATION_TOLERANCE},
            "production_qualified": False, "formal_performance_score": False,
            "scope": "Derived same-run affine clock mapping and completed-command cross-check; not independent physical sensor, unprofiled speed or cross-device/OS qualification."}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="fresh calibration campaign JSON")
    args = parser.parse_args()
    values = []
    try:
        if len({path.resolve() for path in args.reports}) != len(args.reports):
            raise ValueError("duplicate selected report")
        for path in args.reports:
            if path.stat().st_size > 32 * 1024 * 1024:
                raise ValueError("raw JSON exceeds 32 MiB cap")
            document = json.loads(path.read_text(encoding="utf-8"))
            for snapshot in document.get("source_snapshots", []):
                relative = Path(snapshot["path"])
                source = (path.parent / relative).resolve()
                if (relative.is_absolute() or ".." in relative.parts or not source.is_relative_to(path.parent.resolve())
                        or digest(source) != snapshot["sha256"]):
                    raise ValueError("missing/escaping/changed collector snapshot")
            value = normalize(document)
            value.update(id=path.parent.name, raw_report_sha256=digest(path), source_snapshots=document.get("source_snapshots", []))
            values.append(value)
        result = {"schema": "tc-metal-clock-campaign-v1", "normalizer_sha256": digest(Path(__file__)),
                  "runs": values, "production_qualified": False, "formal_performance_score": False}
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"runs": len(values), "pass": sum(value["status"] == "pass" for value in values)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
