#!/usr/bin/env python3
"""Retain complete synthetic timing arrays with strict, non-model evidence bounds."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def validate_probe(value, script_hash):
    if value.get("status") != "completed" or value.get("model_weights_loaded") is not False:
        raise ValueError("only complete weight-free probes accepted")
    if value.get("script_sha256") != script_hash:
        raise ValueError("probe script changed since run")
    expected = value["repetitions"] * value["rounds"]
    if len(value.get("workloads", [])) != 6:
        raise ValueError("full three-size/two-regime matrix required")
    points = {(w["seq"], w["regime"]) for w in value["workloads"]}
    if points != {(n, r) for n in (512, 2048, 8192) for r in ("independent-random", "position-clustered")}:
        raise ValueError("workload identity mismatch")
    for samples in [value["initial_admission"], *[w["admission"] for w in value["workloads"]]]:
        if len(samples) != 3 or any(type(s["device"]) is not int or not 0 <= s["device"] <= 5 for s in samples):
            raise ValueError("missing/refused GPU admission")
    for workload in value["workloads"]:
        if set(workload["timings"]) != {"dense", "full-local-mask", "sliced-local-sink", "all-exact-chunk"}:
            raise ValueError("missing comparison arm")
        for name, stats in workload["timings"].items():
            samples = stats["samples_ms"]
            if len(samples) != expected or not all(math.isfinite(t) and t > 0 for t in samples):
                raise ValueError("timing samples incomplete/nonfinite")
            if statistics.median(samples) != stats["median_ms"]:
                raise ValueError("median mismatch")
            error = workload["errors"][name]["versus_dense"]
            if error["finite"] is not True or not all(math.isfinite(error[k]) for k in ("relative_l2", "max_abs")):
                raise ValueError("nonfinite quality witness")
        if workload["errors"]["all-exact-chunk"]["versus_dense"]["relative_l2"] > .02:
            raise ValueError("all-exact failed")
        if workload["errors"]["sliced-local-sink"]["versus_local"]["relative_l2"] > .02:
            raise ValueError("same-local-semantics failed")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="append", type=Path, required=True)
    parser.add_argument("--math", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output must be new")
    script_hash = hashlib.sha256((Path(__file__).parent / "mlx_probe.py").read_bytes()).hexdigest()
    probes = [validate_probe(json.loads(p.read_text()), script_hash) for p in args.probe]
    if len(probes) != 2 or len({p["seed"] for p in probes}) != 2:
        parser.error("two independent seeds required")
    for probe in probes:
        # A full ps argv/path is never persisted; even numeric PIDs are unnecessary.
        probe["inference_process_snapshot"] = [{k: v for k, v in row.items() if k != "pid"}
                                                for row in probe["inference_process_snapshot"]]
        probe["gpu_exclusivity_proven"] = False
        probe["media_quality_qualified"] = False
    math_value = json.loads(args.math.read_text())
    if math_value.get("lab_id") != "algorithm-evidence" or not math_value.get("synthetic"):
        raise ValueError("wrong CPU witness")
    value = {"schema": "algorithm-campaign-v1", "date": "2026-10-16", "model_quality_qualified": False,
             "scope": "CPU math witnesses and synthetic MLX host-wall components only; no image/video generation",
             "cpu_math": math_value, "probes": probes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"independent_seeds": [p["seed"] for p in probes], "workloads": sum(len(p["workloads"]) for p in probes),
                      "timed_calls": sum(len(t["samples_ms"]) for p in probes for w in p["workloads"] for t in w["timings"].values())}))


if __name__ == "__main__":
    main()
