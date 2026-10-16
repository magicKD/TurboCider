#!/usr/bin/env python3
"""Sanitize/recompute this survey's measured receipts; never executes devices.

Keeps all measured latency/relative-L2 samples, plan device counts and hashes.
Raw source paths, full environment dumps and private tensors are not exported.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


def positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("positive finite timing required")
    return value


def paired(samples):
    if not samples or len(samples) % 4:
        raise ValueError("complete nonempty ABBA samples required")
    ratios = []
    for offset in range(0, len(samples), 4):
        block = samples[offset:offset + 4]
        if [s["arm"] for s in block] != ["baseline", "candidate", "candidate", "baseline"]:
            raise ValueError("invalid ABBA order")
        if len({s["block"] for s in block}) != 1 or block[0]["block"] != offset // 4:
            raise ValueError("invalid ABBA block identity")
        a = statistics.mean(positive(s["ms"]) for s in block if s["arm"] == "baseline")
        b = statistics.mean(positive(s["ms"]) for s in block if s["arm"] == "candidate")
        ratios.append(a / b)
    return {"paired_ratios": ratios, "paired_median_speedup": statistics.median(ratios),
            "baseline_median_ms": statistics.median(s["ms"] for s in samples if s["arm"] == "baseline"),
            "candidate_median_ms": statistics.median(s["ms"] for s in samples if s["arm"] == "candidate")}


def read(path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def admission(root, name):
    data, digest = read(root / name / "admission.json")
    if (data["status"] != "finished" or data["returncode"] != 0
            or len(data["preflight_percent"]) != 5
            or not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v < 5
                       for v in data["preflight_percent"])):
        raise ValueError("completed idle admission required")
    child, child_digest = read(root / name / "child/receipt.json")
    if (child["status"] != "succeeded" or child["returncode"] != 0
            or not child["owned_process_exit_confirmed"]):
        raise ValueError("owned child exit required")
    return {"admission_sha256": digest, "child_receipt_sha256": child_digest,
            "guard_sha256": data["guard_sha256"], "preflight_percent": data["preflight_percent"],
            "during_percent": data["during_percent"], "postflight_percent": data["postflight_percent"],
            "limitation": data["limitation"]}


def curate(root):
    result = {"schema": "gpu-ane-survey-evidence-v1", "date": "2026-10-16",
              "production_qualified": False, "physical_overlap_proven": False,
              "whole_request_tested": False, "integer_mac_observed": False,
              "gpu_runs": [], "compression_runs": []}
    for seed in (53, 61):
        data, digest = read(root / f"engine-seed{seed}.json")
        if data["status"] != "passed" or data["seed"] != seed or len(data["runs"]) != 14:
            raise ValueError("complete GPU protocol required")
        row = {"seed": seed, "raw_report_sha256": digest,
               "source_sha256": data["source_sha256"], "core_binary_sha256": data["core_binary_sha256"],
               "mlx": data["mlx"], "numpy": data["numpy"], "os": data["os"],
               "device": data["device"], "scope": data["scope"],
               "admission": admission(root, f"engine-seed{seed}-admission" + ("-v2" if seed == 53 else "")),
               "runs": []}
        for run in data["runs"]:
            if len(run["samples"]) != 32:
                raise ValueError("32 measured samples required")
            if not all(math.isfinite(s["relative_l2"]) and 0 <= s["relative_l2"] <= run["relative_l2_limit"] for s in run["samples"]):
                raise ValueError("measured numerical gate failed")
            if not all(v["finite"] and math.isfinite(v["relative_l2"]) and v["relative_l2"] <= run["relative_l2_limit"] for v in run["checks"].values()):
                raise ValueError("first-call check failed")
            stats = paired(run["samples"])
            if not math.isclose(stats["paired_median_speedup"], run["paired_median_speedup"], rel_tol=1e-12):
                raise ValueError("upstream ratio inconsistent")
            row["runs"].append({"name": run["name"], "metadata": run["metadata"],
                "first_call_ms_not_cold_process": run["first_call_ms_not_cold_process"],
                "checks": run["checks"], "relative_l2_limit": run["relative_l2_limit"],
                "samples": run["samples"], **stats})
        result["gpu_runs"].append(row)
    for channels, seed in ((256, 59), (512, 67)):
        stem = f"compression-c{channels}-r{channels}"
        data, digest = read(root / stem / "report.json")
        if data["status"] != "finished" or data["seed"] != seed or len(data["samples"]) != 65:
            raise ValueError("complete compression protocol required")
        source_hash = hashlib.sha256((root / stem / "source.py").read_bytes()).hexdigest()
        if source_hash != data["source_sha256"]:
            raise ValueError("compression source snapshot mismatch")
        arms = {}
        for name, arm in data["arms"].items():
            samples = [s for s in data["samples"] if s["arm"] == name]
            measured = [s for s in samples if s["phase"] == "timed"]
            if (arm["status"] != "component-pass" or len(samples) != 13 or len(measured) != 10
                    or sorted(s["trial"] for s in samples) != list(range(-3, 10))):
                raise ValueError("all compression arms/calls required")
            if not all(math.isfinite(s["metrics"]["relative_l2"]) and 0 <= s["metrics"]["relative_l2"] <= arm["gate"] for s in samples):
                raise ValueError("compression numerical gate failed")
            median = statistics.median(positive(s["host_seconds"]) for s in measured)
            if not math.isclose(median, arm["median_host_seconds"], rel_tol=1e-12):
                raise ValueError("compression median inconsistent")
            devices = Counter(op["preferred_device"] for op in arm.get("compute_plan", {}).get("operations", []) if op.get("preferred_device"))
            arms[name] = {"status": arm["status"], "gate": arm["gate"], "median_host_ms": median * 1000,
                "maximum_relative_l2": max(s["metrics"]["relative_l2"] for s in samples),
                "package_bytes": arm.get("package_bytes"),
                "first_prediction_ms": arm["first_prediction_seconds"] * 1000,
                "compile_seconds": arm.get("compile_seconds"), "load_seconds": arm.get("load_seconds"),
                "preferred_device_counts_predicted": dict(devices), "physical_ane_placement": "unverified"}
        result["compression_runs"].append({"seed": seed, "shape": data["shape"], "layers": data["layers"],
            "raw_report_sha256": digest, "source_sha256": source_hash, "environment": data["environment"],
            "scope": data["scope"], "units": data["units"], "weight_sha256": data["weight_sha256"],
            "gates_relative_l2": data["gates_relative_l2"], "arms": arms, "samples": data["samples"],
            "admission": admission(root, stem + "-admission")})
    result["retained_failures"] = [{"stage": "idle-preflight before any device command",
        "status": "admission_or_observation_failed", "cause": "sandbox denied archived IORegistry read",
        "device_command_started": False, "successful_retry": "engine-seed53-admission-v2"}]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output exists; use a fresh path")
    result = curate(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"gpu_runs": len(result["gpu_runs"]), "compression_runs": len(result["compression_runs"]),
                      "production_qualified": False}))


if __name__ == "__main__":
    main()
