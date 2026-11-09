#!/usr/bin/env python3
"""Screen exact GPU RGB conversion on all finite BF16 values and real pixels."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess

from ltx098_comfy_reference import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    probe, source = args.probe.resolve(strict=True), args.input.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or not 2 <= args.repeats <= 20:
        parser.error("new output directory and 2–20 repeats required")
    identities = {"probe_sha256": sha256(probe), "input_sha256": sha256(source)}
    output.mkdir(parents=True)
    report = {"schema": "ltx098-rgb-cast-screen-v1", "status": "running", **identities,
              "scope": "GPU pixel-conversion component only; not full-request timing", "repeats": args.repeats}
    try:
        run = subprocess.run([str(probe), str(source), str(args.repeats)], capture_output=True, text=True, timeout=600)
        (output / "stdout.txt").write_text(run.stdout)
        (output / "stderr.txt").write_text(run.stderr)
        if run.returncode:
            raise RuntimeError(f"probe exited {run.returncode}: {run.stderr[-4000:]}")
        detail = json.loads(run.stdout)
        runs = detail.get("runs", [])
        if (detail.get("finite_bf16_values"), detail.get("exact")) != (65280, True) or len(runs) != 4 or [
                run["fused"] for run in runs] != [False, True, True, False] or any(
                len(run["seconds"]) != args.repeats or any(v <= 0 for v in run["seconds"]) for run in runs):
            raise ValueError("invalid exhaustive parity or ABBA timing")
        if sha256(probe) != identities["probe_sha256"] or sha256(source) != identities["input_sha256"]:
            raise ValueError("probe or input changed during screen")
        baseline = statistics.median(v for row in runs if not row["fused"] for v in row["seconds"])
        fused = statistics.median(v for row in runs if row["fused"] for v in row["seconds"])
        report.update(status="complete", measurement=detail, baseline_median_seconds=baseline,
                      fused_median_seconds=fused, speedup=baseline / fused)
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("baseline_median_seconds", "fused_median_seconds", "speedup")}, indent=2))


if __name__ == "__main__":
    main()
