#!/usr/bin/env python3
"""Benchmark exact decoder padding candidates on representative GPU shapes.

This synthetic-input component screen is not an end-to-end acceleration claim.
The probe checks byte equality before measuring in reference/candidate/
candidate/reference order. No production model code is changed.
"""

import argparse
import json
from pathlib import Path
import statistics
import subprocess

from ltx098_comfy_reference import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    probe = args.probe.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or not 2 <= args.repeats <= 20:
        parser.error("new output directory and 2–20 repeats required")
    cases = ((1, 2, 2, 128, "contiguous"), (2, 3, 5, 128, "bits"),
             (2, 3, 5, 128, "transpose"), (2, 3, 5, 128, "slice"),
             (2, 3, 5, 128, "flip"), (2, 3, 5, 7, "contiguous"),
             (16, 16, 16, 1024, "contiguous"), (31, 32, 32, 512, "contiguous"),
             (61, 64, 64, 256, "contiguous"), (121, 128, 128, 128, "contiguous"))
    output.mkdir(parents=True)
    report = {"schema": "ltx098-vae-pad-screen-v1", "status": "running", "probe_sha256": sha256(probe),
              "scope": "Synthetic BF16 GPU padding only; byte equality, not VAE or full-video quality/performance",
              "repeats_per_group": args.repeats, "cases": []}
    try:
        for f, h, w, c, layout in cases:
            command = [str(probe), str(f), str(h), str(w), str(c), str(args.repeats), layout]
            result = subprocess.run(command, text=True, capture_output=True, timeout=600)
            if result.returncode:
                raise RuntimeError(f"case {f}x{h}x{w}x{c} {layout} exited {result.returncode}: {result.stderr[-4000:]}")
            row = json.loads(result.stdout)
            if (row.get("frames"), row.get("height"), row.get("width"), row.get("channels"), row.get("layout"), row.get("exact")) != (
                    f, h, w, c, layout, True):
                raise ValueError("probe returned an unexpected or inexact case")
            runs = row["runs"]
            if len(runs) != 4 or [run["candidate"] for run in runs] != [False, True, True, False] or any(
                    len(run["seconds"]) != args.repeats or any(value <= 0 for value in run["seconds"]) for run in runs):
                raise ValueError("probe returned an incomplete ABBA measurement")
            baseline = [value for run in runs if not run["candidate"] for value in run["seconds"]]
            candidate = [value for run in runs if run["candidate"] for value in run["seconds"]]
            row.update(reference_median_seconds=statistics.median(baseline), candidate_median_seconds=statistics.median(candidate),
                       median_speedup=statistics.median(baseline) / statistics.median(candidate))
            report["cases"].append(row)
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({key: row[key] for key in ("frames", "height", "width", "channels", "layout", "exact", "fused",
                                                       "reference_median_seconds", "candidate_median_seconds", "median_speedup")}), flush=True)
        if sha256(probe) != report["probe_sha256"]:
            raise ValueError("probe identity changed during screen")
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
