#!/usr/bin/env python3
"""Measure same-process ABBA text-K/V caching with exact-output checks."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess

from ltx098_comfy_reference import MODEL_SHA256, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    checkpoint, data, probe = (p.resolve(strict=True) for p in (args.checkpoint, args.input, args.probe))
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory must not already exist")
    if sha256(checkpoint) != MODEL_SHA256:
        raise ValueError("checkpoint SHA-256 differs from the pinned source")
    output.mkdir(parents=True)
    command = [str(probe), str(checkpoint), str(data), str(output / "outputs.safetensors"), str(args.warmups), str(args.repeats)]
    run = subprocess.run(command, capture_output=True, text=True, timeout=1200)
    (output / "stdout.txt").write_text(run.stdout)
    (output / "stderr.txt").write_text(run.stderr)
    report = {"schema": "ltx098-cache-screen-v1", "checkpoint_sha256": MODEL_SHA256,
              "input_sha256": sha256(data), "probe_sha256": sha256(probe), "returncode": run.returncode,
              "scope": "512x512x121 transformer only; identical input, same process ABBA, exact BF16 equality; eight-step estimate includes one cache preparation"}
    if run.returncode:
        report.update(status="failed", error=run.stderr[-4000:])
    else:
        measurement = json.loads(run.stdout)
        if [row["cached"] for row in measurement["runs"]] != [False, True, True, False] or not all(row["exact"] for row in measurement["runs"]):
            raise ValueError("cached/uncached ABBA or exact-output check failed")
        baseline = statistics.median(v for row in measurement["runs"] if not row["cached"] for v in row["seconds"])
        cached = statistics.median(v for row in measurement["runs"] if row["cached"] for v in row["seconds"])
        report.update(status="complete", measurements=measurement, uncached_median_seconds=baseline,
                      cached_median_seconds=cached, transformer_speedup=baseline / cached,
                      estimated_8step_savings_seconds=8 * (baseline - cached) - measurement["cache_prepare_seconds"])
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
