#!/usr/bin/env python3
"""Repeat biased IOSurface row-split screens across actual LTX blocks/inputs.

The order 0,14,27,27,14,0 provides a simple drift check. CPU_AND_NE remains a
scheduling policy with unknown physical placement; this is not full inference.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    checkpoint, manifest, activations, probe = (path.resolve(strict=True) for path in
        (args.checkpoint, args.manifest, args.activations, args.probe))
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory must not already exist")
    if not 1 <= args.chunks <= 15 or not 2 <= args.repeats <= 20:
        parser.error("invalid chunks or repeats")
    for block in (0, 14, 27):
        if not (activations / f"ffn_{block}_input.safetensors").is_file():
            raise ValueError(f"captured block {block} activation missing")
    output.mkdir(parents=True)
    report = {"schema": "ltx098-ane-bias-blocks-screen-v1", "status": "running",
              "scope": "Real first-step inputs and both biases for blocks 0/14/27; native IOSurface CPU_AND_NE plus GPU split; unknown physical placement; component-only",
              "chunks": args.chunks, "runs": []}
    try:
        for index, block in enumerate((0, 14, 27, 27, 14, 0)):
            destination = output / f"run-{index}-block-{block}"
            command = [sys.executable, str(Path(__file__).with_name("ltx098_ane_bias_split_screen.py")),
                       "--checkpoint", str(checkpoint), "--manifest", str(manifest),
                       "--activation", str(activations / f"ffn_{block}_input.safetensors"),
                       "--probe", str(probe), "--output", str(destination), "--chunks", str(args.chunks),
                       "--repeats", str(args.repeats), "--compute", "ne", "--block", str(block)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=1200)
            (output / f"run-{index}.stdout.txt").write_text(result.stdout)
            (output / f"run-{index}.stderr.txt").write_text(result.stderr)
            if result.returncode:
                raise RuntimeError(f"block {block} exited {result.returncode}: {result.stderr[-4000:]}")
            detail = json.loads((destination / "report.json").read_text())
            if detail.get("status") != "complete" or detail["measurements"]["block"] != block:
                raise ValueError("incomplete or mismatched block result")
            row = {"index": index, "block": block, "gpu_seconds": detail["gpu_median_seconds"],
                   "split_with_stage_seconds": detail["split_with_stage_median_seconds"],
                   "speedup_with_stage": detail["speedup_with_stage"],
                   "relative_l2_all": detail["error_all"]["relative_l2"],
                   "relative_l2_coreml_tail": detail["error_coreml_tail"]["relative_l2"],
                   "gpu_head_exact": detail["error_gpu_head"]["exact"],
                   "physical_ane_placement": detail["measurements"]["physical_ane_placement"]}
            report["runs"].append(row)
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
        report.update(status="complete", median_speedup_with_stage=statistics.median(row["speedup_with_stage"] for row in report["runs"]),
                      minimum_speedup_with_stage=min(row["speedup_with_stage"] for row in report["runs"]),
                      maximum_relative_l2_all=max(row["relative_l2_all"] for row in report["runs"]),
                      maximum_relative_l2_coreml_tail=max(row["relative_l2_coreml_tail"] for row in report["runs"]))
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
