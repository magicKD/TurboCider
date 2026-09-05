#!/usr/bin/env python3
"""Batch-export Stage-1/Stage-2 LTX Video self-QKV Core ML artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def parse_blocks(value: str) -> list[int]:
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        block = int(item)
        if block < 0 or block >= 48:
            raise argparse.ArgumentTypeError("blocks must be in [0, 47]")
        if block not in result:
            result.append(block)
    if not result:
        raise argparse.ArgumentTypeError("at least one block is required")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--blocks", type=parse_blocks, required=True)
    parser.add_argument("--stage1-ane-rows", type=int, default=384)
    parser.add_argument("--stage2-ane-rows", type=int, default=1575)
    parser.add_argument(
        "--variant", choices=("fp16", "int8_pc"), default="int8_pc"
    )
    parser.add_argument(
        "--stage", choices=("both", "stage1", "stage2"), default="both"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.stage1_ane_rows <= 0 or args.stage1_ane_rows >= 1001:
        raise SystemExit("Stage-1 ANE rows must be in [1, 1000]")
    if args.stage2_ane_rows <= 0 or args.stage2_ane_rows >= 4004:
        raise SystemExit("Stage-2 ANE rows must be in [1, 4003]")

    exporter = Path(__file__).with_name("probe_ane_qkv.py")
    stages: list[tuple[str, int]] = []
    if args.stage in ("both", "stage1"):
        stages.append(("stage1", args.stage1_ane_rows))
    if args.stage in ("both", "stage2"):
        stages.append(("stage2", args.stage2_ane_rows))

    total = len(args.blocks) * len(stages)
    completed = 0
    for block in args.blocks:
        for stage, rows in stages:
            completed += 1
            destination = args.output / stage / f"block-{block}"
            command = [
                sys.executable,
                str(exporter),
                str(args.checkpoint),
                "--block", str(block),
                "--rows", str(rows),
                "--variant", args.variant,
                "--warmup", "0",
                "--iterations", "1",
                "--export-dir", str(destination),
                "--compile",
                "--compiled-only",
            ]
            if args.force:
                command.append("--force")
            print(
                f"[{completed}/{total}] block {block} {stage} "
                f"ANE rows={rows}",
                flush=True,
            )
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
