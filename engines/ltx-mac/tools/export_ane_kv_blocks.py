#!/usr/bin/env python3
"""Batch-export LTX-2.5 Video-text K/V Core ML artifacts."""

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
    parser.add_argument(
        "--variant", choices=("fp16", "int8_pc"), default="int8_pc"
    )
    parser.add_argument("--text-rows", type=int, default=1024)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.text_rows <= 0:
        raise SystemExit("--text-rows must be positive")
    exporter = Path(__file__).with_name("export_ane_kv_artifact.py")
    for completed, block in enumerate(args.blocks, 1):
        destination = args.output / f"block-{block}"
        command = [
            sys.executable, str(exporter), str(args.checkpoint),
            str(destination), "--block", str(block),
            "--text-rows", str(args.text_rows),
            "--variants", args.variant, "--compile", "--compiled-only",
        ]
        if args.force:
            command.append("--force")
        print(f"[{completed}/{len(args.blocks)}] block {block}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
