#!/usr/bin/env python3
"""Batch-export matching Stage-1/Stage-2 LTX ANE video-FFN artifacts."""

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
    parser.add_argument("--ane-intermediate", type=int, default=6912)
    parser.add_argument(
        "--variant", choices=("fp16", "int8_pc"), default="int8_pc"
    )
    parser.add_argument(
        "--dual", action="store_true",
        help="export one 1001/4004 enumerated-shape model per block",
    )
    parser.add_argument(
        "--reuse-gpu-root", type=Path,
        help="directory containing block-N artifacts whose GPU files can be hardlinked",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    exporter = Path(__file__).with_name("export_ane_mlp_artifact.py")
    for completed, block in enumerate(args.blocks, 1):
        if args.dual:
            target = args.output / f"block-{block}"
            command = [
                sys.executable, str(exporter), str(args.checkpoint),
                str(target), "--block", str(block),
                "--ane-intermediate", str(args.ane_intermediate),
                "--variants", args.variant, "--compile", "--compiled-only",
                "--rows", "dual",
            ]
            if args.reuse_gpu_root:
                command += [
                    "--reuse-gpu-files-from",
                    str(args.reuse_gpu_root / f"block-{block}"),
                ]
            if args.force:
                command.append("--force")
            print(f"[{completed}/{len(args.blocks)}] block {block} dual",
                  flush=True)
            subprocess.run(command, check=True)
            continue
        stage1 = args.output / "stage1" / f"block-{block}"
        stage2 = args.output / "stage2" / f"block-{block}"
        common = [
            sys.executable, str(exporter), str(args.checkpoint),
            "--block", str(block),
            "--ane-intermediate", str(args.ane_intermediate),
            "--variants", args.variant, "--compile", "--compiled-only",
        ]
        if args.force:
            common.append("--force")
        print(f"[{completed}/{len(args.blocks)}] block {block} Stage 1",
              flush=True)
        subprocess.run(
            common[:3] + [str(stage1)] + common[3:] +
            ["--rows", "1001"], check=True
        )
        print(f"[{completed}/{len(args.blocks)}] block {block} Stage 2",
              flush=True)
        subprocess.run(
            common[:3] + [str(stage2)] + common[3:] +
            ["--rows", "4004", "--reuse-gpu-files-from", str(stage1)],
            check=True,
        )


if __name__ == "__main__":
    main()
