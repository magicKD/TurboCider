#!/usr/bin/env python3
"""Export a resumable fixed-shape LTX Gemma ANE MLP procedure bank.

The child exporter is intentionally invoked one layer at a time.  This keeps
peak host memory bounded, makes an interrupted export resumable, and preserves
disk headroom instead of constructing all 48 Core ML packages concurrently.
The resulting ``block-XX/manifest.json`` layout is consumed directly by the
native Gemma encoder.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


SCHEMA = "ltx-gemma-ane-encoder-bank-v1"
LAYER_SCHEMA = "ltx-gemma-ane-mlp-v1"
LAYERS = 48
DEFAULT_ESTIMATED_BLOCK_BYTES = 256 << 20


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--first-block", type=int, default=0)
    parser.add_argument("--block-count", type=int, default=LAYERS)
    parser.add_argument("--rows", default="64,128,256,512,1024")
    parser.add_argument("--ane-intermediate", type=int, required=True)
    parser.add_argument(
        "--min-profitable-rows",
        help="comma-separated BUCKET:MIN_ACTUAL_ROWS crossovers",
    )
    parser.add_argument("--min-free-gib", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.first_block < 0 or args.first_block >= LAYERS:
        parser.error("--first-block must be in 0...47")
    if (args.block_count < 1 or
            args.first_block + args.block_count > LAYERS):
        parser.error("requested block range must stay within 0...47")
    if args.min_free_gib < 0:
        parser.error("--min-free-gib must be nonnegative")
    if args.resume and args.force:
        parser.error("--resume and --force are mutually exclusive")
    return args


def block_directory(root: Path, block: int) -> Path:
    return root / f"block-{block:02d}"


def profitability_policy(value: str | None, rows: list[int]) -> dict[str, int]:
    result = {}
    if not value:
        return result
    for entry in value.split(","):
        try:
            bucket_text, minimum_text = entry.split(":", 1)
            bucket, minimum = int(bucket_text), int(minimum_text)
        except ValueError as error:
            raise ValueError(
                "--min-profitable-rows must contain BUCKET:MIN pairs"
            ) from error
        if (bucket not in rows or str(bucket) in result or
                minimum < 1 or minimum > bucket):
            raise ValueError(
                "each minimum-profitable-row policy must name one selected "
                "bucket and stay within 1...BUCKET"
            )
        result[str(bucket)] = minimum
    return result


def complete_block(path: Path, block: int, rows: list[int],
                   ane_intermediate: int,
                   minimum_profitable_rows: dict[str, int]) -> bool:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        artifact_name = manifest["compiled_artifacts"]["fp16"]
        artifact = path / artifact_name
        return (
            manifest.get("schema") == LAYER_SCHEMA and
            manifest.get("block_index") == block and
            manifest.get("shape", {}).get("supported_rows") == rows and
            manifest.get("shape", {}).get(
                "minimum_profitable_rows", {}) == minimum_profitable_rows and
            manifest.get("partition", {}).get("ane", {}).get("width") ==
            ane_intermediate and
            artifact.suffix == ".mlmodelc" and artifact.is_dir()
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def write_index(root: Path, args: argparse.Namespace, rows: list[int],
                completed: list[int],
                minimum_profitable_rows: dict[str, int]) -> None:
    stat = args.checkpoint.stat()
    payload = {
        "schema": SCHEMA,
        "source": {
            "path": str(args.checkpoint.resolve()),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "first_block": args.first_block,
        "block_count": args.block_count,
        "completed_blocks": completed,
        "supported_rows": rows,
        "hidden": 3840,
        "intermediate": 15360,
        "ane_intermediate": args.ane_intermediate,
        **({"minimum_profitable_rows": minimum_profitable_rows}
           if minimum_profitable_rows else {}),
        "gpu_intermediate": 15360 - args.ane_intermediate,
        "layout": "block-%02u/manifest.json",
        "failure_policy": "per_layer_exact_gpu_fallback",
        "residency": "streamed_one_layer",
    }
    temporary = root / "bank_manifest.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(root / "bank_manifest.json")


def main() -> int:
    args = arguments()
    args.checkpoint = args.checkpoint.resolve()
    args.out_dir = args.out_dir.resolve()
    if not args.checkpoint.is_file() or args.checkpoint.is_symlink():
        raise ValueError("checkpoint must be a regular non-symlink file")
    rows = [int(value) for value in args.rows.split(",")]
    minimum_profitable_rows = profitability_policy(
        args.min_profitable_rows, rows)
    blocks = list(range(args.first_block,
                        args.first_block + args.block_count))
    plan = {
        "schema": SCHEMA,
        "checkpoint": str(args.checkpoint),
        "output": str(args.out_dir),
        "blocks": blocks,
        "rows": rows,
        "ane_intermediate": args.ane_intermediate,
        "minimum_profitable_rows": minimum_profitable_rows,
        "min_free_bytes": int(args.min_free_gib * (1 << 30)),
        "resume": args.resume,
        "force": args.force,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.out_dir.exists() and not (args.resume or args.force):
        raise ValueError("out_dir exists; use --resume or --force")
    if args.force and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    exporter = Path(__file__).with_name("export_ltx_gemma_mlp.py")
    completed = [block for block in blocks if complete_block(
        block_directory(args.out_dir, block), block, rows,
        args.ane_intermediate, minimum_profitable_rows)]
    write_index(args.out_dir, args, rows, completed,
                minimum_profitable_rows)
    observed_sizes = [tree_bytes(block_directory(args.out_dir, block))
                      for block in completed]
    estimated = (sum(observed_sizes) // len(observed_sizes)
                 if observed_sizes else DEFAULT_ESTIMATED_BLOCK_BYTES)
    minimum_free = int(args.min_free_gib * (1 << 30))

    for block in blocks:
        destination = block_directory(args.out_dir, block)
        if block in completed:
            print(json.dumps({"block": block, "status": "reused"}), flush=True)
            continue
        free = shutil.disk_usage(args.out_dir).free
        if free - estimated < minimum_free:
            raise RuntimeError(
                f"refusing block {block}: {free} free bytes would leave less "
                f"than {minimum_free} bytes of configured headroom"
            )
        command = [
            sys.executable, str(exporter), str(args.checkpoint),
            str(destination), "--block", str(block), "--rows", args.rows,
            "--ane-intermediate", str(args.ane_intermediate),
            "--compile", "--compiled-only",
        ]
        if args.min_profitable_rows:
            command.extend(["--min-profitable-rows", args.min_profitable_rows])
        if destination.exists():
            command.append("--force")
        subprocess.run(command, check=True)
        if not complete_block(destination, block, rows,
                              args.ane_intermediate,
                              minimum_profitable_rows):
            raise RuntimeError(f"block {block} export is incomplete")
        completed.append(block)
        completed.sort()
        size = tree_bytes(destination)
        observed_sizes.append(size)
        estimated = sum(observed_sizes) // len(observed_sizes)
        write_index(args.out_dir, args, rows, completed,
                    minimum_profitable_rows)
        print(json.dumps({
            "block": block, "status": "exported", "bytes": size,
            "free_bytes": shutil.disk_usage(args.out_dir).free,
        }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
