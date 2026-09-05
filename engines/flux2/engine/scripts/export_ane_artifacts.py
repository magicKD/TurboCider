#!/usr/bin/env python3
"""Convenience wrapper around the real-weight Core ML exporter in mac_local_ai."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def main() -> None:
    engine_root = Path(__file__).resolve().parents[1]
    project_root = engine_root.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--buckets", type=int, nargs="+", required=True)
    parser.add_argument("--blocks", type=int, nargs="+")
    parser.add_argument("--ane-mlp-width", type=int)
    parser.add_argument("--variant", choices=("int8_pc", "fp16"), default="int8_pc")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--research-root", type=Path, default=project_root / "mac_local_ai")
    args = parser.parse_args()
    exporter = args.research_root / "scripts" / "prepare_flux2_ane_stack.py"
    if not exporter.is_file():
        raise SystemExit(f"ANE exporter was not found: {exporter}")
    command = [
        "python3",
        str(exporter),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--out-dir",
        str(args.out_dir.resolve()),
        "--buckets",
        *(str(bucket) for bucket in args.buckets),
        "--variants",
        args.variant,
    ]
    if args.blocks is not None:
        command.extend(["--blocks", *(str(block) for block in args.blocks)])
    if args.ane_mlp_width is not None:
        command.extend(["--ane-mlp-width", str(args.ane_mlp_width)])
    if args.overwrite:
        command.append("--overwrite")
    environment = os.environ.copy()
    coreml_path = str((args.research_root / ".deps" / "coreml").resolve())
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (coreml_path, environment.get("PYTHONPATH", "")) if part
    )
    subprocess.run(command, cwd=args.research_root, env=environment, check=True)


if __name__ == "__main__":
    main()
