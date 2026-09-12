#!/usr/bin/env python3
"""Run FastVideo's checked-in FastH3 MLX entrypoint with local assets only.

The normal :mod:`fastvideo` package initializer imports its CUDA and serving
stack.  TurboCider's parity and benchmark work only needs the checked-in MLX
runtime, so this wrapper installs the same minimal namespace used by the
checkpoint converter and then executes the upstream entrypoint unchanged.

H3 model access is deliberately offline here.  ``--model-root`` and
``--mlx-checkpoint`` must point at assets already downloaded from the approved
ModelScope repository.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/h3"))

from run_fastvideo_mlx_converter import install_fastvideo_namespace  # noqa: E402


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    reference = install_fastvideo_namespace()
    entrypoint = reference / "examples/inference/basic/mlx_fasth3.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"FastVideo FastH3 MLX entrypoint is missing: {entrypoint}")
    sys.argv[0] = str(entrypoint)
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
