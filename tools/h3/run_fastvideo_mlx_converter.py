#!/usr/bin/env python3
"""Run FastVideo's checked-in MLX converter without importing its CUDA stack.

Python normally executes ``fastvideo/__init__.py`` before a submodule import;
that top-level package imports the complete server/training dependency graph.
The H3 MLX converter only needs logger.py and mlx_runtime/*.py.  Install a
minimal namespace package pointing at the checked-in FastVideo tree, then run
the unmodified official converter as ``__main__`` with the caller's argv.
"""

from __future__ import annotations

import importlib.machinery
import runpy
import sys
import types
from pathlib import Path


def install_fastvideo_namespace() -> Path:
    reference = (Path(__file__).resolve().parents[2] / "../references/FastVideo").resolve()
    package_root = reference / "fastvideo"
    if not package_root.is_dir():
        raise FileNotFoundError(f"checked-in FastVideo reference is incomplete: {reference}")

    package = types.ModuleType("fastvideo")
    package.__file__ = str(package_root / "__init__.py")
    package.__package__ = "fastvideo"
    package.__path__ = [str(package_root)]
    package.__spec__ = importlib.machinery.ModuleSpec(
        "fastvideo", loader=None, is_package=True
    )
    package.__spec__.submodule_search_locations = package.__path__
    sys.modules["fastvideo"] = package
    sys.path.insert(0, str(reference))
    return reference


def main() -> None:
    reference = install_fastvideo_namespace()
    converter = reference / "scripts/checkpoint_conversion/convert_minimax_h3_mlx.py"
    if not converter.is_file():
        raise FileNotFoundError(f"FastVideo converter is missing: {converter}")
    runpy.run_path(str(converter), run_name="__main__")


if __name__ == "__main__":
    main()
