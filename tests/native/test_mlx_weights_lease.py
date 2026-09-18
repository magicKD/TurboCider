#!/usr/bin/env python3
"""Build and run the fd-backed MLX Weights lease fixture."""

from __future__ import annotations

import os
import re
import subprocess
import sysconfig
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    mlx_root = Path(os.environ.get(
        "MLX_ROOT",
        str(Path(sysconfig.get_paths()["purelib"]) / "mlx"),
    ))
    include = mlx_root / "include"
    library = mlx_root / "lib"
    if not (include / "mlx/mlx.h").is_file() or not (
            library / "libmlx.dylib").is_file():
        raise FileNotFoundError(
            "MLX C++ headers/libraries are unavailable; run make setup or "
            "set MLX_ROOT"
        )
    dylib = ROOT / "build/native/libturbocider.dylib"
    if not dylib.is_file():
        raise FileNotFoundError("native dylib is unavailable; run make build")
    compiler = subprocess.check_output(
        ["xcrun", "--find", "clang++"], text=True
    ).strip()
    sdk = subprocess.check_output(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True
    ).strip()
    load_commands = subprocess.check_output(
        ["otool", "-l", str(library / "libmlx.dylib")], text=True
    )
    minimum = re.search(
        r"cmd LC_BUILD_VERSION.*?\n(?:.*\n)*?\s+minos\s+(\S+)",
        load_commands,
    )
    deployment = minimum.group(1) if minimum else "15.0"
    with tempfile.TemporaryDirectory(prefix="tc-mlx-lease-weights-") as raw:
        temporary = Path(raw)
        binary = temporary / "mlx-lease-weights-test"
        subprocess.run([
            compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
            "-isysroot", sdk, "-mmacosx-version-min=" + deployment,
            "-I", str(ROOT / "native"),
            "-I", str(ROOT / "native/core"),
            "-isystem", str(include),
            str(ROOT / "tests/native/mlx_weights_lease_test.cpp"),
            "-L", str(ROOT / "build/native"), "-lturbocider",
            "-L", str(library), "-lmlx", "-ljaccl", "-licucore",
            "-framework", "Foundation", "-framework", "Metal",
            "-Wl,-rpath," + str(ROOT / "build/native"),
            "-Wl,-rpath," + str(library),
            "-o", str(binary),
        ], check=True)
        result = subprocess.run(
            [str(binary), str(temporary / "fixture")],
            text=True, capture_output=True, timeout=120,
        )
        combined = result.stdout + result.stderr
        unavailable = any(message in combined for message in (
            "No Metal device available",
            "no Metal device",
            "Failed to get the default Metal device",
            "Metal is not supported",
        ))
        if unavailable:
            print("SKIP: MLX lease weights fixture requires Metal access")
            return
        if result.returncode:
            print(result.stdout, end="")
            print(result.stderr, end="", file=os.sys.stderr)
            raise SystemExit(result.returncode)
        if not result.stdout.startswith("PASS MLX lease weights:"):
            print(result.stdout, end="")
            print(result.stderr, end="", file=os.sys.stderr)
            raise RuntimeError(
                "MLX lease weights test exited without its PASS marker"
            )
        print(result.stdout, end="")


if __name__ == "__main__":
    main()
