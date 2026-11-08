#!/usr/bin/env python3
"""Build and run the small, real-MLX staged Qwen21 attention fixture."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native")).resolve()


def main(build_only: bool = False) -> None:
    mlx_root = Path(os.environ.get(
        "MLX_ROOT", str(Path(sysconfig.get_paths()["purelib"]) / "mlx"),
    ))
    include = mlx_root / "include"
    library = mlx_root / "lib"
    if not (include / "mlx/mlx.h").is_file() or not (
            library / "libmlx.dylib").is_file():
        raise FileNotFoundError(
            "MLX C++ headers/libraries are unavailable; run make setup or "
            "set MLX_ROOT"
        )
    native = NATIVE
    if not (native / "libturbocider.dylib").is_file():
        raise FileNotFoundError("native dylib is unavailable; run make build")
    compiler = subprocess.check_output(
        ["xcrun", "--find", "clang++"], text=True,
    ).strip()
    sdk = subprocess.check_output(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True,
    ).strip()
    load_commands = subprocess.check_output(
        ["otool", "-l", str(library / "libmlx.dylib")], text=True,
    )
    minimum = re.search(
        r"cmd LC_BUILD_VERSION.*?\n(?:.*\n)*?\s+minos\s+(\S+)",
        load_commands,
    )
    deployment = minimum.group(1) if minimum else "15.0"
    with tempfile.TemporaryDirectory(prefix="tc-qwen21-dit-w8a8-") as raw:
        binary = Path(raw) / "qwen21-dit-w8a8-test"
        subprocess.run([
            compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
            "-isysroot", sdk, "-mmacosx-version-min=" + deployment,
            "-I", str(ROOT / "native"),
            "-I", str(ROOT / "native/core"),
            "-isystem", str(include),
            str(ROOT / "tests/native/qwen21_dit_w8a8_test.cpp"),
            "-L", str(native), "-lturbocider",
            "-L", str(library), "-lmlx", "-ljaccl", "-licucore",
            "-framework", "Foundation", "-framework", "Metal",
            "-Wl,-rpath," + str(native),
            "-Wl,-rpath," + str(library),
            "-o", str(binary),
        ], check=True, cwd=ROOT)
        if build_only:
            print("BUILT Qwen21 DiT W8A8 fixture (GPU execution deferred)")
            return
        result = subprocess.run(
            [str(binary)], text=True, capture_output=True, timeout=120, cwd=ROOT,
        )
        combined = result.stdout + result.stderr
        if any(message in combined for message in (
                "No Metal device available", "no Metal device",
                "Failed to get the default Metal device", "Metal is not supported")):
            raise unittest.SkipTest(
                "Qwen21 DiT W8A8 fixture requires Metal access"
            )
        if result.returncode:
            print(result.stdout, end="")
            print(result.stderr, end="", file=sys.stderr)
            raise RuntimeError(
                "Qwen21 DiT W8A8 fixture failed: "
                + str(result.returncode)
            )
        if result.stdout.startswith("SKIP "):
            raise unittest.SkipTest(result.stdout.strip())
        if not result.stdout.startswith("PASS Qwen21 DiT W8A8:"):
            print(result.stdout, end="")
            print(result.stderr, end="", file=sys.stderr)
            raise RuntimeError(
                "Qwen21 DiT W8A8 test exited without its PASS marker"
            )
        print(result.stdout, end="")


@unittest.skipUnless(sys.platform == "darwin", "requires Apple MLX")
class DiTW8A8Tests(unittest.TestCase):
    def test_true_activation_quantization_and_persistent_cache(self):
        main()


if __name__ == "__main__":
    if sys.argv[1:] == ["--build-only"]:
        main(build_only=True)
    else:
        unittest.main()
