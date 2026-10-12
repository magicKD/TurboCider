#!/usr/bin/env python3
"""Build and run the fd-backed Qwen21 text layer streaming fixture."""

from __future__ import annotations

import os
import re
import subprocess
import sysconfig
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native")).resolve()


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
    dylib = NATIVE / "libturbocider.dylib"
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
    with tempfile.TemporaryDirectory(prefix="tc-qwen21-text-streaming-") as raw:
        temporary = Path(raw)
        binary = temporary / "qwen21-text-streaming-test"
        subprocess.run([
            compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
            "-isysroot", sdk, "-mmacosx-version-min=" + deployment,
            "-I", str(ROOT / "native"),
            "-I", str(ROOT / "native/core"),
            "-isystem", str(include),
            str(ROOT / "tests/native/qwen21_text_streaming_test.cpp"),
            "-L", str(NATIVE), "-lturbocider",
            "-L", str(library), "-lmlx", "-ljaccl", "-licucore",
            "-framework", "Foundation", "-framework", "Metal",
            "-Wl,-rpath," + str(NATIVE),
            "-Wl,-rpath," + str(library),
            "-o", str(binary),
        ], check=True)
        (temporary / "fixture").mkdir()
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
            print("SKIP: Qwen21 text streaming fixture requires Metal access")
            return
        if result.returncode:
            print(result.stdout, end="")
            print(result.stderr, end="", file=os.sys.stderr)
            raise SystemExit(result.returncode)
        if not result.stdout.startswith("PASS Qwen21 text streaming:"):
            print(result.stdout, end="")
            print(result.stderr, end="", file=os.sys.stderr)
            raise RuntimeError(
                "Qwen21 text streaming test exited without its PASS marker"
            )
        print(result.stdout, end="")


class TextStreamingTests(unittest.TestCase):
    def test_exact_bf16_release_and_retry(self):
        main()


if __name__ == "__main__":
    unittest.main()
