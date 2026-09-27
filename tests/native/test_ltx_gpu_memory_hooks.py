#!/usr/bin/env python3
"""Compile and run the Metal-buffer admission hook test when Metal exists."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LTX = ROOT / "native/models/ltx_runtime"


def main() -> int:
    clang = subprocess.run(
        ["xcrun", "--find", "clang"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    clangxx = subprocess.run(
        ["xcrun", "--find", "clang++"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    sdk = subprocess.run(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="turbocider-ltx-hooks-") as raw:
        directory = Path(raw)
        gpu_object = directory / "ltx_gpu.o"
        test_object = directory / "ltx_gpu_memory_hooks_test.o"
        binary = directory / "ltx-gpu-memory-hooks-test"
        subprocess.run([
            clang, "-std=c11", "-O0", "-g", "-fobjc-arc",
            "-isysroot", sdk, "-I", str(LTX), "-c",
            str(LTX / "ltx_gpu.m"), "-o", str(gpu_object),
        ], check=True)
        subprocess.run([
            clangxx, "-std=c++20", "-O0", "-g", "-fobjc-arc",
            "-isysroot", sdk, "-I", str(LTX), "-c",
            str(ROOT / "tests/native/ltx_gpu_memory_hooks_test.mm"),
            "-o", str(test_object),
        ], check=True)
        subprocess.run([
            clangxx, "-isysroot", sdk, str(test_object), str(gpu_object),
            "-framework", "Foundation", "-framework", "Metal",
            "-framework", "MetalPerformanceShaders",
            "-framework", "MetalPerformanceShadersGraph",
            "-o", str(binary),
        ], check=True)
        completed = subprocess.run(
            [str(binary), str(LTX / "ltx_shaders.metal")],
            check=False, capture_output=True, text=True,
        )
    print(completed.stdout.strip())
    if completed.returncode:
        print(completed.stderr.strip())
        completed.check_returncode()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
