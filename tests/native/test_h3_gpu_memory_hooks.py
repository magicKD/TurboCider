#!/usr/bin/env python3
"""Compile and run the H3 Metal allocator-admission hook test."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
H3 = ROOT / "native/models/h3_runtime"


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
    with tempfile.TemporaryDirectory(prefix="turbocider-h3-hooks-") as raw:
        directory = Path(raw)
        gpu_object = directory / "h3_gpu.o"
        test_object = directory / "h3_gpu_memory_hooks_test.o"
        binary = directory / "h3-gpu-memory-hooks-test"
        subprocess.run([
            clang, "-std=c11", "-O0", "-g", "-fobjc-arc",
            "-D_DARWIN_C_SOURCE", "-isysroot", sdk, "-I", str(H3), "-c",
            str(H3 / "h3_gpu.m"), "-o", str(gpu_object),
        ], check=True)
        subprocess.run([
            clangxx, "-std=c++20", "-O0", "-g", "-fobjc-arc",
            "-D_DARWIN_C_SOURCE", "-isysroot", sdk, "-I", str(H3), "-c",
            str(ROOT / "tests/native/h3_gpu_memory_hooks_test.mm"),
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
            [str(binary), str(H3 / "h3_shaders.metal")],
            check=True, capture_output=True, text=True,
        )
    print(completed.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
