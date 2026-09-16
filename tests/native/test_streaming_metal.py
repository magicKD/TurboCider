#!/usr/bin/env python3
"""Small real-Metal slot-lifecycle test; no model weights or pressure workload."""
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    compiler = subprocess.check_output(["xcrun", "--find", "clang++"], text=True).strip()
    sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
    with tempfile.TemporaryDirectory(prefix="tc-streaming-metal-") as raw:
        binary = Path(raw) / "test"
        sources = ["tests/native/streaming_metal_test.mm", "native/runtime/streaming/context.cpp",
                   "native/runtime/streaming/slot_pool.cpp", "native/runtime/streaming/io_executor.cpp"]
        subprocess.run([compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-fobjc-arc",
                        "-isysroot", sdk, "-I", str(ROOT / "native/runtime"),
                        *[str(ROOT / s) for s in sources], "-framework", "Foundation",
                        "-framework", "Metal", "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True, timeout=60)


if __name__ == "__main__":
    main()
