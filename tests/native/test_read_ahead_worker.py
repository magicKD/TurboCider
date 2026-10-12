#!/usr/bin/env python3
"""CPU-only bounded pread worker ownership/error/cancellation fixture."""
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    compiler = subprocess.check_output(["xcrun", "--find", "clang++"], text=True).strip()
    sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
    flags = ["-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread", "-isysroot", sdk]
    sanitizer = os.environ.get("TC_STREAMING_SANITIZER", "")
    if sanitizer:
        if sanitizer not in ("address,undefined", "thread"):
            raise ValueError("unsupported sanitizer")
        flags += ["-O1", "-g", "-fno-omit-frame-pointer", "-fsanitize=" + sanitizer]
    sources = ["tests/native/read_ahead_worker_test.cpp", "native/runtime/streaming/source_lease.cpp",
               "native/runtime/streaming/canonical_encoding.cpp", "native/runtime/memory_manifest.cpp",
               "native/core/common.cpp", "native/core/json_keys.cpp"]
    with tempfile.TemporaryDirectory(prefix="tc-read-ahead-") as raw:
        binary = Path(raw) / "read-ahead-test"
        subprocess.run([compiler, *flags, *(str(ROOT / p) for p in sources), "-o", str(binary)], check=True)
        result = subprocess.run([str(binary), str(Path(raw) / "fixture.bin")], text=True, capture_output=True, timeout=30)
        if result.returncode or not result.stdout.startswith("PASS bounded pread worker:"):
            raise RuntimeError(result.stdout + result.stderr)
        print(result.stdout, end="")


if __name__ == "__main__":
    main()
