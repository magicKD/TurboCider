#!/usr/bin/env python3
"""Build and run the CPU-only SourceLease/ValueProbe fixture."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    compiler = subprocess.check_output(["xcrun", "--find", "clang++"], text=True).strip()
    sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
    flags = ["-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread", "-isysroot", sdk,
             "-I", str(ROOT / "native"), "-I", str(ROOT / "native/runtime")]
    sanitizer = os.environ.get("TC_STREAMING_SANITIZER", "")
    if sanitizer:
        if sanitizer not in ("address,undefined", "thread"):
            raise ValueError("unsupported sanitizer")
        flags += ["-O1", "-g", "-fno-omit-frame-pointer", "-fsanitize=" + sanitizer]
    sources = [
        ROOT / "tests/native/streaming_source_lease_test.cpp",
        ROOT / "native/runtime/streaming/source_lease.cpp",
        ROOT / "native/runtime/streaming/value_probe.cpp",
        ROOT / "native/runtime/streaming/canonical_encoding.cpp",
        ROOT / "native/runtime/memory_manifest.cpp",
        ROOT / "native/core/common.cpp",
        ROOT / "native/core/json_keys.cpp",
    ]
    with tempfile.TemporaryDirectory(prefix="tc-streaming-source-lease-") as raw:
        binary = Path(raw) / "source-lease-test"
        subprocess.run([compiler, *flags, *(str(source) for source in sources), "-o", str(binary)], check=True)
        fixture = Path(raw) / "fixture"
        result = subprocess.run([str(binary), str(fixture)], text=True, capture_output=True, timeout=30)
        if result.returncode:
            print(result.stdout, end="")
            print(result.stderr, end="")
            raise SystemExit(result.returncode)
        if not result.stdout.startswith("PASS source lease/value probe:"):
            print(result.stdout, end="")
            print(result.stderr, end="")
            raise RuntimeError("source lease fixture exited without PASS marker")
        print(result.stdout, end="")


if __name__ == "__main__":
    main()
