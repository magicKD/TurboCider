#!/usr/bin/env python3
"""Host-only Actual Receipt v2 tests; no GPU or model weights."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    compiler = subprocess.check_output(
        ["xcrun", "--find", "clang++"], text=True
    ).strip()
    sdk = subprocess.check_output(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True
    ).strip()
    flags = [
        "-std=c++20",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-isysroot",
        sdk,
        "-I",
        str(ROOT / "native/runtime"),
    ]
    sanitizer = os.environ.get("TC_STREAMING_SANITIZER", "")
    if sanitizer:
        if sanitizer not in ("address,undefined", "thread"):
            raise ValueError("unsupported sanitizer")
        flags += [
            "-O1",
            "-g",
            "-fno-omit-frame-pointer",
            "-fsanitize=" + sanitizer,
        ]
    else:
        flags += ["-O2"]
    sources = [
        ROOT / "tests/native/streaming_actual_receipt_test.cpp",
        ROOT / "native/runtime/streaming/actual_receipt.cpp",
        ROOT / "native/runtime/streaming/canonical_encoding.cpp",
        ROOT / "native/runtime/memory_manifest.cpp",
        ROOT / "native/core/common.cpp",
    ]
    with tempfile.TemporaryDirectory(prefix="tc-streaming-receipt-") as raw:
        binary = Path(raw) / "streaming-actual-receipt-test"
        subprocess.run(
            [compiler, *flags, *map(str, sources), "-o", str(binary)],
            check=True,
        )
        subprocess.run([str(binary)], check=True, timeout=30)


if __name__ == "__main__":
    main()
