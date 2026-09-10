#!/usr/bin/env python3
"""Compile and execute the dependency-free H3 streaming policy test."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    compiler = subprocess.run(
        ["xcrun", "--find", "clang"], check=True, capture_output=True, text=True
    ).stdout.strip()
    sdk = subprocess.run(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="turbocider-h3-stream-policy-") as raw:
        binary = Path(raw) / "h3-streaming-policy-test"
        subprocess.run(
            [
                compiler,
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-isysroot",
                sdk,
                "-I",
                str(ROOT / "native/models/h3_runtime"),
                str(ROOT / "tests/native/h3_streaming_policy_test.c"),
                str(ROOT / "native/models/h3_runtime/h3_streaming_policy.c"),
                str(ROOT / "native/runtime/block_residency.c"),
                "-o",
                str(binary),
            ],
            check=True,
        )
        completed = subprocess.run(
            [str(binary)], check=True, capture_output=True, text=True
        )
    print(completed.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
