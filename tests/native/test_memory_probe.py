#!/usr/bin/env python3
"""Compile and execute metadata-only capability probe tests."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    compiler = subprocess.run(
        ["xcrun", "--find", "clang++"], check=True, capture_output=True, text=True
    ).stdout.strip()
    sdk = subprocess.run(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="turbocider-memory-probe-") as raw:
        binary = Path(raw) / "memory-probe-test"
        subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-fobjc-arc",
                "-isysroot",
                sdk,
                "-I",
                str(ROOT / "native/core"),
                "-I",
                str(ROOT / "native/runtime"),
                "-I",
                str(ROOT / "native/platform/apple"),
                str(ROOT / "tests/native/memory_probe_test.mm"),
                str(ROOT / "native/platform/apple/memory_probe.mm"),
                str(ROOT / "native/runtime/memory_manifest.cpp"),
                str(ROOT / "native/runtime/memory_policy.cpp"),
                str(ROOT / "native/core/common.cpp"),
                "-framework",
                "Foundation",
                "-o",
                str(binary),
            ],
            check=True,
        )
        completed = subprocess.run(
            [str(binary)], check=False, capture_output=True, text=True
        )
    if completed.stdout:
        print(completed.stdout.strip())
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr.strip())
        return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
