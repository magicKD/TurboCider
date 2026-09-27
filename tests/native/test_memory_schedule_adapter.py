#!/usr/bin/env python3
"""Compile and execute the dependency-free C schedule adapter tests."""

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
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with tempfile.TemporaryDirectory(
        prefix="turbocider-memory-schedule-adapter-"
    ) as raw:
        binary = Path(raw) / "memory-schedule-adapter-test"
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
                str(ROOT / "native/core"),
                str(ROOT / "tests/native/memory_schedule_adapter_test.c"),
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
