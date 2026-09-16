#!/usr/bin/env python3
"""Compile and run H3 schedule host-accounting tests without Metal."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
H3 = ROOT / "native/models/h3_runtime"


def main() -> int:
    compiler = subprocess.run(
        ["xcrun", "--find", "clang"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    sdk = subprocess.run(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    with tempfile.TemporaryDirectory(prefix="turbocider-h3-schedule-memory-") as raw:
        binary = Path(raw) / "h3-schedule-memory-test"
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
                str(H3),
                str(ROOT / "tests/native/h3_schedule_memory_test.c"),
                "-o",
                str(binary),
            ],
            check=True,
        )
        completed = subprocess.run(
            [str(binary)], check=True, capture_output=True, text=True,
        )
    print(completed.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
