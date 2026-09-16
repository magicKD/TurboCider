#!/usr/bin/env python3
"""Compile and execute the capability-to-execution gate tests."""

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
    with tempfile.TemporaryDirectory(prefix="turbocider-memory-execution-") as raw:
        binary = Path(raw) / "memory-execution-test"
        subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-isysroot",
                sdk,
                "-I",
                str(ROOT / "native/core"),
                "-I",
                str(ROOT / "native/runtime"),
                str(ROOT / "tests/native/memory_execution_test.cpp"),
                str(ROOT / "native/runtime/memory_execution.cpp"),
                str(ROOT / "native/runtime/memory_scheduler.cpp"),
                str(ROOT / "native/runtime/memory_watchdog.cpp"),
                str(ROOT / "native/runtime/memory_trace.cpp"),
                str(ROOT / "native/runtime/memory_accounting.cpp"),
                str(ROOT / "native/runtime/memory_plan.cpp"),
                str(ROOT / "native/runtime/memory_schedule.cpp"),
                str(ROOT / "native/runtime/memory_manifest.cpp"),
                str(ROOT / "native/runtime/memory_policy.cpp"),
                str(ROOT / "native/core/common.cpp"),
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
