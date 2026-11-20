#!/usr/bin/env python3
"""Compile and execute the dependency-free memory watchdog tests."""

from memory_test_runner import run_memory_test


def main() -> int:
    return run_memory_test("memory_watchdog", "runtime/memory_watchdog", "core/common")


if __name__ == "__main__":
    raise SystemExit(main())
