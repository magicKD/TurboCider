#!/usr/bin/env python3
"""Compile and execute the dependency-free bounded memory trace tests."""

from memory_test_runner import run_memory_test


def main() -> int:
    return run_memory_test("memory_trace", "runtime/memory_trace")


if __name__ == "__main__":
    raise SystemExit(main())
