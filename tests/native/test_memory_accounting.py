#!/usr/bin/env python3
"""Compile and execute the dependency-free memory ledger tests."""

from memory_test_runner import run_memory_test


def main() -> int:
    return run_memory_test("memory_accounting", "runtime/memory_accounting", "core/common")


if __name__ == "__main__":
    raise SystemExit(main())
