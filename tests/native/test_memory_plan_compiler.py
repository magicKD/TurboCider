#!/usr/bin/env python3
"""Compile and execute the dependency-free memory plan compiler tests."""

from memory_test_runner import run_memory_test


def main() -> int:
    return run_memory_test(
        "memory_plan_compiler",
        "runtime/memory_plan",
        "runtime/memory_schedule",
        "runtime/memory_manifest",
        "runtime/memory_policy",
        "core/common",
    )


if __name__ == "__main__":
    raise SystemExit(main())
