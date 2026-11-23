#!/usr/bin/env python3
"""Compile and execute the capability-to-execution gate tests."""

from memory_test_runner import run_memory_test


def main() -> int:
    return run_memory_test(
        "memory_execution",
        "runtime/memory_execution",
        "runtime/memory_scheduler",
        "runtime/memory_watchdog",
        "runtime/memory_trace",
        "runtime/memory_accounting",
        "runtime/memory_plan",
        "runtime/memory_schedule",
        "runtime/memory_manifest",
        "runtime/memory_policy",
        "core/common",
    )


if __name__ == "__main__":
    raise SystemExit(main())
