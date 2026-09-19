#!/usr/bin/env python3
"""CPU/source checks for the audit-only streaming instrumentation."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))

from run_streaming_audit import AuditError, evaluate_snapshot  # noqa: E402


SYMBOLS = (
    "tc_streaming_audit_reset",
    "tc_streaming_audit_snapshot_json",
)


def exported_symbols(path: Path) -> str:
    return subprocess.check_output(
        ["nm", "-gU", str(path)], text=True, stderr=subprocess.STDOUT
    )


class StreamingAuditTests(unittest.TestCase):
    def test_disabled_is_noop_and_enabled_counts(self):
        with tempfile.TemporaryDirectory(prefix="tc-streaming-audit-") as folder:
            root = Path(folder)
            common = [
                "clang++", "-std=c++20", "-Wall", "-Wextra", "-Werror",
                str(ROOT / "tests/native/streaming_audit_test.cpp"),
            ]
            disabled = root / "disabled"
            subprocess.run(common + ["-o", str(disabled)], check=True)
            subprocess.run([str(disabled)], check=True)
            enabled = root / "enabled"
            subprocess.run(common + [
                "-DTURBOCIDER_ENABLE_AUDIT_COUNTERS=1",
                str(ROOT / "native/runtime/streaming/audit.cpp"),
                "-o", str(enabled),
            ], check=True)
            subprocess.run([str(enabled)], check=True)

    def test_streaming_executor_reports_active_framework_in_audit_build(self):
        compiler = subprocess.check_output(
            ["xcrun", "--find", "clang++"], text=True
        ).strip()
        sdk = subprocess.check_output(
            ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True
        ).strip()
        sources = [
            "native/core/common.cpp",
            "native/core/json_keys.cpp",
            "native/runtime/memory_manifest.cpp",
            "native/runtime/streaming/config.cpp",
            "native/runtime/streaming/layout.cpp",
            "native/runtime/streaming/canonical_encoding.cpp",
            "native/runtime/streaming/actual_receipt.cpp",
            "native/runtime/streaming/slot_pool.cpp",
            "native/runtime/streaming/io_executor.cpp",
            "native/runtime/streaming/context.cpp",
            "native/runtime/streaming/audit.cpp",
        ]
        with tempfile.TemporaryDirectory(prefix="tc-streaming-audit-exec-") as folder:
            binary = Path(folder) / "streaming-executor-audit"
            subprocess.run([
                compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                "-DTURBOCIDER_ENABLE_AUDIT_COUNTERS=1", "-pthread",
                "-isysroot", sdk, "-I", str(ROOT / "native/runtime"),
                str(ROOT / "tests/native/streaming_executor_test.cpp"),
                *[str(ROOT / source) for source in sources],
                "-o", str(binary),
            ], check=True)
            subprocess.run([str(binary)], check=True, timeout=30)

    def test_build_callsite_and_private_abi_contract(self):
        build = (ROOT / "tools/native/build.sh").read_text()
        runner = (ROOT / "tools/native/run_streaming_audit.py").read_text()
        api = (ROOT / "native/api/c_api.mm").read_text()
        execution = (ROOT / "native/runtime/memory_execution.cpp").read_text()
        context = (ROOT / "native/runtime/streaming/context.cpp").read_text()
        io = (ROOT / "native/runtime/streaming/io_executor.cpp").read_text()
        public = (
            ROOT / "bindings/c/include/turbocider/turbocider.h"
        ).read_text()
        self.assertIn("TURBOCIDER_BUILD_AUDIT_COUNTERS", build)
        self.assertIn("native/runtime/streaming/audit.cpp", build)
        self.assertIn("TURBOCIDER_ENABLE_AUDIT_COUNTERS", api)
        self.assertIn("tc_engine_test_set_streaming_catalog_json", runner)
        self.assertIn("test_streaming_catalog_sha256", runner)
        for symbol in SYMBOLS:
            self.assertIn(symbol, api)
            self.assertNotIn(symbol, public)
        self.assertIn("AuditCounter::FrameworkHooks", api)
        self.assertIn("AuditCounter::CacheClearOrUnloadCalls", api)
        flux = (ROOT / "native/models/flux2/pipeline.cpp").read_text()
        self.assertEqual(
            flux.count("AuditCounter::CacheClearOrUnloadCalls"), 2
        )
        self.assertIn("AuditCounter::MemoryProbes", execution)
        self.assertIn("AuditCounter::PoolAllocations", context)
        self.assertIn("AuditCounter::SteadyFrameworkAllocations", context)
        self.assertIn("AuditCounter::WorkerThreads", io)

    def test_snapshot_evaluation_is_fail_closed(self):
        snapshot = {
            "format": "turbocider-streaming-audit-snapshot-v1",
            "new_framework_hooks": 0,
            "new_memory_probes": 0,
            "new_worker_threads": 0,
            "new_pool_allocations": 0,
            "new_cache_clear_or_unload_calls": 0,
            "steady_framework_allocations": 0,
            "steady_framework_thread_creates": 0,
        }
        self.assertTrue(
            evaluate_snapshot(snapshot, "default-zero", True)["passed"]
        )
        snapshot["new_worker_threads"] = 1
        self.assertFalse(
            evaluate_snapshot(snapshot, "default-zero", True)["passed"]
        )
        self.assertTrue(
            evaluate_snapshot(snapshot, "framework-active", False)["passed"]
        )
        snapshot["new_worker_threads"] = -1
        with self.assertRaises(AuditError):
            evaluate_snapshot(snapshot, "observe", True)

    def test_release_library_does_not_export_audit_symbols(self):
        path = Path(os.environ.get(
            "TURBOCIDER_RELEASE_LIBRARY",
            ROOT / "build/native/libturbocider.dylib",
        ))
        if not path.is_file():
            self.skipTest("release dylib is unavailable")
        symbols = exported_symbols(path)
        for symbol in SYMBOLS:
            self.assertNotIn(symbol, symbols)

    def test_audit_library_exports_private_symbols_when_provided(self):
        raw = os.environ.get("TURBOCIDER_AUDIT_LIBRARY")
        if not raw:
            self.skipTest("TURBOCIDER_AUDIT_LIBRARY was not provided")
        path = Path(raw)
        self.assertTrue(path.is_file())
        symbols = exported_symbols(path)
        for symbol in SYMBOLS:
            self.assertIn(symbol, symbols)
        for symbol in (
            "tc_engine_test_ltx_exact_destroy_failures",
            "tc_engine_test_ltx_exact_cancel_first_fill",
        ):
            self.assertNotIn(symbol, symbols)


if __name__ == "__main__":
    unittest.main(verbosity=2)
