"""Observer interval math and optional real Metal ownership/coverage checks."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("metal_trace", ROOT / "tools/profiling/image_startup/metal_trace.py")
TRACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRACE)


def setup(pid=42):
    return dict(type="setup", schema_version=1, pid=pid, clock="mach_absolute_time_ns",
                library_source=True, library_url=True, library_data=True,
                pipeline_function=True, pipeline_options=True, pipeline_descriptor=True)


class TraceMathTests(unittest.TestCase):
    def test_union_and_phase_clipping_do_not_double_count_nested_calls(self):
        def call(kind, start, end):
            return dict(type="metal_api", pid=42, kind=kind, start_ns=start, end_ns=end,
                        success=True, name="test")
        rows = [setup(), call("library_source", 0, 30), call("function", 5, 20),
                call("pipeline", 15, 40), call("library_load", 60, 70)]
        events = [dict(phase="denoise", completed=i, observed_monotonic_ns=t)
                  for i, t in enumerate((10, 50, 80))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("\n".join(map(json.dumps, rows)))
            result = TRACE.summarize_trace(path, 10, 80, events, 42)
        self.assertAlmostEqual(result["request"]["all_api_host_union_seconds"], 40e-9)
        self.assertAlmostEqual(result["request"]["build_api_host_union_seconds"], 30e-9)
        self.assertAlmostEqual(result["steps"][0]["build_api_host_union_seconds"], 30e-9)
        self.assertEqual(result["steps"][1]["build_api_host_union_seconds"], 0)

    def test_coverage_process_clock_and_truncation_fail_closed(self):
        for rows in ([], [setup()], [setup(99)], [dict(setup(), pipeline_descriptor=False)],
                     [dict(setup(), clock="other")], [setup(), dict(type="truncated")]):
            with self.subTest(rows=rows), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "trace.jsonl"
                path.write_text("\n".join(map(json.dumps, rows)))
                with self.assertRaises(ValueError):
                    TRACE.summarize_trace(path, 0, 100, [], 42)

    def test_adjacent_and_disjoint_intervals(self):
        self.assertEqual(TRACE.union_ns([]), 0)
        self.assertEqual(TRACE.union_ns([(1, 4), (4, 8), (2, 3), (10, 13)]), 10)

    def test_observer_is_not_part_of_shipping_builds(self):
        for relative in ("tools/native/build.sh", "tools/native/build_app.sh", "tools/native/package.sh"):
            self.assertNotIn("profiling/image_startup", (ROOT / relative).read_text())


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_METAL_TRACE") == "1", "opt-in real Metal observer test")
class MetalObserverTests(unittest.TestCase):
    def test_real_calls_ownership_errors_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="tc-metal-trace-") as directory:
            directory = Path(directory)
            dylib, binary = directory / "trace.dylib", directory / "probe"
            subprocess.run(["bash", str(ROOT / "tools/profiling/image_startup/build_metal_trace.sh"), str(dylib)], check=True)
            subprocess.run(["xcrun", "clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror", "-fno-objc-arc",
                            str(ROOT / "tests/native/metal_trace_probe.mm"), "-framework", "Foundation", "-framework", "Metal",
                            "-o", str(binary)], check=True)
            path = directory / "trace.jsonl"
            env = dict(os.environ, DYLD_INSERT_LIBRARIES=str(dylib), TURBOCIDER_METAL_TRACE_FILE=str(path))
            subprocess.run([str(binary)], env=env, check=True)
            rows = list(map(json.loads, path.read_text().splitlines()))
            self.assertEqual(rows[0]["type"], "setup")
            self.assertTrue(all(rows[0][key] for key in ("library_source", "pipeline_function", "pipeline_descriptor")))
            calls = [row for row in rows if row["type"] == "metal_api"]
            self.assertTrue(any(row["kind"] == "library_source" and not row["success"] for row in calls))
            self.assertTrue(any(row["kind"] == "function_specialize" for row in calls))
            self.assertTrue(any(row["kind"] == "pipeline" for row in calls))
            before = path.read_bytes()
            subprocess.run([str(binary)], env=env, check=True)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
