"""CPU-only checks for the performance report accounting."""
import importlib.util
from pathlib import Path
import unittest
import subprocess
import sys
import tempfile
import os
import json
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "z_image_benchmark", ROOT / "tools/native/benchmark_z_image_metal.py")
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class ZImageBenchmarkTests(unittest.TestCase):
    def test_default_control_only_changes_candidate_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / "fake.dylib"
            library.write_bytes(b"test-only")
            calls = []

            def fake_run(cmd, env, check):
                calls.append(env)
                dest = Path(cmd[cmd.index("--output") + 1])
                dest.mkdir(parents=True)
                report = {
                    "variant": cmd[cmd.index("--worker") + 1],
                    "library_sha256": BENCHMARK.hashlib.sha256(b"test-only").hexdigest(),
                    "runs": [{"warmup": False, "parity": False, "wall_seconds": 2,
                              "metrics": {"timings_seconds": {"denoise": 1}}}],
                }
                (dest / "report.json").write_text(json.dumps(report))

            args = ["benchmark", "--model", "unused", "--library", str(library),
                    "--output", str(root / "out"), "--schedule", "baseline,fused",
                    "--control-default", "--virtual-norm-threads", "256", "--mpp",
                    "--gate-norm-virtual-threads", "256", "--cache-context"]
            with patch.object(sys, "argv", args), patch.dict(os.environ, {}, clear=True), \
                    patch.object(BENCHMARK.subprocess, "run", side_effect=fake_run):
                BENCHMARK.main()
            self.assertFalse(any(k.startswith("TURBOCIDER_Z_") for k in calls[0]))
            self.assertEqual(calls[1]["TURBOCIDER_Z_VIRTUAL_NORM_THREADS"], "256")
            self.assertEqual(calls[1]["TURBOCIDER_Z_MPP_SWIGLU"], "1")
            self.assertEqual(calls[1]["TURBOCIDER_Z_GATE_NORM_VIRTUAL_THREADS"], "256")
            self.assertEqual(calls[1]["TURBOCIDER_Z_CACHE_CONTEXT"], "1")
            summary = json.loads((root / "out/summary.json").read_text())
            self.assertEqual(summary["conditions"]["control"], "production_default")

    def test_presence_switch_requires_unset_for_disabled(self):
        for key in BENCHMARK.PRESENCE_FLAGS:
            for value in ("0", "false", "False", " off ", "no", ""):
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, "unset it to disable"):
                        BENCHMARK.validate_presence_flags({key: value})
        BENCHMARK.validate_presence_flags({})
        BENCHMARK.validate_presence_flags({"TURBOCIDER_Z_MPP_SWIGLU": "1"})
        BENCHMARK.validate_presence_flags({"TURBOCIDER_Z_NORM_THREADS": "512"})

    def test_worker_rejects_false_presence_switch_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "not-created"
            env = {k: v for k, v in os.environ.items() if not k.startswith("TURBOCIDER_Z_")}
            env["TURBOCIDER_Z_MPP_SWIGLU"] = "0"
            result = subprocess.run(
                [sys.executable, str(ROOT / "tools/native/benchmark_z_image_metal.py"),
                 "--model", "unused", "--output", str(output), "--worker", "fused"],
                env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("unset it to disable", result.stderr)
            self.assertFalse(output.exists())

    def test_worker_rejects_ignored_experiment_flags(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "not-created"
            for flag in (["--norm-threads", "512"], ["--norm-vector"], ["--scalar-norm"],
                         ["--mpp"], ["--no-mpp"], ["--virtual-norm-threads", "256"],
                         ["--control-default"], ["--control-legacy-swiglu"],
                         ["--gate-norm-virtual-threads", "256"], ["--cache-context"],
                         ["--mpp-qkv-prepare"]):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "tools/native/benchmark_z_image_metal.py"),
                     "--model", "unused", "--output", str(output),
                     "--worker", "fused", *flag], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("worker mode requires kernel environment variables", result.stderr)
                self.assertFalse(output.exists())

    def test_loaded_runtime_library_metadata(self):
        for name, record in BENCHMARK.loaded_runtime_libraries().items():
            self.assertIn(name, {"libmlx.dylib", "libjaccl.dylib", "libturbocider.dylib"})
            self.assertTrue(Path(record["path"]).is_absolute())
            self.assertEqual(len(record["sha256"]), 64)

    def test_separate_wall_and_denoise_and_exclude_untimed(self):
        def row(wall, denoise, warmup=False, parity=False):
            return {"warmup": warmup, "parity": parity, "wall_seconds": wall,
                    "metrics": {"timings_seconds": {"denoise": denoise}}}

        result = BENCHMARK.summarize_reports([
            {"variant": "baseline", "runs": [row(100, 90, warmup=True), row(30, 24)]},
            {"variant": "fused", "runs": [row(21, 15), row(200, 190, parity=True)]},
            {"variant": "baseline", "runs": [row(32, 26)]},
        ])
        self.assertEqual(result["baseline"]["warm_samples"], 2)
        self.assertEqual(result["baseline"]["median_wall_seconds"], 31)
        self.assertEqual(result["baseline"]["median_timings_seconds"]["denoise"], 25)
        self.assertEqual(result["fused"]["warm_samples"], 1)
        self.assertAlmostEqual(result["wall_speedup"], 31 / 21)

    def test_health_anchor_rejects_pre_reboot_scale_slowdown(self):
        def report(denoise, vae):
            return {"variant": "fused", "hardware": {"gpu": "Apple M4 Max"},
                    "runs": [{"warmup": False, "parity": False,
                              "metrics": {"timings_seconds": {
                                  "denoise": denoise, "vae_decode": vae}}}]}
        conditions = {"size": 512, "steps": 8}
        self.assertEqual(BENCHMARK.assess_system_state([report(7.0, .25)], conditions)["status"],
                         "healthy")
        self.assertEqual(BENCHMARK.assess_system_state([report(14.7, 5.0)], conditions)["status"],
                         "invalid_for_performance")

    def test_partial_schedule(self):
        self.assertEqual(BENCHMARK.summarize_reports([]), {})
        self.assertEqual(BENCHMARK.summarize_reports([
            {"variant": "baseline", "runs": []}]), {})


if __name__ == "__main__":
    unittest.main()
