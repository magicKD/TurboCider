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
    def test_route_switch_requires_matching_images_and_real_hybrid_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = []
            for index, route in enumerate(("gpu", "gpu_ane", "gpu", "gpu_ane")):
                output = root / f"image-{index}.png"
                output.write_bytes(b"gpu" if route == "gpu" else b"hybrid")
                runs.append({"route": route, "error": None,
                             "request": {"output": str(output)},
                             "metrics": {"runtime_backend": "mlx_cpp_metal" if route == "gpu"
                                         else "mlx_cpp_metal+coreml",
                                         "runtime_precision": "bf16" if route == "gpu" else
                                         "gpu_bf16+coreml_w8a8+image_only_ane_gpu_bf16_caption",
                                         "hybrid": {} if route == "gpu" else {
                                             "runtime_calls_session_total": 256,
                                             "runtime_failed": False,
                                             "output_copy_bytes_session_total": 0}}})
            result = BENCHMARK.verify_route_switch(runs, 8)
            self.assertTrue(result["verified"])
            self.assertFalse(result["speed_qualified"])
            self.assertEqual(result["hybrid_ffn_calls_per_request"], 256)
            runs[2]["metrics"]["runtime_backend"] = "mlx_cpp_metal+coreml"
            with self.assertRaisesRegex(RuntimeError, "return to BF16 GPU"):
                BENCHMARK.verify_route_switch(runs, 8)
            runs[2]["metrics"]["runtime_backend"] = "mlx_cpp_metal"
            runs[3]["metrics"]["hybrid"]["runtime_calls_session_total"] = 255
            with self.assertRaisesRegex(RuntimeError, "all W8A8 FFNs"):
                BENCHMARK.verify_route_switch(runs, 8)
            runs[3]["metrics"]["hybrid"]["runtime_calls_session_total"] = 256
            (root / "image-2.png").write_bytes(b"wrong")
            with self.assertRaisesRegex(RuntimeError, "changed the gpu image"):
                BENCHMARK.verify_route_switch(runs, 8)

    def test_route_switch_requires_explicit_single_session_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = str(ROOT / "tools/native/benchmark_z_image_metal.py")
            base = [sys.executable, script, "--model", "unused", "--output",
                    str(root / "out"), "--probe-route-switch"]
            for flags in ([], ["--worker", "fused"],
                          ["--ane-manifest", "unused"],
                          ["--worker", "fused", "--ane-manifest", "unused",
                           "--size", "768"],
                          ["--worker", "fused", "--ane-manifest", "unused",
                           "--probe-auto"]):
                result = subprocess.run(base + flags, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("--probe-route-switch requires", result.stderr)
                self.assertFalse((root / "out").exists())

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
        BENCHMARK.validate_presence_flags({"TURBOCIDER_Z_HYBRID_W8_DEQUANT_GEMM": "bf16"})
        BENCHMARK.validate_presence_flags({"TURBOCIDER_Z_HYBRID_W8_DEQUANT_GEMM": "gate_up_fp16"})
        BENCHMARK.validate_presence_flags({"TURBOCIDER_Z_HYBRID_W8_DEQUANT_GEMM": "scaled_all_fp16"})
        with self.assertRaisesRegex(ValueError, "must be bf16"):
            BENCHMARK.validate_presence_flags({"TURBOCIDER_Z_HYBRID_W8_DEQUANT_GEMM": "fp16"})

    def test_image_only_abba_enables_switches_only_in_hybrid_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / "fake.dylib"
            library.write_bytes(b"test-only")
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            calls = []

            def fake_run(cmd, env, check):
                calls.append((cmd, env))
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
                    "--control-default", "--hybrid-manifest", str(manifest),
                    "--hybrid-image-only"]
            with patch.object(sys, "argv", args), patch.dict(os.environ, {}, clear=True), \
                    patch.object(BENCHMARK.subprocess, "run", side_effect=fake_run):
                BENCHMARK.main()
            self.assertFalse(any(k.startswith("TURBOCIDER_Z_") for k in calls[0][1]))
            self.assertNotIn("--ane-manifest", calls[0][0])
            self.assertEqual(calls[1][1]["TURBOCIDER_Z_HYBRID_GPU_W8_DISABLE"], "1")
            self.assertEqual(calls[1][1]["TURBOCIDER_Z_W8A8_IMAGE_ONLY"], "1")
            self.assertEqual(calls[1][0][calls[1][0].index("--ane-manifest") + 1],
                             str(manifest))
            summary = json.loads((root / "out/summary.json").read_text())
            self.assertTrue(summary["conditions"]["hybrid_image_only"])

    def test_marked_image_only_abba_needs_no_research_switches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / "fake.dylib"
            library.write_bytes(b"test-only")
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            calls = []

            def fake_run(cmd, env, check):
                calls.append((cmd, env))
                dest = Path(cmd[cmd.index("--output") + 1])
                dest.mkdir(parents=True)
                report = {"variant": cmd[cmd.index("--worker") + 1],
                          "library_sha256": BENCHMARK.hashlib.sha256(b"test-only").hexdigest(),
                          "runs": [{"warmup": False, "parity": False, "wall_seconds": 2,
                                    "metrics": {"timings_seconds": {"denoise": 1}}}]}
                (dest / "report.json").write_text(json.dumps(report))

            args = ["benchmark", "--model", "unused", "--library", str(library),
                    "--output", str(root / "out"), "--schedule", "baseline,fused",
                    "--control-default", "--hybrid-manifest", str(manifest),
                    "--hybrid-image-only-marked"]
            with patch.object(sys, "argv", args), patch.dict(os.environ, {}, clear=True), \
                    patch.object(BENCHMARK.subprocess, "run", side_effect=fake_run):
                BENCHMARK.main()
            self.assertTrue(all(not any(k.startswith("TURBOCIDER_Z_") for k in env)
                                for _, env in calls))
            self.assertNotIn("--ane-manifest", calls[0][0])
            self.assertEqual(calls[1][0][calls[1][0].index("--ane-manifest") + 1],
                             str(manifest))
            summary = json.loads((root / "out/summary.json").read_text())
            self.assertTrue(summary["conditions"]["hybrid_image_only_marked"])

    def test_hybrid_pre_qkv_is_only_in_the_marked_candidate_arm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / "fake.dylib"
            library.write_bytes(b"test-only")
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            calls = []

            def fake_run(cmd, env, check):
                calls.append(env)
                dest = Path(cmd[cmd.index("--output") + 1])
                dest.mkdir(parents=True)
                report = {"variant": cmd[cmd.index("--worker") + 1],
                          "library_sha256": BENCHMARK.hashlib.sha256(b"test-only").hexdigest(),
                          "runs": [{"warmup": False, "parity": False, "wall_seconds": 2,
                                    "metrics": {"timings_seconds": {"denoise": 1}}}]}
                (dest / "report.json").write_text(json.dumps(report))

            args = ["benchmark", "--model", "unused", "--library", str(library),
                    "--output", str(root / "out"), "--schedule", "baseline,fused",
                    "--control-default", "--hybrid-manifest", str(manifest),
                    "--hybrid-image-only-marked", "--hybrid-pre-qkv"]
            with patch.object(sys, "argv", args), patch.dict(os.environ, {}, clear=True), \
                    patch.object(BENCHMARK.subprocess, "run", side_effect=fake_run):
                BENCHMARK.main()
            self.assertNotIn("TURBOCIDER_Z_HYBRID_FUSED_QKV", calls[0])
            self.assertEqual(calls[1]["TURBOCIDER_Z_HYBRID_FUSED_QKV"], "1")
            summary = json.loads((root / "out/summary.json").read_text())
            self.assertTrue(summary["conditions"]["hybrid_pre_qkv"])

    def test_hybrid_qkv_crossover_routes_both_arms_to_the_same_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / "fake.dylib"
            library.write_bytes(b"test-only")
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            calls = []

            def fake_run(cmd, env, check):
                calls.append((cmd, env))
                dest = Path(cmd[cmd.index("--output") + 1])
                dest.mkdir(parents=True)
                report = {"variant": cmd[cmd.index("--worker") + 1],
                          "library_sha256": BENCHMARK.hashlib.sha256(b"test-only").hexdigest(),
                          "runs": [{"warmup": False, "parity": False, "wall_seconds": 2,
                                    "metrics": {"timings_seconds": {"denoise": 1}}}]}
                (dest / "report.json").write_text(json.dumps(report))

            args = ["benchmark", "--model", "unused", "--library", str(library),
                    "--output", str(root / "out"), "--schedule", "baseline,fused",
                    "--control-default", "--hybrid-manifest", str(manifest),
                    "--hybrid-image-only-marked", "--hybrid-pre-qkv",
                    "--hybrid-qkv-crossover"]
            with patch.object(sys, "argv", args), patch.dict(os.environ, {}, clear=True), \
                    patch.object(BENCHMARK.subprocess, "run", side_effect=fake_run):
                BENCHMARK.main()
            self.assertEqual(len(calls), 2)
            for cmd, _ in calls:
                self.assertEqual(cmd[cmd.index("--ane-manifest") + 1], str(manifest))
            self.assertNotIn("TURBOCIDER_Z_HYBRID_FUSED_QKV", calls[0][1])
            self.assertEqual(calls[0][1]["TURBOCIDER_Z_HYBRID_DISABLE_FUSED_QKV"], "1")
            self.assertEqual(calls[1][1]["TURBOCIDER_Z_HYBRID_FUSED_QKV"], "1")
            self.assertNotIn("TURBOCIDER_Z_HYBRID_DISABLE_FUSED_QKV", calls[1][1])
            summary = json.loads((root / "out/summary.json").read_text())
            self.assertEqual(summary["conditions"]["control"], "hybrid_default")

    def test_hybrid_gate_crossover_uses_default_qkv_in_both_arms(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / "fake.dylib"
            library.write_bytes(b"test-only")
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            calls = []

            def fake_run(cmd, env, check):
                calls.append((cmd, env))
                dest = Path(cmd[cmd.index("--output") + 1])
                dest.mkdir(parents=True)
                (dest / "report.json").write_text(json.dumps({
                    "variant": cmd[cmd.index("--worker") + 1],
                    "library_sha256": BENCHMARK.hashlib.sha256(b"test-only").hexdigest(),
                    "runs": [{"warmup": False, "parity": False, "wall_seconds": 2,
                              "metrics": {"timings_seconds": {"denoise": 1}}}]}))

            args = ["benchmark", "--model", "unused", "--library", str(library),
                    "--output", str(root / "out"), "--schedule", "baseline,fused",
                    "--control-default", "--hybrid-manifest", str(manifest),
                    "--hybrid-image-only-marked", "--hybrid-pre-gate-norm",
                    "--hybrid-gate-crossover"]
            with patch.object(sys, "argv", args), patch.dict(os.environ, {}, clear=True), \
                    patch.object(BENCHMARK.subprocess, "run", side_effect=fake_run):
                BENCHMARK.main()
            for cmd, _ in calls:
                self.assertEqual(cmd[cmd.index("--ane-manifest") + 1], str(manifest))
            self.assertNotIn("TURBOCIDER_Z_HYBRID_DISABLE_FUSED_QKV", calls[0][1])
            self.assertNotIn("TURBOCIDER_Z_HYBRID_FUSED_GATE_NORM", calls[0][1])
            self.assertEqual(calls[0][1]["TURBOCIDER_Z_HYBRID_DISABLE_FUSED_GATE_NORM"], "1")
            self.assertEqual(calls[1][1]["TURBOCIDER_Z_HYBRID_FUSED_GATE_NORM"], "1")
            self.assertNotIn("TURBOCIDER_Z_HYBRID_DISABLE_FUSED_GATE_NORM", calls[1][1])
            self.assertEqual(json.loads((root / "out/summary.json").read_text())
                             ["conditions"]["control"], "hybrid_default_qkv")

    def test_image_only_abba_rejects_missing_manifest_or_gpu_w8_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = str(ROOT / "tools/native/benchmark_z_image_metal.py")
            base = [sys.executable, script, "--model", "unused", "--output",
                    str(root / "out"), "--hybrid-image-only", "--control-default"]
            for extra in ([], ["--hybrid-manifest", "unused", "--hybrid-w8-mode", "bf16"]):
                result = subprocess.run(base + extra, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("--hybrid-image-only requires", result.stderr)
            self.assertFalse((root / "out").exists())

    def test_marked_image_only_rejects_missing_manifest_or_mixed_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = str(ROOT / "tools/native/benchmark_z_image_metal.py")
            base = [sys.executable, script, "--model", "unused", "--output",
                    str(root / "out"), "--control-default"]
            for flags, reason in (
                (["--hybrid-image-only-marked"], "--hybrid-image-only-marked requires"),
                (["--hybrid-image-only-marked", "--hybrid-manifest", "unused",
                  "--hybrid-w8-mode", "bf16"], "--hybrid-image-only-marked requires"),
                (["--hybrid-image-only", "--hybrid-image-only-marked",
                  "--hybrid-manifest", "unused"], "choose one image-only"),
            ):
                result = subprocess.run(base + flags, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn(reason, result.stderr)
                self.assertFalse((root / "out").exists())

    def test_hybrid_pre_qkv_rejects_unmarked_or_gpu_only_experiment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = str(ROOT / "tools/native/benchmark_z_image_metal.py")
            base = [sys.executable, script, "--model", "unused", "--output",
                    str(root / "out"), "--control-default", "--hybrid-pre-qkv"]
            for flags in ([], ["--hybrid-manifest", "unused"]):
                result = subprocess.run(base + flags, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("--hybrid-pre-qkv requires", result.stderr)
                self.assertFalse((root / "out").exists())
            result = subprocess.run(base[:-1] + ["--hybrid-image-only-marked",
                "--hybrid-manifest", "unused", "--hybrid-qkv-crossover"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("--hybrid-qkv-crossover requires", result.stderr)

    def test_negative_worker_probes_require_manifest_and_worker_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = str(ROOT / "tools/native/benchmark_z_image_metal.py")
            base = [sys.executable, script, "--model", "unused", "--output",
                    str(root / "out")]
            for flags, reason in (
                (["--omit-approximation"], "--omit-approximation requires"),
                (["--probe-auto"], "--probe-auto requires"),
                (["--worker", "fused", "--probe-auto"], "--probe-auto requires"),
                (["--worker", "fused", "--ane-manifest", "unused",
                  "--probe-auto", "--omit-approximation"], "--probe-auto requires"),
            ):
                result = subprocess.run(base + flags, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn(reason, result.stderr)
                self.assertFalse((root / "out").exists())

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

    def test_capture_once_requires_explicit_capture_and_zero_warm_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "not-created"
            script = str(ROOT / "tools/native/benchmark_z_image_metal.py")
            base = [sys.executable, script, "--model", "unused", "--output", str(output),
                    "--worker", "fused", "--capture-once", "--runs", "0"]
            env = {k: v for k, v in os.environ.items() if not k.startswith("TURBOCIDER_Z_")}
            for command, settings in ((base, env),
                                      (base[:-1] + ["1"], {**env,
                                       "TURBOCIDER_Z_FFN_CAPTURE_DIR": temporary}),
                                      (base[0:base.index("--worker")] +
                                       base[base.index("--capture-once"):], {**env,
                                       "TURBOCIDER_Z_FFN_CAPTURE_DIR": temporary})):
                result = subprocess.run(command, env=settings, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("--capture-once requires", result.stderr)
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

    def test_health_anchor_does_not_misclassify_long_caption(self):
        reports = [{"variant": "baseline", "hardware": {"gpu": "Apple M4 Max"},
                    "runs": [{"warmup": False, "parity": False,
                              "metrics": {"text_tokens": 494,
                                          "timings_seconds": {"denoise": 10.05,
                                                              "vae_decode": .22}}}]}]
        result = BENCHMARK.assess_system_state(reports, {"size": 512, "steps": 8})
        self.assertEqual(result["status"], "not_applicable")
        self.assertIn("128", result["reason"])
        reports[0]["runs"][0]["metrics"]["timings_seconds"]["vae_decode"] = 5.0
        result = BENCHMARK.assess_system_state(reports, {"size": 512, "steps": 8})
        self.assertEqual(result["status"], "invalid_for_performance")

    def test_partial_schedule(self):
        self.assertEqual(BENCHMARK.summarize_reports([]), {})
        self.assertEqual(BENCHMARK.summarize_reports([
            {"variant": "baseline", "runs": []}]), {})


if __name__ == "__main__":
    unittest.main()
