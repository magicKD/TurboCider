import ast
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from flux9_bf16_framework_screen import generation, check_generations, registered_hashes, requests, validate_native
from unittest.mock import patch


class Flux9FrameworkScreenTests(unittest.TestCase):
    def test_original_weight_checks_do_not_read_payloads(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "weights.bin"
            path.write_bytes(b"source")
            snapshot = generation(path)
            with patch.object(Path, "open", side_effect=AssertionError("payload was read")):
                check_generations({str(path): snapshot})
            path.write_bytes(b"change")
            os.utime(path, ns=(path.stat().st_atime_ns, snapshot["mtime_ns"]))
            with self.assertRaises(ValueError):
                check_generations({str(path): snapshot})

    def test_empty_python_package_valid_but_empty_model_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            source, model = Path(raw) / "__init__.py", Path(raw) / "model.safetensors"
            source.touch(); model.touch()
            self.assertEqual(generation(source)["bytes"], 0)
            with self.assertRaises(ValueError):
                generation(model)

    def test_unchanged_generation_reuses_initial_sha_and_changed_source_rechecks(self):
        with tempfile.TemporaryDirectory() as raw:
            path, registration = Path(raw) / "model.bin", Path(raw) / "registrations/source.json"
            path.write_bytes(b"original")
            first = registered_hashes([path], registration)
            with patch("flux9_bf16_framework_screen.sha256_file", side_effect=AssertionError("payload reread")):
                self.assertEqual(registered_hashes([path], registration), first)
            path.write_bytes(b"modified")
            os.utime(path, ns=(path.stat().st_atime_ns, 0))
            second = registered_hashes([path], registration)
            self.assertNotEqual(first, second)

    def test_matching_fresh_then_repeat_workload_and_actual_receipts(self):
        task = requests(Path("/tmp/not-written"))
        self.assertNotEqual(task[1]["prompt"], task[2]["prompt"])
        self.assertEqual(task[2]["prompt"], task[3]["prompt"])
        self.assertTrue(all(row["compile_gpu"] is False for row in task))
        self.assertTrue(all(row["compile_gpu"] is True for row in requests(Path("/tmp/not-written"), True)))
        rows = [dict(request_wall_including_export=1.0, metrics=dict(
            model="flux2-klein-9b", operation="image.generate", width=512, height=512,
            steps=4, seed=42, text_tokens=512, prompt_cache_hit=index == 3, hybrid=None)) for index in range(4)]
        validate_native(rows)
        rows[3]["metrics"]["prompt_cache_hit"] = False
        with self.assertRaises(ValueError):
            validate_native(rows)

    def test_benchmark_native_helper_remains_parseable(self):
        ast.parse((ROOT / "tools/native/benchmark_native.py").read_text())

    def test_hybrid_requests_are_explicit_and_distinct_from_compiled_gpu(self):
        task = requests(Path("/tmp/not-written"), ane_manifest="/tmp/source-bound.json")
        self.assertTrue(all(row["execution"] == "gpu_ane" and row["allow_approximation"] and
                            row["warmup_iterations"] == 0 and row["compile_gpu"] is False and
                            row["ane_manifest"] == "/tmp/source-bound.json" and row["dynamic_text"] is False
                            for row in task))
        with self.assertRaises(ValueError):
            requests(Path("/tmp/not-written"), True, "/tmp/source-bound.json")

    def test_hybrid_requires_actual_geometry_source_success_and_all_calls(self):
        rows = [dict(request_wall_including_export=1.0, native_event_counts={}, metrics=dict(
            model="flux2-klein-9b", operation="image.generate", width=512, height=512,
            steps=4, seed=42, text_tokens=512, prompt_cache_hit=index == 3, hybrid=dict(
                bucket=1536, hidden=4096, output_channels=4096, block_count=24, mlp_width=12288,
                ane_mlp_range=[0,4096], checkpoint_sha256_verified=True, runtime_failed=False,
                runtime_failures_session_total=0, runtime_calls_session_total=96*(index+1),
                calls_session_total=96*(index+1), warmup_calls_session_total=0,
                weight_variant="int8_pc", compute_units="cpuAndNeuralEngine"))) for index in range(4)]
        validate_native(rows, True)
        with self.assertRaises(ValueError):
            validate_native(rows)
        mutations = dict(bucket=1024, hidden=3072, block_count=20, mlp_width=9216,
            ane_mlp_range=[0,5120], checkpoint_sha256_verified=False, runtime_failed=True,
            runtime_failures_session_total=1, runtime_calls_session_total=95,
            calls_session_total=95, warmup_calls_session_total=24, weight_variant="fp16",
            compute_units="all")
        for key, value in mutations.items():
            with self.subTest(field=key):
                bad = copy.deepcopy(rows)
                bad[0]["metrics"]["hybrid"][key] = value
                with self.assertRaises(ValueError):
                    validate_native(bad, True)
        fallback = copy.deepcopy(rows)
        fallback[1]["native_event_counts"]["acceleration_gpu_fallback"] = 1
        with self.assertRaises(ValueError):
            validate_native(fallback, True)

    def test_portable_evidence_binds_actual_fresh_samples_and_rejection(self):
        data = json.loads((ROOT / "docs/design/validation/flux9-bf16-framework-20261011.json").read_text())
        self.assertEqual(data["status"], "complete_diagnostic")
        self.assertEqual(data["order"], ["native", "mflux", "mflux", "native"])
        self.assertEqual(len(data["trials"]), 4)
        self.assertTrue(all(row["memory"]["complete"] is True for row in data["trials"]))
        for trial in data["trials"]:
            if trial["engine"] == "mflux":
                self.assertEqual(trial["report"]["instrumentation"],
                                 "host-step-counter-only; no extra GPU sync or encoder wrapper")
                self.assertTrue(all("timings_seconds" not in row for row in trial["report"]["runs"]))
        for engine in ("native", "mflux"):
            samples = [row["request_wall_including_export"] for trial in data["trials"]
                       if trial["engine"] == engine for row in trial["report"]["runs"][1:3]]
            self.assertEqual(data["medians"][engine]["samples"], samples)
        self.assertEqual(data["mlxserve"]["result"]["status"], "startup_rejected")
        self.assertIs(data["mlxserve"]["result"]["latency_comparison_valid"], False)
        self.assertFalse(data["mlxserve"]["result"]["requests"])
        self.assertNotIn("/Users/", json.dumps(data))


if __name__ == "__main__":
    unittest.main()
