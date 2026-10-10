"""CPU-only tests for a bounded supervisor; no real CLI/device/model is run."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import signal
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
import zlib


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "native_lora_pair", ROOT / "tools/validation/run_qwen_native_lora_e2e.py")
PAIR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAIR)
BUILD = "tc-runtime-build-v1-" + "1" * 64


def png_header(width=512, height=512):
    ihdr = b"IHDR" + struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + ihdr
            + struct.pack(">I", zlib.crc32(ihdr) & 0xffffffff)
            + b"\x00\x00\x00\x00IEND\xaeB`\x82")


def result_for(route, request):
    result = dict(model="qwen-image-2.1", operation="image.generate", output=request["output"],
                  actual_denoise_steps=6, steps=6, width=512, height=512, seed=42, reference_tokens=0,
                  warmup=False, prompt_cache_hit=False, runtime_backend=route["runtime_backend"],
                  lora_strategy="inference_time", lora_applied_projections=227,
                  plan=dict(qwen21_dit_cache="off", algorithm_approximations=[PAIR.STUDENT_APPROXIMATION],
                            execution="gpu"),
                  timings_seconds=dict(request_wall=4.0, denoise=2.0))
    if route["name"] == "native":
        result["gpu_graph"] = "native_full_dit_euler_runtime_lora"
        result["runtime_precision"] = "bf16_gpu+runtime_lora_fp32_rank"
        result["native_denoiser_lora"] = dict(
            schema_version=1, enabled=True, completed=True, variant="viggle-r128-6step",
            source_kind="verified-held-fd", source_generation=1,
            verified_artifact_digest="a" * 64, adapter_sha256=PAIR.ADAPTER_SHA256,
            adapter_revision="b" * 64, adapter_count=1, resolved_bindings=195,
            logical_projections=227, ordered_terms=227, max_rank=128, source_strength=1.0,
            completed_steps=6, scratch_bytes=12345,
            precision_recipe=PAIR.LORA_RECIPES[request.get("qwen21_native_lora_kernel", "scalar")])
        result["native_denoiser_memory_phase"] = dict(memory_admission=dict(
            execution_state="succeeded", worker_quarantined=False, tainted=False,
            **{key: 0 for key in PAIR.ZERO_LEDGER}))
    return result


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.route = PAIR.ROUTES[1]
        self.request = PAIR.request_for(self.route, Path("/existing") / PAIR.ADAPTER_NAME,
                                        Path("/owned/new-pair"))
        self.result = result_for(self.route, self.request)

    def test_fixed_recipe_and_actual_native_receipt(self):
        self.assertEqual([r["name"] for r in PAIR.ROUTES], ["mlx", "native"])
        for route in PAIR.ROUTES:
            request = PAIR.request_for(route, Path("/existing") / PAIR.ADAPTER_NAME, Path("/owned/new"))
            self.assertEqual((request["steps"], request["seed"], request["width"], request["height"]),
                             (6, 42, 512, 512))
            self.assertEqual(request["loras"][0]["strength"], 1.0)
            self.assertEqual(request["inputs"], [])
            self.assertEqual(request["qwen21_dit_cache"], "off")
            self.assertTrue(request["allow_approximation"])
            self.assertFalse(request["compile_gpu"])
            self.assertFalse(request["qwen21_w8a8"])
            self.assertNotIn("qwen21_native_scale_cache", request)
            self.assertNotIn("qwen21_native_lora_kernel", request)
            PAIR.verify_result(result_for(route, request), route, request)

    def test_sgmma_selection_is_native_only_and_requires_actual_v2_recipe(self):
        recipe = "runtime-lora-fp32-sgmma8x8-aligned-r128-scalar-tail-ordered-bf16-bias-last-v2"
        for route in PAIR.ROUTES:
            request = PAIR.request_for(route, Path("/existing") / PAIR.ADAPTER_NAME,
                                       Path("/owned/new"), "sgmma")
            if route["name"] == "mlx":
                self.assertNotIn("qwen21_native_lora_kernel", request)
                self.assertEqual(request, PAIR.request_for(route, Path("/existing") / PAIR.ADAPTER_NAME,
                                                           Path("/owned/new")))
                continue
            self.assertEqual(request["qwen21_native_lora_kernel"], "sgmma")
            result = result_for(route, request)
            self.assertEqual(result["native_denoiser_lora"]["precision_recipe"], recipe)
            PAIR.verify_result(result, route, request)
            # A planned SGMMA selection never compensates for scalar, missing,
            # or incompletely specified actual arithmetic provenance.
            result["plan"]["qwen21_native_lora_kernel"] = "sgmma"
            for actual in (PAIR.LORA_RECIPE, None, recipe.replace("scalar-tail-", "")):
                result["native_denoiser_lora"]["precision_recipe"] = actual
                with self.subTest(actual=actual), self.assertRaises(ValueError):
                    PAIR.verify_result(result, route, request)
            del result["native_denoiser_lora"]
            with self.assertRaises(ValueError):
                PAIR.verify_result(result, route, request)
        self.result["native_denoiser_lora"]["precision_recipe"] = recipe
        with self.assertRaises(ValueError):
            PAIR.verify_result(self.result, self.route, self.request)
        with self.assertRaises(ValueError):
            PAIR.request_for(self.route, Path("/adapter"), Path("/new"), "automatic")

    def test_source_pin_identity_and_real_counts_cannot_be_requested_echo(self):
        for key, value in (("schema_version", True), ("enabled", 1), ("completed", False),
                           ("variant", "base20"), ("source_kind", "pathname"),
                           ("source_generation", 0), ("source_generation", True),
                           ("verified_artifact_digest", "stat-only"), ("adapter_sha256", "0" * 64),
                           ("adapter_revision", ""), ("adapter_count", 2), ("resolved_bindings", 194),
                           ("logical_projections", 226), ("ordered_terms", 226), ("max_rank", 16),
                           ("source_strength", True), ("completed_steps", 5),
                           ("scratch_bytes", 0), ("precision_recipe", "fp16-low")):
            result = copy.deepcopy(self.result)
            result["native_denoiser_lora"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                PAIR.verify_result(result, self.route, self.request)
        del self.result["native_denoiser_lora"]
        with self.assertRaises(ValueError):
            PAIR.verify_result(self.result, self.route, self.request)

    def test_complete_shape_gpu_route_and_cache_policy(self):
        for key, value in (("actual_denoise_steps", 5), ("actual_denoise_steps", 6.0),
                           ("width", 1024), ("height", 256), ("reference_tokens", 1),
                           ("warmup", True), ("prompt_cache_hit", True), ("seed", 43),
                           ("runtime_backend", "mlx_cpp_metal"),
                           ("gpu_graph", "native_full_dit_euler"), ("lora_strategy", "merge")):
            result = copy.deepcopy(self.result)
            result[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                PAIR.verify_result(result, self.route, self.request)
        for key, value in (("qwen21_dit_cache", "approximate"), ("execution", "gpu_ane"),
                           ("algorithm_approximations", ["skip-step"])):
            result = copy.deepcopy(self.result)
            result["plan"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                PAIR.verify_result(result, self.route, self.request)

    def test_native_requires_explicit_clean_phase(self):
        for key in PAIR.ZERO_LEDGER + ("worker_quarantined", "tainted", "execution_state"):
            result = copy.deepcopy(self.result)
            result["native_denoiser_memory_phase"]["memory_admission"].pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                PAIR.verify_result(result, self.route, self.request)
        self.result["native_denoiser_memory_phase"]["memory_admission"]["ledger_unknown_bytes"] = False
        with self.assertRaises(ValueError):
            PAIR.verify_result(self.result, self.route, self.request)

    def test_mlx_pin_path_does_not_manufacture_native_provenance(self):
        route = PAIR.ROUTES[0]
        request = PAIR.request_for(route, Path("/adapter"), Path("/new"))
        result = result_for(route, request)
        PAIR.verify_result(result, route, request)
        self.assertNotIn("native_denoiser_lora", result)
        result["lora_applied_projections"] = 0
        with self.assertRaises(ValueError):
            PAIR.verify_result(result, route, request)

    def test_nonfinite_or_missing_timings_fail(self):
        for value in (None, True, -1, 0, float("nan"), float("inf")):
            result = copy.deepcopy(self.result)
            result["timings_seconds"]["request_wall"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                PAIR.verify_result(result, self.route, self.request)

    def test_malformed_nested_receipt_is_a_validation_error(self):
        for key in ("plan", "timings_seconds", "native_denoiser_lora", "native_denoiser_memory_phase"):
            for value in (None, [], "invalid"):
                result = copy.deepcopy(self.result)
                result[key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    PAIR.verify_result(result, self.route, self.request)

    def test_timeout_zero_exit_race_remains_failure(self):
        process = Mock(pid=987654)
        process.wait.side_effect = [subprocess.TimeoutExpired("owned", 30), 0]
        with patch.object(PAIR.os, "killpg", side_effect=ProcessLookupError) as kill:
            status = PAIR.wait_bounded(process, 30)
        self.assertEqual(status["process_returncode"], 0)
        self.assertTrue(status["timeout"])
        self.assertTrue(status["termination_observed"])
        kill.assert_called_once_with(987654, signal.SIGTERM)

    def test_unknown_termination_and_invalid_budget(self):
        process = Mock(pid=987655)
        process.wait.side_effect = subprocess.TimeoutExpired("owned", 30)
        with patch.object(PAIR.os, "killpg") as kill:
            status = PAIR.wait_bounded(process, 30)
        self.assertFalse(status["termination_observed"])
        self.assertIsNone(status["process_returncode"])
        self.assertEqual([c.args for c in kill.call_args_list],
                         [(987655, signal.SIGTERM), (987655, signal.SIGKILL)])
        for timeout in (0, 29, 121, True):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                PAIR.wait_bounded(process, timeout)


class SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tc-lora-supervisor-cpu-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        native, model = self.root / "native", self.root / "model"
        (native / "runtime-build").mkdir(parents=True)
        (native / "runtime-build/runtime-build-manifest.json").write_text(json.dumps({"runtime_build_id": BUILD}))
        (native / "turbocider").write_text("CPU fixture; never execute\n")
        (native / "turbocider").chmod(0o700)
        (native / "libturbocider.dylib").write_bytes(b"CPU fake library")
        for relative in PAIR.MODEL_FILES:
            path = model / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"CPU metadata fixture only")
        adapter = self.root / PAIR.ADAPTER_NAME
        adapter.write_bytes(b"CPU adapter metadata fixture only")
        self.args = argparse.Namespace(native_dir=native, model_root=model, adapter=adapter,
                                       output_dir=self.root / "fresh", expected_runtime_build_id=BUILD,
                                       timeout_seconds=30, confirm_one_pair=True, native_lora_kernel="scalar")

    def fake_launch(self, command, *, stdout, stderr, **kwargs):
        self.assertTrue(kwargs["start_new_session"])
        request = json.loads(Path(command[-1]).read_text())
        route = next(route for route in PAIR.ROUTES if route["backend"] == request["qwen21_denoiser_backend"])
        stdout.write(json.dumps(result_for(route, request)))
        stderr.write("123456 peak memory footprint\n23456 maximum resident set size\n0 swaps\n")
        Path(request["output"]).write_bytes(png_header())
        return Mock(pid=987654)

    def execute(self, launch=None, termination=None):
        status = termination or dict(process_returncode=0, timeout=False, cancelled=False,
                                     termination_observed=True, signals=[])
        with patch.object(PAIR.subprocess, "Popen", side_effect=launch or self.fake_launch) as popen, \
                patch.object(PAIR, "wait_bounded", return_value=status), \
                patch.object(PAIR, "system_snapshot", return_value={"CPU_mock": True}):
            code = PAIR.run_pair(self.args)
        return code, popen.call_count, json.loads((self.args.output_dir / "receipt.json").read_text())

    def test_one_pair_writes_receipt_without_hashing_model_payloads(self):
        real_digest = PAIR.digest
        def only_artifacts(path):
            self.assertNotIn(path, [self.args.adapter, *(self.args.model_root / x for x in PAIR.MODEL_FILES)])
            return real_digest(path)
        with patch.object(PAIR, "digest", side_effect=only_artifacts):
            code, count, receipt = self.execute()
        self.assertEqual((code, count), (0, 2))
        self.assertTrue(receipt["pair_complete"])
        self.assertEqual(receipt["order"], ["mlx", "native"])
        self.assertEqual(len(receipt["source_before"]), 5)
        self.assertFalse(receipt["performance_qualified"])
        self.assertFalse(receipt["image_quality_verified"])
        self.assertFalse(receipt["runs"][0]["device_physical_completion_verified"])
        self.assertTrue(receipt["runs"][1]["native_phase_zero_ledger_verified"])
        self.assertEqual(receipt["runs"][1]["process_memory"]["peak_footprint_bytes"], 123456)
        self.assertEqual(receipt["native_lora_kernel_requested"], "scalar")
        self.assertEqual(receipt["native_lora_actual_precision_recipe"], PAIR.LORA_RECIPE)

    def test_sgmma_pair_receipt_preserves_actual_evidence(self):
        self.args.native_lora_kernel = "sgmma"
        code, count, receipt = self.execute()
        self.assertEqual((code, count), (0, 2))
        self.assertEqual(receipt["native_lora_kernel_requested"], "sgmma")
        self.assertEqual(receipt["native_lora_recipe_expected"], PAIR.LORA_SGMMA_RECIPE)
        self.assertEqual(receipt["native_lora_actual_precision_recipe"], PAIR.LORA_SGMMA_RECIPE)
        self.assertNotIn("qwen21_native_lora_kernel", receipt["runs"][0]["request"])
        native = receipt["runs"][1]
        self.assertEqual(native["request"]["qwen21_native_lora_kernel"], "sgmma")
        self.assertEqual(native["native_lora_actual_precision_recipe"],
                         native["result"]["native_denoiser_lora"]["precision_recipe"])

    def test_sgmma_mismatched_actual_recipe_stops_without_retry(self):
        self.args.native_lora_kernel = "sgmma"
        def scalar_actual(command, *, stdout, stderr, **kwargs):
            request = json.loads(Path(command[-1]).read_text())
            if request["qwen21_denoiser_backend"] != "native-metal-graph":
                return self.fake_launch(command, stdout=stdout, stderr=stderr, **kwargs)
            result = result_for(PAIR.ROUTES[1], request)
            result["plan"]["qwen21_native_lora_kernel"] = "sgmma"
            result["native_denoiser_lora"]["precision_recipe"] = PAIR.LORA_RECIPE
            stdout.write(json.dumps(result))
            return Mock(pid=987654)
        code, count, receipt = self.execute(launch=scalar_actual)
        self.assertEqual((code, count), (1, 2))
        self.assertTrue(receipt["pair_stopped"])
        self.assertFalse(receipt["pair_complete"])
        self.assertNotIn("native_lora_actual_precision_recipe", receipt)
        self.assertFalse(receipt["runs"][1]["device_physical_completion_verified"])

    def test_kernel_cli_default_and_explicit_selection(self):
        common = ["--native-dir", str(self.args.native_dir), "--model-root", str(self.args.model_root),
                  "--adapter", str(self.args.adapter), "--output-dir", str(self.args.output_dir),
                  "--expected-runtime-build-id", BUILD, "--confirm-one-pair"]
        for extra, expected in (([], "scalar"), (["--native-lora-kernel", "sgmma"], "sgmma")):
            with patch.object(PAIR, "run_pair", return_value=0) as run:
                self.assertEqual(PAIR.main(common + extra), 0)
            self.assertEqual(run.call_args.args[0].native_lora_kernel, expected)
        self.args.native_lora_kernel = "automatic"
        with patch.object(PAIR.subprocess, "Popen") as popen, self.assertRaises(ValueError):
            PAIR.run_pair(self.args)
        popen.assert_not_called()
        self.assertFalse(self.args.output_dir.exists())

    def test_timeout_stops_pair_without_device_completion_claim(self):
        status = dict(process_returncode=0, timeout=True, cancelled=False,
                      termination_observed=True, signals=["SIGTERM:already-exited"])
        code, count, receipt = self.execute(termination=status)
        self.assertEqual((code, count), (1, 1))
        self.assertTrue(receipt["pair_stopped"])
        self.assertFalse(receipt["pair_complete"])
        self.assertFalse(receipt["runs"][0]["device_physical_completion_verified"])

    def test_source_replacement_stops_before_second_route(self):
        def changed_source(*args, **kwargs):
            process = self.fake_launch(*args, **kwargs)
            self.args.adapter.write_bytes(b"changed CPU fixture")
            return process
        code, count, receipt = self.execute(launch=changed_source)
        self.assertEqual((code, count), (1, 1))
        self.assertFalse(receipt["sources_unchanged"])

    def test_no_overwrite_no_implicit_budget_no_wrong_adapter(self):
        self.args.output_dir.mkdir()
        marker = self.args.output_dir / "keep.txt"
        marker.write_text("prior evidence")
        with patch.object(PAIR.subprocess, "Popen") as popen, self.assertRaises(ValueError):
            PAIR.run_pair(self.args)
        self.assertEqual(marker.read_text(), "prior evidence")
        popen.assert_not_called()
        self.args.output_dir = self.root / "unused"
        self.args.confirm_one_pair = False
        with self.assertRaises(ValueError):
            PAIR.run_pair(self.args)
        self.args.confirm_one_pair = True
        self.args.adapter = self.root / "wrong-adapter.safetensors"
        with self.assertRaises(ValueError):
            PAIR.run_pair(self.args)

    def test_bad_output_shape_or_crc_and_nonfinite_json_are_rejected(self):
        path = self.root / "image.png"
        path.write_bytes(png_header())
        self.assertEqual(PAIR.image_evidence(path)["width"], 512)
        for payload in (png_header(256, 512), png_header()[:29] + b"0000" + png_header()[33:]):
            path.write_bytes(payload)
            with self.assertRaises(ValueError):
                PAIR.image_evidence(path)
        for text in ('{"x":NaN}', '{"x":1,"x":2}', '[]'):
            path.write_text(text)
            with self.assertRaises(ValueError):
                PAIR.load_json(path)


if __name__ == "__main__":
    unittest.main()
