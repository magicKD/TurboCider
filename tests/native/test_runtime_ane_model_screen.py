"""CPU-only checks of benchmark reporting, without MLX/Core ML or checkpoints."""
import copy
import hashlib
import json
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "runtime_ane_model_screen", ROOT / "tools/validation/runtime_ane_model_screen.py")
SCREEN = importlib.util.module_from_spec(SPEC)
with mock.patch.object(sys, "path", [str(ROOT / "tools/validation"), *sys.path]):
    SPEC.loader.exec_module(SCREEN)
    import runtime_ane_common as COMMON


class ScreenTests(unittest.TestCase):
    def test_fixed_async_requires_actual_successful_untimed_heads(self):
        row=copy.deepcopy(self.row);runtime=row["hybrid"]["runtime_weight"]
        runtime.update(hybrid_blocks_session_total=8,untimed_hybrid_blocks_session_total=8,async_hybrid_blocks_session_total=8)
        SCREEN.validate_fixed_async([row],True)
        with self.assertRaises(ValueError):SCREEN.validate_fixed_async([row],False)
        for change in ({"hybrid_blocks_session_total":0},{"untimed_hybrid_blocks_session_total":7},
                       {"async_hybrid_blocks_session_total":7},{"async_hybrid_blocks_session_total":True}):
            bad=copy.deepcopy(row);bad["hybrid"]["runtime_weight"].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):SCREEN.validate_fixed_async([bad],True)
        runtime.update(untimed_hybrid_blocks_session_total=0,async_hybrid_blocks_session_total=0)
        SCREEN.validate_fixed_async([row],False)
        with self.assertRaises(ValueError):SCREEN.validate_fixed_async([],True)
    def test_lora_channel_correction_counters_are_complete_and_monotonic(self):
        row=copy.deepcopy(self.row)
        row["hybrid"]["runtime_weight"].update(lora_channel_range_calls_session_total=2,lora_channel_full_calls_session_total=1)
        SCREEN.validate_results([row],"runtime",1)
        for key in ("lora_channel_range_calls_session_total","lora_channel_full_calls_session_total"):
            for value in (-1,True,.5):
                bad=copy.deepcopy(row);bad["hybrid"]["runtime_weight"][key]=value
                with self.subTest(key=key,value=value),self.assertRaises(ValueError):SCREEN.validate_results([bad],"runtime",1)
            bad=copy.deepcopy(row);del bad["hybrid"]["runtime_weight"][key]
            with self.assertRaises(ValueError):SCREEN.validate_results([bad],"runtime",1)
            later=copy.deepcopy(row);later["hybrid"]["runtime_weight"][key]=0
            with self.assertRaises(ValueError):SCREEN.validate_results([row,later],"runtime",2)
        later=copy.deepcopy(self.row)
        with self.assertRaises(ValueError):SCREEN.validate_results([row,later],"runtime",2)

    def test_explicit_lora_channel_ablation_requires_actual_executed_receipt(self):
        row=copy.deepcopy(self.row);row["lora_strategy"]="inference_time"
        runtime=row["hybrid"]["runtime_weight"]
        runtime.update(executor_backend="private_ane",partition_axis="intermediate_channels",
                       lora_channel_range_calls_session_total=2,lora_channel_full_calls_session_total=0)
        SCREEN.validate_lora_channel_range([row],True)
        with self.assertRaises(ValueError):SCREEN.validate_lora_channel_range([row],False)
        for change in ({"lora_channel_range_calls_session_total":0},{"lora_channel_full_calls_session_total":1},
                       {"executor_backend":"public_coreml"},{"partition_axis":"rows"}):
            bad=copy.deepcopy(row);bad["hybrid"]["runtime_weight"].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):SCREEN.validate_lora_channel_range([bad],True)
        runtime.update(lora_channel_range_calls_session_total=0,lora_channel_full_calls_session_total=2)
        SCREEN.validate_lora_channel_range([row],False)
        with self.assertRaises(ValueError):SCREEN.validate_lora_channel_range([],False)

    def test_specialized_stage_receipts_are_complete_and_bounded(self):
        row=copy.deepcopy(self.row)
        row["hybrid"]["runtime_weight"].update(stage_specialized=True,stage_pipeline_variants=18)
        SCREEN.validate_results([row],"runtime",1)
        for change in ({"stage_specialized":False},{"stage_specialized":1},{"stage_pipeline_variants":19},
                       {"stage_pipeline_variants":True}):
            bad=copy.deepcopy(row);bad["hybrid"]["runtime_weight"].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):SCREEN.validate_results([bad],"runtime",1)
        for name in ("stage_specialized","stage_pipeline_variants"):
            bad=copy.deepcopy(row);del bad["hybrid"]["runtime_weight"][name]
            with self.assertRaises(ValueError):SCREEN.validate_results([bad],"runtime",1)
        later=copy.deepcopy(row);later["hybrid"]["runtime_weight"]["stage_pipeline_variants"]=17
        with self.assertRaises(ValueError):SCREEN.validate_results([row,later],"runtime",1)
        later=copy.deepcopy(row)
        for name in ("stage_specialized","stage_pipeline_variants"):del later["hybrid"]["runtime_weight"][name]
        with self.assertRaises(ValueError):SCREEN.validate_results([row,later],"runtime",1)
    def test_a8_lookahead_receipts_are_complete_and_bounded(self):
        row = copy.deepcopy(self.row)
        runtime = row["hybrid"]["runtime_weight"]
        good = {"a8_lookahead_enabled": True, "a8_prefetches_session_total": 1,
                "a8_wait_seconds_session_total": .001}
        runtime.update(good)
        SCREEN.validate_results([row], "runtime", 1)
        for change in ({"a8_lookahead_enabled": False}, {"a8_lookahead_enabled": 1},
                       {"a8_prefetches_session_total": True}, {"a8_prefetches_session_total": 999},
                       {"a8_wait_seconds_session_total": float("nan")}, {"a8_wait_seconds_session_total": True}):
            bad = copy.deepcopy(row)
            bad["hybrid"]["runtime_weight"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                SCREEN.validate_results([bad], "runtime", 1)
        for name in good:
            bad = copy.deepcopy(row)
            del bad["hybrid"]["runtime_weight"][name]
            with self.assertRaises(ValueError):
                SCREEN.validate_results([bad], "runtime", 1)
        for change in ({"a8_prefetches_session_total": 0}, {"a8_wait_seconds_session_total": 0},
                       {"a8_lookahead_enabled": False, "a8_prefetches_session_total": 0}):
            later = copy.deepcopy(row)
            later["hybrid"]["runtime_weight"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                SCREEN.validate_results([row, later], "runtime", 1)
        runtime.update(a8_lookahead_enabled=False, a8_prefetches_session_total=0)
        SCREEN.validate_results([row], "runtime", 1)
    def test_scale_cache_receipts_are_complete_bounded_metadata(self):
        row=copy.deepcopy(self.row);runtime=row["hybrid"]["runtime_weight"]
        good={"scale_cache_enabled":True,"scale_cache_hits_session_total":4,"scale_cache_misses_session_total":3,
              "scale_cache_entries":3,"scale_cache_bytes":1536,"scale_cache_evictions_session_total":0}
        runtime.update(good);SCREEN.validate_results([row],"runtime",1)
        for change in ({"scale_cache_enabled":False},{"scale_cache_enabled":1},{"scale_cache_entries":129},
                       {"scale_cache_bytes":(4<<20)+1},{"scale_cache_hits_session_total":True},{"scale_cache_entries":0}):
            bad=copy.deepcopy(row);bad["hybrid"]["runtime_weight"].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):SCREEN.validate_results([bad],"runtime",1)
        for name in good:
            if name=="scale_cache_enabled":continue
            bad=copy.deepcopy(row);del bad["hybrid"]["runtime_weight"][name]
            with self.assertRaises(ValueError):SCREEN.validate_results([bad],"runtime",1)
    def test_future_bank_receipts_require_complete_bounded_counters(self):
        row=copy.deepcopy(self.row)
        receipt=row["hybrid"]["runtime_weight"]
        good={"prefetch_enabled":True,"prefetch_submissions_session_total":5,"prefetch_hits_session_total":3,
              "prefetch_discards_session_total":1,"prefetch_failures_session_total":0,"prefetch_wait_seconds_session_total":.01}
        receipt.update(good);SCREEN.validate_results([row],"runtime",1)
        for change in ({"prefetch_hits_session_total":5},{"prefetch_enabled":False},{"prefetch_enabled":1},
                       {"prefetch_discards_session_total":True},{"prefetch_wait_seconds_session_total":float("nan")}):
            bad=copy.deepcopy(row);bad["hybrid"]["runtime_weight"].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):SCREEN.validate_results([bad],"runtime",1)
        for name in good:
            if name=="prefetch_enabled":continue
            bad=copy.deepcopy(row);del bad["hybrid"]["runtime_weight"][name]
            with self.assertRaises(ValueError):SCREEN.validate_results([bad],"runtime",1)
    def test_w8a8_screen_rejects_fp16_or_missing_data_path(self):
        row = copy.deepcopy(self.row)
        row["runtime_backend"] = "mlx_cpp_metal+private_ane_runtime_weight_experimental"
        runtime = row["hybrid"]["runtime_weight"]
        runtime.update(executor_backend="private_ane", io_path="gpu_iosurface", data_path="w8a8_hadamard",
                       device_io_calls_session_total=row["hybrid"]["runtime_calls_session_total"])
        SCREEN.validate_results([row], "runtime", 1, runtime_backend="private", expect_device_io=True,
                                expected_data_path="w8a8_hadamard")
        for path in (None, "fp16", "w8a8"):
            bad = copy.deepcopy(row); bad["hybrid"]["runtime_weight"]["data_path"] = path
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "data path"):
                SCREEN.validate_results([bad], "runtime", 1, runtime_backend="private", expect_device_io=True,
                                        expected_data_path="w8a8_hadamard")

    def test_private_gpu_io_requires_matching_receipt_for_all_predictions(self):
        row = copy.deepcopy(self.row)
        row["runtime_backend"] = "mlx_cpp_metal+private_ane_runtime_weight_experimental"
        runtime = row["hybrid"]["runtime_weight"]
        runtime.update(executor_backend="private_ane", io_path="gpu_iosurface",
                       device_io_calls_session_total=row["hybrid"]["runtime_calls_session_total"])
        SCREEN.validate_results([row], "runtime", 1, runtime_backend="private", expect_device_io=True)
        for change in ({"io_path": "host"}, {"device_io_calls_session_total": 0},
                       {"device_io_calls_session_total": True}, {"executor_backend": "public_coreml"}):
            bad = copy.deepcopy(row); bad["hybrid"]["runtime_weight"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                SCREEN.validate_results([bad], "runtime", 1, runtime_backend="private", expect_device_io=True)
    def test_private_backend_is_explicit_and_cannot_masquerade_as_public(self):
        row = copy.deepcopy(self.row)
        row["runtime_backend"] = "mlx_cpp_metal+private_ane_runtime_weight_experimental"
        runtime = row["hybrid"]["runtime_weight"]
        runtime["executor_backend"] = "private_ane"
        for policy in ("private", "auto"):
            SCREEN.validate_results([row], "runtime", 1, runtime_backend=policy)
        with self.assertRaisesRegex(ValueError, "backend"):
            SCREEN.validate_results([row], "runtime", 1)
        runtime["executor_backend"] = "public_coreml"
        with self.assertRaisesRegex(ValueError, "telemetry"):
            SCREEN.validate_results([row], "runtime", 1, runtime_backend="private")
        row["runtime_backend"] = "mlx_cpp_metal+coreml_runtime_weight"
        SCREEN.validate_results([row], "runtime", 1, runtime_backend="auto")
        with self.assertRaisesRegex(ValueError, "backend"):
            SCREEN.validate_results([row], "runtime", 1, runtime_backend="private")
        private = copy.deepcopy(row)
        private["runtime_backend"] = "mlx_cpp_metal+private_ane_runtime_weight_experimental"
        private["hybrid"]["runtime_weight"]["executor_backend"] = "private_ane"
        with self.assertRaisesRegex(ValueError, "changed"):
            SCREEN.validate_results([row, private], "runtime", 2, runtime_backend="auto")
    def test_qkv_receipts_do_not_accept_gpu_fallback_or_ffn_counts(self):
        row = {"runtime_backend": "mlx_cpp_metal+coreml_runtime_qkv",
               "timings_seconds": {"request_wall": 2., "denoise": 1.},
               "hybrid": {},
               "qkv": {"mode": "runtime_weight_qkv", "failed": False,
                       "failure_reason": "", "failures_session_total": 0,
                       "fallback_blocks_session_total": 0, "calls_session_total": 32,
                       "hybrid_blocks_session_total": 32,
                       "observed_ane_residency": "unknown"}}
        SCREEN.validate_results([row], "qkv", 1, "qwen-image-2.1")
        for change in ({"failed": True}, {"fallback_blocks_session_total": 1},
                       {"calls_session_total": 0}, {"mode": "runtime_weight_ffn"}):
            bad = copy.deepcopy(row)
            bad["qkv"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                SCREEN.validate_results([bad], "qkv", 1, "qwen-image-2.1")
        with self.assertRaises(ValueError):
            SCREEN.validate_results([row, row], "qkv", 2, "qwen-image-2.1")

    def test_qkv_auto_may_decline_every_block_of_a_later_request(self):
        receipt = {"mode": "runtime_weight_qkv", "auto_scheduling": True,
                   "failed": False, "failure_reason": "", "failures_session_total": 0,
                   "fallback_blocks_session_total": 0, "calls_session_total": 32,
                   "hybrid_blocks_session_total": 32, "gpu_blocks_session_total": 64,
                   "gpu_probe_blocks_session_total": 32,
                   "measured_hybrid_blocks_session_total": 32,
                   "gpu_probe_seconds_session_total": 1.,
                   "measured_hybrid_seconds_session_total": 1.,
                   "observed_ane_residency": "unknown"}
        first = {"runtime_backend": "mlx_cpp_metal+coreml_runtime_qkv",
                 "timings_seconds": {"request_wall": 2., "denoise": 1.},
                 "hybrid": {}, "qkv": receipt}
        second = copy.deepcopy(first)
        second["qkv"]["gpu_blocks_session_total"] = 224
        SCREEN.validate_results([first, second], "qkv", 2, "qwen-image-2.1")
        for change in ({"auto_scheduling": False}, {"gpu_blocks_session_total": 64},
                       {"failed": True}, {"gpu_probe_blocks_session_total": 225}):
            bad = copy.deepcopy(second)
            bad["qkv"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                SCREEN.validate_results([first, bad], "qkv", 2, "qwen-image-2.1")
        bad = copy.deepcopy(second)
        bad["qkv"].update(calls_session_total=33, hybrid_blocks_session_total=33,
                          gpu_blocks_session_total=63)
        with self.assertRaises(ValueError):
            SCREEN.validate_results([first, bad], "qkv", 2, "qwen-image-2.1")

    def test_shared_contracts_have_one_implementation(self):
        with mock.patch.object(sys, "path", [str(ROOT / "tools/validation"), *sys.path]):
            import runtime_lora_shared_graph_switch as switch
        for name in ("benchmark_environment", "validate_results", "validate_edit_results", "wait_for_idle",
                     "qwen_qk_environment", "validate_qwen_qk_receipts"):
            self.assertIs(getattr(SCREEN, name), getattr(COMMON, name))
            self.assertIs(getattr(switch, name), getattr(COMMON, name))

    def test_common_import_needs_no_optional_sdk_or_runner_side_effects(self):
        script = """
import os, subprocess, sys, time
sys.path.insert(0, sys.argv[1])
def forbidden(*args, **kwargs):
    raise AssertionError("import must not start a subprocess or wait")
subprocess.run = subprocess.check_output = subprocess.Popen = time.sleep = forbidden
before = dict(os.environ)
import runtime_ane_common
assert dict(os.environ) == before
assert not any(name in sys.modules for name in (
    "mlx", "coremltools", "runtime_ane_model_screen", "runtime_lora_shared_graph_switch"))
"""
        result = subprocess.run(
            [sys.executable, "-S", "-c", script, str(ROOT / "tools/validation")],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_evidence_hashing_has_one_streaming_implementation(self):
        with mock.patch.object(sys, "path", [str(ROOT / "tools/validation"), *sys.path]):
            import runtime_ane_memory as memory
            import runtime_lora_shared_graph_switch as switch
        self.assertIs(SCREEN.sha256_file, COMMON.sha256_file)
        self.assertIs(memory.sha256, COMMON.sha256_file)
        self.assertIs(switch.sha256, COMMON.sha256_file)

    def test_file_hash_matches_binary_content_without_whole_file_read(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "evidence.bin"
            for data in (b"", b"abc", bytes(range(256)) * 8193):
                path.write_bytes(data)
                with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("whole-file read")):
                    for name in (path, str(path)):
                        self.assertEqual(COMMON.sha256_file(name), hashlib.sha256(data).hexdigest())

    def test_missing_evidence_hash_fails_without_creating_a_file(self):
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch) / "missing.bin"
            with self.assertRaises(FileNotFoundError):
                COMMON.sha256_file(path)
            self.assertFalse(path.exists())

    def test_chunk_policy_matches_native_options(self):
        for value in ("auto", "0", "1", "128"):
            self.assertEqual(SCREEN.chunk_policy(value), value)
        self.assertEqual(SCREEN.chunk_policy("001"), "1")
        for value in ("", "-1", "+1", "129", "1000", "1.0", "Auto", " 1", "１"):
            with self.subTest(value=value), self.assertRaises(SCREEN.argparse.ArgumentTypeError):
                SCREEN.chunk_policy(value)

    def test_invalid_run_options_fail_before_output_creation(self):
        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch)/"screen"
            for option in (("--chunks", "129"), ("--timeout", "0"),
                           ("--private-data-path", "w8a8"),
                           ("--private-lora-channel-range", "0"), ("--private-lora-channel-range", "1"),
                           ("--fixed-async", "1"),
                           ("--private-channels", "512"), ("--private-channels", "-1"),
                           ("--sample-memory", "--memory-interval-ms", "0"),
                           ("--sample-memory", "--memory-max-gap-ms", "99"),
                           ("--memory-interval-ms", "200"), ("--qwen-qk-norm-rope",)):
                result = subprocess.run(
                    [sys.executable, str(ROOT/"tools/validation/runtime_ane_model_screen.py"),
                     "--model", "unused", "--model-id", "z-image-turbo", "--routes", "gpu",
                     "--steps", "8", "--output", str(output), *option],
                    capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertFalse(output.exists())

    def test_qwen_qk_environment_is_explicit_and_shape_limited(self):
        self.assertEqual(COMMON.qwen_qk_environment("z-image-turbo", 1024, False), {})
        self.assertEqual(COMMON.qwen_qk_environment("qwen-image-2.1", 512, True),
                         {"TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE": "1"})
        self.assertEqual(COMMON.qwen_qk_environment("qwen-image-2.1", 1024, True, base_generation=True),
                         {"TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE": "1"})
        for model, size in (("z-image-turbo", 512), ("qwen-image-2.1", 1024)):
            with self.assertRaisesRegex(ValueError, "Qwen 512px"):
                COMMON.qwen_qk_environment(model, size, True)
        for model, size, base in (("z-image-turbo", 1024, True), ("qwen-image-2.1", 768, True),
                                  ("qwen-image-2.1", 1024, "true")):
            with self.assertRaisesRegex(ValueError, "Qwen 512px"):
                COMMON.qwen_qk_environment(model, size, True, base_generation=base)

    def test_qwen_qk_1024_lora_and_edit_fail_before_reading_artifacts(self):
        for extra in (("--lora", "missing.safetensors"), ("--reference", "missing.png")):
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "unused"
                result = subprocess.run(
                    [sys.executable, "-B", str(ROOT / "tools/validation/runtime_ane_model_screen.py"),
                     "--model", "unused", "--model-id", "qwen-image-2.1", "--size", "1024",
                     "--steps", "40", "--routes", "gpu", "--output", str(output),
                     "--qwen-qk-norm-rope", *extra], capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("1024px base generation", result.stderr)
                self.assertFalse(output.exists())

    def test_qwen_qk_receipts_require_plan_and_actual_agreement(self):
        row = {"acceleration_selection": "experimental fused Metal Q/K norm-RoPE",
               "plan": {"algorithm_approximations": ["qwen21_metal_qk_norm_rope"]}}
        COMMON.validate_qwen_qk_receipts([row], True)
        COMMON.validate_qwen_qk_receipts([{}], False)
        for bad in ({}, {**row, "plan": {}}, {**row, "plan": None},
                    {**row, "acceleration_selection": "gpu"},
                    {**row, "acceleration_selection": None},
                    {**row, "plan": {"algorithm_approximations": "qwen21_metal_qk_norm_rope"}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                COMMON.validate_qwen_qk_receipts([bad], True)
        with self.assertRaises(ValueError):
            COMMON.validate_qwen_qk_receipts([row], False)
        with self.assertRaises(ValueError):
            COMMON.validate_qwen_qk_receipts([], True)

    def test_qwen_qk_1024_base_uses_same_selection_on_every_route(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "build/native").mkdir(parents=True)
            (root / "build/native/libturbocider.dylib").write_bytes(b"fixture")
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            enabled = False
            seen = []
            def run(command, **kwargs):
                env = kwargs["env"]
                self.assertEqual(env.get("TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE"),
                                 "1" if enabled else None)
                for name in command[3:]:
                    request = json.loads(Path(name).read_text())
                    self.assertEqual((request["width"], request["height"]), (1024, 1024))
                    self.assertEqual(request["operation"], "image.generate")
                    self.assertNotIn("loras", request)
                    self.assertNotIn("inputs", request)
                    route = "gpu" if request["execution"] == "gpu" else (
                        "runtime" if request.get("hybrid_mlp_mode") == "runtime" else "frozen")
                    self.assertEqual(env.get("TURBOCIDER_QWEN21_1024_W8A8_DIAGNOSTIC"),
                                     "1" if route == "frozen" else None)
                    seen.append(route)
                    row = copy.deepcopy(self.row)
                    row["runtime_backend"] = "mlx_cpp_metal" + {
                        "gpu": "", "runtime": "+coreml_runtime_weight", "frozen": "+coreml"}[route]
                    row["acceleration_selection"] = "experimental fused Metal Q/K norm-RoPE" if enabled else "gpu"
                    row["plan"] = {"algorithm_approximations": ["qwen21_metal_qk_norm_rope"] if enabled else []}
                    kwargs["stdout"].write(json.dumps(row) + "\n")
                return subprocess.CompletedProcess(command, 0)
            args = ["screen", "--model", str(root), "--model-id", "qwen-image-2.1",
                    "--size", "1024", "--steps", "40", "--warm-repeats", "1",
                    "--routes", "gpu,runtime,frozen", "--runtime-manifest", str(manifest),
                    "--frozen-manifest", str(manifest)]
            with mock.patch.object(SCREEN, "ROOT", root), \
                    mock.patch.object(SCREEN, "wait_for_idle", return_value="idle"), \
                    mock.patch.object(SCREEN, "system_memory", return_value={}), \
                    mock.patch.object(SCREEN, "run_owned", side_effect=run), \
                    mock.patch.dict(COMMON.os.environ, {"TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE": "1"}), \
                    mock.patch("builtins.print"):
                for enabled in (False, True):
                    seen.clear()
                    output = root / f"fusion-{enabled}"
                    extra = ["--qwen-qk-norm-rope"] if enabled else []
                    with mock.patch.object(sys, "argv", [*args, "--output", str(output), *extra]):
                        SCREEN.main()
                    summary = json.loads((output / "summary.json").read_text())
                    self.assertEqual(summary["status"], "complete")
                    self.assertEqual(summary.get("qwen_qk_norm_rope", False), enabled)
                    self.assertEqual(seen, ["gpu", "gpu", "runtime", "runtime", "frozen", "frozen"])

    def test_qwen_lora_fp16_rejects_unsupported_workloads_before_output(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            adapter = root / "adapter.safetensors"
            adapter.write_bytes(b"fixture")
            for model, steps, size, lora in (
                    ("z-image-turbo", 8, 512, True),
                    ("qwen-image-2.1", 6, 512, False),
                    ("qwen-image-2.1", 40, 512, True),
                    ("qwen-image-2.1", 6, 1024, True)):
                output = root / "unused"
                args = [sys.executable, str(ROOT / "tools/validation/runtime_ane_model_screen.py"),
                        "--model", "unused", "--model-id", model, "--routes", "gpu",
                        "--steps", str(steps), "--size", str(size), "--output", str(output),
                        "--qwen-lora-fp16"]
                if lora:
                    args += ["--lora", str(adapter)]
                result = subprocess.run(args, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("requires Qwen LoRA", result.stderr)
                self.assertFalse(output.exists())

    def test_qwen_lora_precision_receipt_is_not_base_dtype(self):
        standard = {"runtime_precision": "bf16", "acceleration_selection": "runtime LoRA"}
        fp16 = {**standard, "acceleration_selection":
                "runtime LoRA; experimental FP16 low-rank LoRA matmuls"}
        SCREEN.validate_qwen_lora_precision([standard], False)
        SCREEN.validate_qwen_lora_precision([fp16], True)
        for row, enabled in ((standard, True), (fp16, False), ({}, False),
                             ({"acceleration_selection": None}, False),
                             ({"acceleration_selection": 1}, True)):
            with self.assertRaisesRegex(ValueError, "rank precision"):
                SCREEN.validate_qwen_lora_precision([row], enabled)
        with self.assertRaises(ValueError):
            SCREEN.validate_qwen_lora_precision([fp16, standard], True)

    def test_qwen_lora_fp16_matches_every_route_and_failed_receipts_stay_incomplete(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root / "build/native").mkdir(parents=True)
            (root / "build/native/libturbocider.dylib").write_bytes(b"library")
            adapter, manifest = root / "adapter.safetensors", root / "manifest.json"
            adapter.write_bytes(b"adapter")
            manifest.write_text("{}")
            enabled, mismatch, qk_enabled, qk_mismatch = False, False, False, False
            seen = []
            def run(command, **kwargs):
                env = kwargs["env"]
                self.assertEqual(env.get("TURBOCIDER_QWEN21_VIGGLE_LORA_FP16"), "1" if enabled else None)
                self.assertEqual(env.get("TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE"),
                                 "1" if qk_enabled else None)
                seen.append(env)
                for path in command[3:]:
                    request = json.loads(Path(path).read_text())
                    self.assertTrue(request["allow_approximation"])
                    self.assertEqual(request["lora_strategy"], "inference_time")
                    route = "gpu" if request["execution"] == "gpu" else (
                        "runtime" if request.get("hybrid_mlp_mode") == "runtime" else "frozen")
                    row = copy.deepcopy(self.row)
                    row.update(runtime_backend="mlx_cpp_metal" + {
                        "gpu": "", "runtime": "+coreml_runtime_weight", "frozen": "+coreml"}[route],
                        lora_strategy="inference_time", lora_applied_projections=227,
                        acceleration_selection="LoRA" + ("; experimental FP16 low-rank LoRA matmuls"
                            if enabled and not mismatch else ""))
                    row["hybrid"]["mlp_output_kind"] = (
                        "runtime_weight_swiglu_lora_inputs" if route == "runtime" else "fused_lora")
                    if qk_enabled:
                        row["plan"] = {"algorithm_approximations": ["qwen21_metal_qk_norm_rope"]}
                        if not qk_mismatch:
                            row["acceleration_selection"] += "; experimental fused Metal Q/K norm-RoPE"
                    kwargs["stdout"].write(json.dumps(row) + "\n")
                return subprocess.CompletedProcess(command, 0)
            args = ["screen", "--model", str(root), "--model-id", "qwen-image-2.1",
                    "--routes", "gpu,runtime,frozen", "--steps", "6", "--warm-repeats", "1",
                    "--runtime-manifest", str(manifest), "--frozen-manifest", str(manifest),
                    "--lora", str(adapter)]
            with mock.patch.object(SCREEN, "ROOT", root), \
                    mock.patch.object(SCREEN, "wait_for_idle", return_value="idle"), \
                    mock.patch.object(SCREEN, "system_memory", return_value={}), \
                    mock.patch.object(SCREEN, "run_owned", side_effect=run), \
                    mock.patch.dict(COMMON.os.environ, {"TURBOCIDER_QWEN21_VIGGLE_LORA_FP16": "1",
                                                      "TURBOCIDER_QWEN21_METAL_QK_NORM_ROPE": "1"}), \
                    mock.patch("builtins.print"):
                for enabled, qk_enabled in ((False, False), (True, False), (False, True), (True, True)):
                    seen.clear()
                    output = root / f"fp16-{enabled}-qk-{qk_enabled}"
                    options = ["--qwen-lora-fp16"] if enabled else []
                    if qk_enabled:
                        options += ["--qwen-qk-norm-rope"]
                    with mock.patch.object(sys, "argv", [*args, "--output", str(output), *options]):
                        SCREEN.main()
                    summary = json.loads((output / "summary.json").read_text())
                    self.assertEqual(summary["status"], "complete")
                    self.assertEqual(summary["lora"]["rank_matmul_dtype"], "fp16" if enabled else "fp32")
                    self.assertEqual(summary.get("qwen_qk_norm_rope", False), qk_enabled)
                    self.assertEqual(len(seen), 3)
                qk_mismatch = True
                output = root / "failed-qk"
                with mock.patch.object(sys, "argv", [*args, "--output", str(output),
                                                     "--qwen-lora-fp16", "--qwen-qk-norm-rope"]), \
                        self.assertRaisesRegex(ValueError, "Q/K norm-RoPE"):
                    SCREEN.main()
                self.assertEqual(json.loads((output / "summary.json").read_text())["status"], "incomplete")
                qk_enabled = False
                mismatch = True
                output = root / "failed"
                with mock.patch.object(sys, "argv", [*args, "--output", str(output), "--qwen-lora-fp16"]), \
                        self.assertRaisesRegex(ValueError, "rank precision"):
                    SCREEN.main()
                self.assertEqual(json.loads((output / "summary.json").read_text())["status"], "incomplete")
                self.assertTrue((output / "0-gpu.stdout.jsonl").is_file())

    def setUp(self):
        self.row = {
            "runtime_backend": "mlx_cpp_metal+coreml_runtime_weight",
            "timings_seconds": {"request_wall": 7., "denoise": 6.},
            "hybrid": {"runtime_failed": False, "runtime_failures_session_total": 0,
                       "runtime_calls_session_total": 2,
                       "runtime_weight": {"fallback_blocks_session_total": 0,
                                          "failure_reason": ""}},
        }

    def test_edit_workload_keeps_reference_order_and_native_limits(self):
        with tempfile.TemporaryDirectory() as scratch:
            paths = [Path(scratch) / f"{i}.png" for i in range(3)]
            for i, path in enumerate(paths):
                path.write_bytes(bytes([i]))  # metadata fixture, not image decode
            for count in (1, 2, 3):
                edit = SCREEN.edit_workload("qwen-image-2.1", paths[:count], 512, 512,
                                            ["gpu", "runtime", "frozen"], False)
                self.assertEqual(edit["operation"], "image.edit")
                self.assertEqual(edit["qwen21_reference_size"], 512)
                self.assertEqual([r["path"] for r in edit["inputs"]],
                                 [str(p.resolve()) for p in paths[:count]])
            self.assertEqual(SCREEN.edit_workload("z-image-turbo", [], 1024, 512, ["gpu"], False), {})
            for model, refs, ref_size, size, routes, lora in (
                ("z-image-turbo", paths, 512, 512, ["gpu"], False),
                ("qwen-image-2.1", paths + paths[:1], 512, 512, ["gpu"], False),
                ("qwen-image-2.1", [], 512, 512, ["gpu"], False),
                ("qwen-image-2.1", paths, 256, 512, ["runtime"], True),
                ("qwen-image-2.1", paths, 1024, 1024, ["runtime"], True),
                ("qwen-image-2.1", paths, 1024, 1024, ["frozen"], False),
                ("qwen-image-2.1", paths, 123, 512, ["gpu"], False),
                ("qwen-image-2.1", [Path(scratch)/"missing.png"], 512, 512, ["gpu"], False),
            ):
                with self.subTest(model=model, refs=len(refs), size=size, lora=lora), \
                        self.assertRaises(ValueError):
                    SCREEN.edit_workload(model, refs, ref_size, size, routes, lora)

    def test_reference_receipts_detect_mutation_and_reordering(self):
        with tempfile.TemporaryDirectory() as scratch:
            paths = [Path(scratch)/f"{i}.png" for i in range(2)]
            for i, path in enumerate(paths):
                path.write_bytes(bytes([i]))
            edit = SCREEN.edit_workload("qwen-image-2.1", paths, 512, 512, ["runtime"], False)
            receipts = SCREEN.reference_receipts(edit)
            SCREEN.check_references(edit, receipts)
            reversed_edit = {**edit, "inputs": list(reversed(edit["inputs"]))}
            with self.assertRaisesRegex(ValueError, "reference images changed"):
                SCREEN.check_references(reversed_edit, receipts)
            paths[0].write_bytes(b"x")  # same size is not the same input
            with self.assertRaisesRegex(ValueError, "reference images changed"):
                SCREEN.check_references(edit, receipts)
            self.assertEqual(SCREEN.reference_receipts({}), [])

    def test_edit_environment_only_enables_required_diagnostics(self):
        for route in ("gpu", "runtime", "frozen"):
            self.assertEqual(SCREEN.reference_environment(route, {}, True), {})
            self.assertEqual(SCREEN.reference_environment(route, {"qwen21_reference_size": 512}, True),
                             {"TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC": "1"})
            self.assertEqual(SCREEN.reference_environment(route, {"qwen21_reference_size": 512}, False), {})
            expected = {"TURBOCIDER_QWEN21_FULL_REF_W8A8_DIAGNOSTIC": "1"} if route == "frozen" else {}
            self.assertEqual(SCREEN.reference_environment(route, {"qwen21_reference_size": 1024}, False), expected)

    def test_edit_results_reject_dropped_references_or_wrong_operation(self):
        edit = {"operation": "image.edit", "qwen21_reference_size": 512}
        row = {"operation": "image.edit", "reference_tokens": 3072,
               "plan": {"operation": "image.edit", "qwen21_reference_size": 512}}
        self.assertEqual(SCREEN.validate_edit_results([row, row], edit), 3072)
        self.assertIsNone(SCREEN.validate_edit_results([row], {}))
        for invalid in ({**row, "operation": "image.generate"},
                        {**row, "plan": {}},
                        {**row, "plan": {**row["plan"], "qwen21_reference_size": 1024}},
                        {**row, "plan": {**row["plan"], "qwen21_reference_size": 512.0}},
                        *[{**row, "reference_tokens": v} for v in (0, -1, True, 1024.0, None)]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                SCREEN.validate_edit_results([invalid], edit)
        with self.assertRaisesRegex(ValueError, "inconsistent reference tokens"):
            SCREEN.validate_edit_results([row, {**row, "reference_tokens": 1024}], edit)
        with self.assertRaises(ValueError):
            SCREEN.validate_edit_results([], edit)

    def test_invalid_edit_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch)/"screen"
            image = Path(scratch)/"input.png"
            image.write_bytes(b"fixture")
            result = subprocess.run(
                [sys.executable, str(ROOT/"tools/validation/runtime_ane_model_screen.py"),
                 "--model", "unused", "--model-id", "z-image-turbo", "--routes", "gpu",
                 "--steps", "8", "--output", str(output), "--reference", str(image)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("Qwen21", result.stderr)
            self.assertFalse(output.exists())

    def test_explicit_cli_requires_executable_and_adjacent_library_before_output(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            for case in ("missing-cli", "directory-cli", "non-executable", "missing-library"):
                with self.subTest(case=case):
                    build = root / case
                    build.mkdir()
                    cli, output = build / "turbocider", build / "evidence"
                    if case == "directory-cli":
                        cli.mkdir()
                    elif case != "missing-cli":
                        cli.write_text("fixture")
                        cli.chmod(0o644 if case == "non-executable" else 0o755)
                    if case != "missing-library":
                        (build / "libturbocider.dylib").write_bytes(b"fixture library")
                    result = subprocess.run(
                        [sys.executable, str(ROOT / "tools/validation/runtime_ane_model_screen.py"),
                         "--cli", str(cli), "--model", "unused", "--model-id", "qwen-image-2.1",
                         "--routes", "gpu", "--steps", "6", "--output", str(output)],
                        capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("existing executable and adjacent libturbocider.dylib", result.stderr)
                    self.assertFalse(output.exists())

    def test_main_edit_requests_are_matched_and_reject_cross_route_geometry(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            (root/"build/native").mkdir(parents=True)
            (root/"build/native/libturbocider.dylib").write_bytes(b"fixture library")
            manifest = root/"manifest.json"
            manifest.write_text("{}")
            references = [root/f"ref-{i}.png" for i in range(3)]
            for i, path in enumerate(references):
                path.write_bytes(bytes([i]))
            seen = []
            mismatch = False
            returncode = 0

            def run(command, **kwargs):
                for path in command[3:]:
                    request = json.loads(Path(path).read_text())
                    seen.append(request)
                    route = "gpu" if request["execution"] == "gpu" else (
                        "runtime" if request.get("hybrid_mlp_mode") == "runtime" else "frozen")
                    row = copy.deepcopy(self.row)
                    row["runtime_backend"] = "mlx_cpp_metal" + {
                        "gpu": "", "runtime": "+coreml_runtime_weight", "frozen": "+coreml"}[route]
                    row.update(operation=request["operation"],
                               reference_tokens=1024 if mismatch and route == "runtime" else 3072,
                               plan={"operation": request["operation"],
                                     "qwen21_reference_size": request["qwen21_reference_size"]})
                    kwargs["stdout"].write(json.dumps(row) + "\n")
                return subprocess.CompletedProcess(command, returncode)

            args = ["screen", "--model", str(root), "--model-id", "qwen-image-2.1",
                    "--routes", "gpu,runtime,frozen", "--steps", "40", "--warm-repeats", "1",
                    "--runtime-manifest", str(manifest), "--frozen-manifest", str(manifest),
                    "--reference-size", "512"]
            for path in references:
                args += ["--reference", str(path)]
            with mock.patch.object(SCREEN, "ROOT", root), \
                    mock.patch.object(SCREEN, "wait_for_idle", return_value="idle"), \
                    mock.patch.object(SCREEN, "system_memory", return_value={}), \
                    mock.patch.object(SCREEN, "run_owned", side_effect=run), \
                    mock.patch("builtins.print"):
                output = root/"good"
                with mock.patch.object(sys, "argv", [*args, "--output", str(output)]):
                    SCREEN.main()
                summary = json.loads((output/"summary.json").read_text())
                self.assertEqual(summary["status"], "complete")
                self.assertEqual(summary["chunks"], "auto")
                self.assertFalse(summary["profile"])
                self.assertFalse(summary["memory_sampling"]["enabled"])
                self.assertEqual(summary["warm_repeats"], 1)
                self.assertEqual(summary["operation"], "image.edit")
                self.assertEqual(summary["reference_tokens"], 3072)
                self.assertEqual(summary["reference_size"], 512)
                self.assertEqual(len(summary["references"]), 3)
                self.assertEqual(len(seen), 6)
                for request in seen:
                    self.assertEqual([item["path"] for item in request["inputs"]],
                                     [str(path.resolve()) for path in references])
                    self.assertEqual(request["operation"], "image.edit")
                # Optional sampling applies identically to every route, not
                # just runtime. A lost sample leaves the campaign incomplete.
                def sampled(command, **kwargs):
                    run(command, **kwargs)
                    self.assertEqual(kwargs["interval_ms"], 100)
                    self.assertEqual(kwargs["max_gap_ms"], 500)
                    return {"verified": {"complete": True}, "evidence": kwargs["stem"] + "-memory.jsonl"}
                output = root / "sampled"
                with mock.patch.object(SCREEN, "run_sampled", side_effect=sampled) as sample, \
                        mock.patch.object(sys, "argv", [*args, "--output", str(output), "--sample-memory"]):
                    SCREEN.main()
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(sample.call_count, 3)
                self.assertTrue(summary["memory_sampling"]["enabled"])
                self.assertEqual(summary["status"], "complete")
                self.assertTrue(all(t["sampled_memory"]["verified"]["complete"] for t in summary["trials"]))
                # Cross-build campaigns must hash the selected CLI's library,
                # never silently attribute results to the repository build.
                isolated = root / "isolated"
                isolated.mkdir()
                cli = isolated / "turbocider"
                cli.write_bytes(b"isolated cli")
                cli.chmod(0o755)
                (isolated / "libturbocider.dylib").write_bytes(b"isolated library")
                output = root / "isolated-screen"
                with mock.patch.object(sys, "argv", [*args, "--cli", str(cli), "--output", str(output)]), \
                        mock.patch.object(SCREEN, "run_owned", side_effect=run) as launch:
                    SCREEN.main()
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["library_sha256"], hashlib.sha256(b"isolated library").hexdigest())
                self.assertEqual(summary["cli"]["sha256"], hashlib.sha256(b"isolated cli").hexdigest())
                self.assertTrue(all(call.args[0][0] == str(cli.resolve()) for call in launch.call_args_list))
                output = root / "sample-failed"
                with mock.patch.object(SCREEN, "run_sampled", side_effect=RuntimeError("sample incomplete")), \
                        mock.patch.object(sys, "argv", [*args, "--output", str(output), "--sample-memory"]), \
                        self.assertRaisesRegex(RuntimeError, "sample incomplete"):
                    SCREEN.main()
                summary = json.loads((output / "summary.json").read_text())
                self.assertEqual(summary["status"], "incomplete")
                self.assertEqual(summary["trials"], [])
                mismatch = True
                output = root/"bad"
                with mock.patch.object(sys, "argv", [*args, "--output", str(output)]), \
                        self.assertRaisesRegex(ValueError, "geometry differs across routes"):
                    SCREEN.main()
                # Keep failed native evidence; never report a mismatched pair.
                self.assertTrue((output/"1-runtime.stdout.jsonl").is_file())
                partial = json.loads((output/"summary.json").read_text())
                self.assertEqual(partial["status"], "incomplete")
                self.assertEqual(len(partial["trials"]), 1)
                returncode = 1
                output = root/"first-route-failed"
                with mock.patch.object(sys, "argv", [*args, "--output", str(output)]), \
                        self.assertRaisesRegex(RuntimeError, "gpu failed"):
                    SCREEN.main()
                partial = json.loads((output/"summary.json").read_text())
                self.assertEqual(partial["status"], "incomplete")
                self.assertEqual(partial["trials"], [])
                self.assertTrue((output/"0-gpu.stdout.jsonl").is_file())

    def test_process_check_identifies_python_workloads_without_saving_arguments(self):
        snapshot = "PID %CPU COMM\n101 15.0 Python\n102 8.0 Python\n103 7.0 llama-server\n"
        arguments = ("101 Python /work/ComfyUI/main.py --token private-value\n"
                     "102 Python /work/download_ltx25.py\n103 llama-server\n")
        with mock.patch.object(SCREEN.subprocess, "check_output", side_effect=[snapshot, arguments]), \
                mock.patch.object(COMMON.os, "getpid", return_value=999):
            saved, busy = SCREEN.check_load()
        self.assertEqual(saved, snapshot)
        self.assertEqual(busy, snapshot.splitlines()[1:])
        self.assertNotIn("private-value", saved + str(busy))

    def test_process_check_ignores_self_idle_and_unrelated_processes(self):
        snapshot = "PID %CPU COMM\n101 90.0 turbocider\n102 0.0 Python\n103 90.0 editor\n"
        arguments = "101 turbocider\n102 Python /work/ComfyUI/main.py\n103 editor\n"
        with mock.patch.object(SCREEN.subprocess, "check_output", side_effect=[snapshot, arguments]), \
                mock.patch.object(COMMON.os, "getpid", return_value=101):
            self.assertEqual(SCREEN.check_load(), (snapshot, []))

    def test_idle_preflight_waits_then_returns_latest_snapshot(self):
        with mock.patch.object(COMMON, "check_load", side_effect=[("busy", ["worker"]), ("idle", [])]), \
                mock.patch.object(COMMON.time, "sleep") as sleep, mock.patch("builtins.print"):
            self.assertEqual(SCREEN.wait_for_idle(), "idle")
        sleep.assert_called_once_with(10)

    def test_idle_preflight_stops_after_bounded_wait(self):
        with mock.patch.object(COMMON, "check_load", return_value=("busy", ["worker"])) as check, \
                mock.patch.object(COMMON.time, "sleep") as sleep, mock.patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "no benchmark started"):
                SCREEN.wait_for_idle()
        self.assertEqual(check.call_count, 13)
        self.assertEqual(sleep.call_count, 12)

    def test_benchmark_environment_is_shared_and_does_not_modify_parent(self):
        source = {"PATH": "bin", "TURBOCIDER_QWEN21_PROFILE_GPU_OPS": "1",
                  "TURBOCIDER_Z_RUNTIME_LORA_DIRECT_FP16": "1",
                  "TURBOCIDER_PRIVATE_ANE_DATA_PATH": "w8a8",
                  "TURBOCIDER_ALLOW_PRIVATE_ANE": "1", "TURBOCIDER_ANE_BACKEND": "private",
                  "TURBOCIDER_RUNTIME_ANE_CHUNKS": "99"}
        with mock.patch.dict(COMMON.os.environ, source, clear=True):
            self.assertEqual(SCREEN.benchmark_environment(), {"PATH": "bin"})
            self.assertEqual(dict(COMMON.os.environ), source)

    def test_success_and_gpu_only_scheduler_ablation(self):
        SCREEN.validate_results([self.row], "runtime", 1)
        self.row["hybrid"]["runtime_calls_session_total"] = 0
        SCREEN.validate_results([self.row], "runtime", 1)

    def test_serialized_ffn_spans_are_finite_cumulative_subsets(self):
        row = copy.deepcopy(self.row)
        runtime = row["hybrid"]["runtime_weight"]
        runtime.update(lora_gate_up_seconds_session_total=.1,
                       post_join_seconds_session_total=.2,
                       hybrid_ffn_seconds_session_total=.5)
        SCREEN.validate_results([row, row], "runtime", 2)
        for change in ({"lora_gate_up_seconds_session_total": None},
                       {"lora_gate_up_seconds_session_total": True},
                       {"lora_gate_up_seconds_session_total": -.1},
                       {"post_join_seconds_session_total": float("nan")},
                       {"post_join_seconds_session_total": .6},
                       {"hybrid_ffn_seconds_session_total": None}):
            invalid = copy.deepcopy(row)
            invalid["hybrid"]["runtime_weight"].update(change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "serialized FFN"):
                SCREEN.validate_results([invalid], "runtime", 1)
        decreased = copy.deepcopy(row)
        decreased["hybrid"]["runtime_weight"]["post_join_seconds_session_total"] = .1
        with self.assertRaisesRegex(ValueError, "serialized FFN"):
            SCREEN.validate_results([row, decreased], "runtime", 2)
        runtime["lora_gate_up_seconds_session_total"] = 0
        SCREEN.validate_results([row], "runtime", 1)  # no-adapter/unsplit is valid

    def test_combined_lora_readiness_is_a_subset_of_pre_not_ffn(self):
        row = copy.deepcopy(self.row)
        runtime = row["hybrid"]["runtime_weight"]
        runtime.update(lora_input_ready_seconds_session_total=.8,
                       pre_ffn_seconds_session_total=1.,
                       lora_gate_up_seconds_session_total=0.,
                       post_join_seconds_session_total=.1,
                       hybrid_ffn_seconds_session_total=.2)
        SCREEN.validate_results([row, row], "runtime", 2)
        for key in ("lora_input_ready_seconds_session_total", "pre_ffn_seconds_session_total"):
            for value in (None, True, -.1, float("nan"), float("inf")):
                invalid = copy.deepcopy(row)
                invalid["hybrid"]["runtime_weight"][key] = value
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, "readiness"):
                    SCREEN.validate_results([invalid], "runtime", 1)
        for change in ({"lora_input_ready_seconds_session_total": 1.1},
                       {"pre_ffn_seconds_session_total": .7}):
            invalid = copy.deepcopy(row)
            invalid["hybrid"]["runtime_weight"].update(change)
            with self.assertRaisesRegex(ValueError, "readiness"):
                SCREEN.validate_results([invalid], "runtime", 1)
        for key in ("lora_input_ready_seconds_session_total", "pre_ffn_seconds_session_total"):
            decreased = copy.deepcopy(row)
            decreased["hybrid"]["runtime_weight"][key] = .9 if key.startswith("pre_") else .7
            with self.assertRaisesRegex(ValueError, "readiness"):
                SCREEN.validate_results([row, decreased], "runtime", 2)

    def test_full_gpu_probes_are_a_valid_cumulative_subset(self):
        row = copy.deepcopy(self.row)
        runtime = row["hybrid"]["runtime_weight"]
        runtime.update(gpu_blocks_session_total=4, unsplit_gpu_blocks_session_total=3,
                       full_gpu_probe_blocks_session_total=2, full_gpu_probe_seconds_session_total=.5)
        SCREEN.validate_results([row, row], "runtime", 2)
        for change in ({"full_gpu_probe_blocks_session_total": 4},
                       {"full_gpu_probe_blocks_session_total": True},
                       {"unsplit_gpu_blocks_session_total": 5},
                       {"full_gpu_probe_seconds_session_total": None},
                       {"full_gpu_probe_seconds_session_total": float("nan")},
                       {"full_gpu_probe_seconds_session_total": 0},
                       {"full_gpu_probe_seconds_session_total": True}):
            invalid = copy.deepcopy(row)
            invalid["hybrid"]["runtime_weight"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                SCREEN.validate_results([invalid], "runtime", 1)
        fewer = copy.deepcopy(row)
        fewer["hybrid"]["runtime_weight"]["full_gpu_probe_blocks_session_total"] = 1
        with self.assertRaisesRegex(ValueError, "full GPU probe"):
            SCREEN.validate_results([row, fewer], "runtime", 2)
        zero = copy.deepcopy(row)
        zero["hybrid"]["runtime_weight"].update(full_gpu_probe_blocks_session_total=0,
                                                full_gpu_probe_seconds_session_total=0)
        SCREEN.validate_results([zero], "runtime", 1)

    def test_untimed_hybrid_is_a_cumulative_subset(self):
        row = copy.deepcopy(self.row)
        row["hybrid"]["runtime_weight"].update(
            untimed_hybrid_blocks_session_total=2, hybrid_blocks_session_total=4)
        SCREEN.validate_results([row, row], "runtime", 2)
        for value in (None, True, -1, 5, 1.0):
            bad = copy.deepcopy(row)
            bad["hybrid"]["runtime_weight"]["untimed_hybrid_blocks_session_total"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                SCREEN.validate_results([bad], "runtime", 1)
        fewer = copy.deepcopy(row)
        fewer["hybrid"]["runtime_weight"]["untimed_hybrid_blocks_session_total"] = 1
        with self.assertRaisesRegex(ValueError, "untimed hybrid"):
            SCREEN.validate_results([row, fewer], "runtime", 2)

    def test_async_join_is_an_untimed_subset_not_exposed_gpu_time(self):
        row = copy.deepcopy(self.row)
        runtime = row["hybrid"]["runtime_weight"]
        runtime.update(async_hybrid_blocks_session_total=2,
                       untimed_hybrid_blocks_session_total=3,
                       hybrid_blocks_session_total=5,
                       async_ane_wait_seconds_session_total=.4,
                       hybrid_ffn_seconds_session_total=.8,
                       join_seconds_session_total=.01,
                       gpu_ffn_seconds_session_total=.02)
        SCREEN.validate_results([row, row], "runtime", 2)
        # The async wait can exceed both measured-only counters; treating it
        # as exposed ANE time would invent a regression rather than measure it.
        for change in ({"async_hybrid_blocks_session_total": 4},
                       {"async_hybrid_blocks_session_total": 0},
                       {"async_hybrid_blocks_session_total": True},
                       {"async_ane_wait_seconds_session_total": None},
                       {"async_ane_wait_seconds_session_total": float("nan")},
                       {"async_ane_wait_seconds_session_total": float("inf")},
                       {"async_ane_wait_seconds_session_total": -.1},
                       {"async_ane_wait_seconds_session_total": True},
                       {"async_ane_wait_seconds_session_total": 0},
                       {"async_ane_wait_seconds_session_total": .9},
                       {"hybrid_ffn_seconds_session_total": None}):
            invalid = copy.deepcopy(row)
            invalid["hybrid"]["runtime_weight"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                SCREEN.validate_results([invalid], "runtime", 1)
        for key in ("async_hybrid_blocks_session_total", "async_ane_wait_seconds_session_total"):
            missing = copy.deepcopy(row)
            del missing["hybrid"]["runtime_weight"][key]
            with self.assertRaises(ValueError):
                SCREEN.validate_results([missing], "runtime", 1)
            decreasing = copy.deepcopy(row)
            decreasing["hybrid"]["runtime_weight"][key] /= 2
            if key == "async_hybrid_blocks_session_total":
                decreasing["hybrid"]["runtime_weight"][key] = 1
            with self.assertRaisesRegex(ValueError, "async hybrid"):
                SCREEN.validate_results([row, decreasing], "runtime", 2)
        runtime.update(async_hybrid_blocks_session_total=0, async_ane_wait_seconds_session_total=0)
        SCREEN.validate_results([row], "runtime", 1)  # fixed/profile paths
        old = copy.deepcopy(row)
        del old["hybrid"]["runtime_weight"]["async_hybrid_blocks_session_total"]
        del old["hybrid"]["runtime_weight"]["async_ane_wait_seconds_session_total"]
        SCREEN.validate_results([old, old], "runtime", 2)  # old receipts remain valid
        for pair in ([old, row], [row, old]):
            with self.assertRaisesRegex(ValueError, "availability changed"):
                SCREEN.validate_results(pair, "runtime", 2)

    def test_backend_and_count_must_match(self):
        with self.assertRaisesRegex(ValueError, "result count"):
            SCREEN.validate_results([self.row], "runtime", 2)
        with self.assertRaisesRegex(ValueError, "backend"):
            SCREEN.validate_results([self.row], "gpu", 1)

    def test_gguf_cannot_be_reported_as_dense_or_vice_versa(self):
        for route, suffix in (("gpu", ""), ("runtime", "+coreml_runtime_weight"), ("frozen", "+coreml")):
            row = copy.deepcopy(self.row)
            row["runtime_backend"] = "mlx_cpp_metal_gguf" + suffix
            SCREEN.validate_results([row], route, 1, "z-image-turbo-gguf")
            for dense in ("z-image-turbo", "qwen-image-2.1"):
                with self.subTest(route=route, dense=dense), self.assertRaises(ValueError):
                    SCREEN.validate_results([row], route, 1, dense)
            row["runtime_backend"] = "mlx_cpp_metal" + suffix
            with self.assertRaises(ValueError):
                SCREEN.validate_results([row], route, 1, "z-image-turbo-gguf")

    def test_invalid_timings_rejected(self):
        for value in (None, "7", True, 0, -1, float("nan"), float("inf")):
            for name in ("request_wall", "denoise"):
                row = copy.deepcopy(self.row)
                row["timings_seconds"][name] = value
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    SCREEN.validate_results([row], "runtime", 1)

    def test_failure_or_missing_telemetry_rejected(self):
        variants = [{}, None, {"runtime_failed": True},
                    {**self.row["hybrid"], "runtime_failures_session_total": 1}]
        for hybrid in variants:
            with self.subTest(hybrid=hybrid), self.assertRaises(ValueError):
                SCREEN.validate_results([{**self.row, "hybrid": hybrid}], "runtime", 1)
        for runtime in ({}, None, {"fallback_blocks_session_total": 1},
                        {"fallback_blocks_session_total": 0, "failure_reason": "budget"}):
            row = copy.deepcopy(self.row)
            row["hybrid"]["runtime_weight"] = runtime
            with self.subTest(runtime=runtime), self.assertRaises(ValueError):
                SCREEN.validate_results([row], "runtime", 1)

    def test_frozen_requires_predictions(self):
        self.row["runtime_backend"] = "mlx_cpp_metal+coreml"
        SCREEN.validate_results([self.row], "frozen", 1)
        self.row["hybrid"]["runtime_calls_session_total"] = 0
        with self.assertRaisesRegex(ValueError, "predictions"):
            SCREEN.validate_results([self.row], "frozen", 1)

    def test_lora_requires_binding_and_complete_nonmerged_graph(self):
        for route, suffix, kind in (("gpu", "", None),
                ("runtime", "+coreml_runtime_weight", "runtime_weight_swiglu_lora_inputs"),
                ("frozen", "+coreml", "fused_lora")):
            row = copy.deepcopy(self.row)
            row.update(runtime_backend="mlx_cpp_metal" + suffix,
                       lora_strategy="inference_time", lora_applied_projections=227)
            row["hybrid"]["mlp_output_kind"] = kind
            SCREEN.validate_results([row], route, 1, expect_lora=True)
            for change in ({"lora_applied_projections": 0}, {"lora_applied_projections": True},
                           {"lora_strategy": "in_memory_merge"}):
                with self.subTest(route=route, change=change), self.assertRaises(ValueError):
                    SCREEN.validate_results([{**row, **change}], route, 1, expect_lora=True)
            if route != "gpu":
                row["hybrid"]["mlp_output_kind"] = ""
                with self.assertRaises(ValueError):
                    SCREEN.validate_results([row], route, 1, expect_lora=True)

    def test_missing_or_malformed_session_counters_rejected(self):
        for route in ("runtime", "frozen"):
            for name in ("runtime_calls_session_total", "runtime_failures_session_total"):
                for value in (None, True, False, -1, 0.0, "0"):
                    row = copy.deepcopy(self.row)
                    if route == "frozen":
                        row["runtime_backend"] = "mlx_cpp_metal+coreml"
                    if value is None:
                        del row["hybrid"][name]
                    else:
                        row["hybrid"][name] = value
                    with self.subTest(route=route, name=name, value=value), self.assertRaises(ValueError):
                        SCREEN.validate_results([row], route, 1)
        for value in (None, True, False, -1, 0.0, "0"):
            row = copy.deepcopy(self.row)
            row["hybrid"]["runtime_weight"]["fallback_blocks_session_total"] = value
            with self.subTest(fallback=value), self.assertRaises(ValueError):
                SCREEN.validate_results([row], "runtime", 1)
        del self.row["hybrid"]["runtime_weight"]["failure_reason"]
        with self.assertRaises(ValueError):
            SCREEN.validate_results([self.row], "runtime", 1)

    def test_session_counters_cannot_reset_within_resident_batch(self):
        later = copy.deepcopy(self.row)
        later["hybrid"]["runtime_calls_session_total"] += 1
        SCREEN.validate_results([self.row, later, later], "runtime", 3)
        with self.assertRaisesRegex(ValueError, "counter decreased"):
            SCREEN.validate_results([later, self.row], "runtime", 2)

    def test_gpu_does_not_require_hybrid_telemetry(self):
        self.row["runtime_backend"] = "mlx_cpp_metal"
        del self.row["hybrid"]
        SCREEN.validate_results([self.row], "gpu", 1)

    def test_invalid_prompt_seed_rejected_before_creating_artifacts(self):
        with tempfile.TemporaryDirectory() as scratch:
            output = Path(scratch) / "screen"
            for option in (["--prompt", "  "], ["--seed", "-1"], ["--seed", "2147483648"]):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "tools/validation/runtime_ane_model_screen.py"),
                     "--model", "unused", "--model-id", "z-image-turbo", "--routes", "gpu",
                     "--steps", "8", "--output", str(output), *option],
                    capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("nonempty prompt and seed", result.stderr)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
