"""CPU-only checks that the paired benchmark cannot accept hidden cache/fallback work."""

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "qwen21_convrot_benchmark", ROOT / "tools/benchmarks/qwen21_convrot_w8a8_benchmark.py")
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class ConvrotBenchmarkTests(unittest.TestCase):
    def test_cli_batch_decodes_multiple_pretty_objects_and_rejects_noise(self):
        first, second = {"first": 1, "nested": {"v": 2}}, {"second": 3}
        wire = json.dumps(first, indent=2) + "\n" + json.dumps(second, indent=2) + "\n"
        self.assertEqual(benchmark.decode_cli_records(wire), [first, second])
        with self.assertRaises(ValueError):
            benchmark.decode_cli_records(wire + "unstructured diagnostic")
        with self.assertRaises(ValueError):
            benchmark.decode_cli_records("[]")

    def test_each_arm_isolates_diagnostics_and_keeps_precision_environment(self):
        inherited = {"PATH": "/usr/bin", "MLX_ENABLE_TF32": "0",
                     "TURBOCIDER_QWEN21_CONVROT_W8A8": "1", "TURBOCIDER_UNRELATED_CACHE": "1"}
        for arm in ("00", "10", "01", "11"):
            env = benchmark.arm_environment(arm, inherited)
            self.assertEqual(env["PATH"], "/usr/bin")
            self.assertEqual(env["MLX_ENABLE_TF32"], "0")
            self.assertNotIn("TURBOCIDER_UNRELATED_CACHE", env)
            self.assertEqual(env["TURBOCIDER_QWEN21_CONVROT_W8A8"], "0")
            self.assertEqual(env["TURBOCIDER_QWEN21_CONVROT_W8A8_ENCODER"], arm[0])
            self.assertEqual(env["TURBOCIDER_QWEN21_CONVROT_W8A8_DIT"], arm[1])

    def test_prompt_sequence_recomputes_encoder_and_pairs_identical_intent(self):
        args = argparse.Namespace(warmup=1, repeats=3, smoke=False, dimension=512, seed=73, dump_tensors=False,
                                  residency="streamed")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = benchmark.request_specs(args, root)
            self.assertEqual([row["case"] for row in left], ["warmup", "short", "long", "short", "long", "short", "long"])
            prompts = [row["request"]["prompt"] for row in left]
            self.assertTrue(all(a != b for a, b in zip(prompts, prompts[1:])))
            self.assertTrue(all(row["request"]["steps"] == 8 and row["request"]["seed"] == 73 for row in left))
            self.assertTrue(all(row["request"]["residency"] == "streamed" for row in left))
            self.assertTrue(all("dump_tensors" not in row["request"] for row in left))

    def test_actual_execution_counters_are_required_for_each_component(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output.png"
            output.write_bytes(b"fixture")
            request = {"model": "qwen-image-2.1", "model_variant": "qwen-image-2.1-turbo", "operation": "image.generate",
                       "width": 256, "height": 256, "steps": 8, "seed": 73, "output": str(output)}
            spec = {"index": 0, "request": request}
            for arm in ("00", "10", "01", "11"):
                record = {**request, "checkpoint": benchmark.CHECKPOINT, "actual_denoise_steps": 8,
                          "qwen21_sampling_schedule": benchmark.SCHEDULE, "prompt_cache_hit": False,
                          "timings_seconds": {key: 1.0 for key in ("request_wall", "text_encode", "denoise", "vae_decode")},
                          "memory": {"mlx_peak_bytes": 100, "mlx_active_bytes": 50}, "qwen_convrot_w8a8": {}}
                for index, component in enumerate(("encoder", "dit")):
                    record["qwen_convrot_w8a8"].update({component + "_enabled": arm[index] == "1",
                        component + "_projections": int(arm[index]) * 7, component + "_activation_rows": int(arm[index]) * 128})
                benchmark.validate_record(record, spec, arm)
                cached = copy.deepcopy(record)
                cached["prompt_cache_hit"] = True
                with self.assertRaisesRegex(ValueError, "conditioning cache"):
                    benchmark.validate_record(cached, spec, arm)
                for component in ("encoder", "dit"):
                    missing = copy.deepcopy(record)
                    del missing["qwen_convrot_w8a8"][component + "_projections"]
                    with self.assertRaisesRegex(ValueError, "requested execution"):
                        benchmark.validate_record(missing, spec, arm)
                short = copy.deepcopy(record)
                short["actual_denoise_steps"] = 2
                with self.assertRaisesRegex(ValueError, "eight denoise"):
                    benchmark.validate_record(short, spec, arm)

    def test_bfloat16_dumps_compare_on_cpu_without_losing_bits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def tensor(path, words):
                header = json.dumps({"tensor": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}).encode()
                path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<HH", *words))
            first, second = root / "a.safetensors", root / "b.safetensors"
            tensor(first, (0x3f80, 0xc000))  # 1, -2
            tensor(second, (0x3f80, 0xc000))
            self.assertEqual(benchmark.read_tensor(first).tolist(), [1.0, -2.0])
            self.assertTrue(benchmark.tensor_metrics(first, second)["exact"])
            tensor(second, (0x3f80, 0xbf80))  # 1, -1
            metrics = benchmark.tensor_metrics(first, second)
            self.assertFalse(metrics["exact"])
            self.assertEqual(metrics["maximum_absolute"], 1.0)
            self.assertAlmostEqual(metrics["relative_l2"], 1 / 5**0.5)

    def test_stream_budget_requires_authorized_prefix_and_complete_uncached_suffix(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output.png"
            output.write_bytes(b"fixture")
            request = dict(model="qwen-image-2.1", model_variant="qwen-image-2.1-turbo", operation="image.generate",
                width=512, height=512, steps=8, seed=73, output=str(output), residency="streamed",
                qwen21_stream_weight_budget_bytes=2 << 30)
            record = {**request, "checkpoint": benchmark.CHECKPOINT, "actual_denoise_steps": 8,
                "qwen21_sampling_schedule": benchmark.SCHEDULE, "prompt_cache_hit": False,
                "plan": {"residency": "streamed"},
                "timings_seconds": {key: 1. for key in ("request_wall", "text_encode", "denoise", "vae_decode")},
                "memory": {"mlx_peak_bytes": 100, "mlx_active_bytes": 50},
                "qwen_convrot_w8a8": {f"{name}_{key}": value for name in ("encoder", "dit")
                    for key, value in (("enabled", True), ("projections", 100), ("activation_rows", 128))},
                "qwen_convrot_streaming": {"requested_weight_budget_bytes": 2 << 30}}
            for name, layers, passes, prefix in (("encoder", 36, 1, 0), ("denoiser", 32, 8, 6)):
                fills = (layers-prefix)*passes
                record["qwen_convrot_streaming"][name] = dict(prefix=prefix, source_precision="original_i8_convrot",
                    matmul_mode="int8_tensorops_w8a8", drained=True, slots=2, peak_live_layer_owners=1,
                    completed_passes=passes, completed_layers=layers*passes, completed_prefix_layers=prefix*passes,
                    fills=fills, completed_streamed_layers=fills, reader_fences=fills, completed_reader_fences=fills,
                    managed_weight_capacity_bytes=1 << 30, derived_weight_capacity_bytes=0,
                    managed_weight_budget_bytes=2 << 30, streamed_source_bytes=1000)
            spec = {"index": 0, "request": request}
            benchmark.validate_record(record, spec, "11")
            evicted = copy.deepcopy(record)
            stage = evicted["qwen_convrot_streaming"]["denoiser"]
            stage.update(prefix_per_pass=[6, 6, 0, 0, 0, 0, 0, 0], final_prefix=0, cache_evictions=1,
                         completed_prefix_layers=12)
            stage.update({key: 244 for key in ("fills", "completed_streamed_layers", "reader_fences", "completed_reader_fences")})
            benchmark.validate_record(evicted, spec, "11")
            invalid = copy.deepcopy(evicted)
            invalid["qwen_convrot_streaming"]["denoiser"]["prefix_per_pass"][3] = 6
            with self.assertRaisesRegex(ValueError, "eviction/pass"):
                benchmark.validate_record(invalid, spec, "11")
            for field, value in (("completed_prefix_layers", 0), ("fills", 256), ("prefix", 32),
                                 ("managed_weight_capacity_bytes", 3 << 30), ("managed_weight_budget_bytes", 4 << 30)):
                bad = copy.deepcopy(record)
                bad["qwen_convrot_streaming"]["denoiser"][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    benchmark.validate_record(bad, spec, "11")
            legacy = copy.deepcopy(spec)
            legacy["request"].pop("qwen21_stream_weight_budget_bytes")
            with self.assertRaises(ValueError):
                benchmark.validate_record(record, legacy, "11")


if __name__ == "__main__":
    unittest.main()
