import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from flux4_derived_hybrid_screen import ORDER, requests, validate_runs


class Flux4DerivedHybridProtocolTests(unittest.TestCase):
    def rows(self, route):
        rows = []
        for index in range(4):
            hybrid = dict(bucket=1536, hidden=3072, output_channels=3072, block_count=20,
                mlp_width=9216, ane_mlp_range=[0, 6144], checkpoint_sha256_verified=True,
                runtime_failed=False, runtime_failures_session_total=0,
                runtime_calls_session_total=80 * (index + 1), calls_session_total=80 * (index + 1),
                warmup_calls_session_total=0, weight_variant="int8_pc", compute_units="cpuAndNeuralEngine")
            rows.append(dict(request_wall_including_export=1.,
                native_event_counts=dict(transformer_block=100, compiled_single_gpu_block=80 if route == "gpu" else 0),
                metrics=dict(model="flux2-klein-4b", width=512, height=512, steps=4, actual_denoise_steps=4,
                    seed=42, text_tokens=512, lora_strategy="none", prompt_cache_hit=index == 3,
                    **({"hybrid": hybrid} if route == "hybrid" else {}))))
        return rows

    def test_two_routes_in_forward_reverse_order_and_complete_actual_work(self):
        self.assertEqual(ORDER, ("gpu", "hybrid", "hybrid", "gpu"))
        for route in ("gpu", "hybrid"):
            validate_runs(self.rows(route), route)
            with self.assertRaises(ValueError):
                validate_runs(self.rows(route)[:1], route)

    def test_source_geometry_or_missing_actual_compilation_is_not_a_label(self):
        for edit in (
            lambda row: row["metrics"].update(model="flux2-klein-9b"),
            lambda row: row["metrics"].update(actual_denoise_steps=3),
            lambda row: row["metrics"].update(prompt_cache_hit=True),
            lambda row: row["native_event_counts"].update(compiled_single_gpu_block=0),
            lambda row: row["native_event_counts"].update(acceleration_gpu_fallback=1),
            lambda row: row.update(request_wall_including_export=float("nan")),
        ):
            rows = self.rows("gpu")
            edit(rows[0])
            with self.assertRaises(ValueError):
                validate_runs(rows, "gpu")

    def test_wrong_source_admission_call_count_warmup_or_runtime_failure_rejected(self):
        for field, value in (("checkpoint_sha256_verified", False), ("runtime_calls_session_total", 79),
                             ("runtime_failed", True), ("warmup_calls_session_total", 20),
                             ("ane_mlp_range", [0, 4096]), ("block_count", 24)):
            rows = copy.deepcopy(self.rows("hybrid"))
            rows[0]["metrics"]["hybrid"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_runs(rows, "hybrid")

    def test_requests_are_base_same_source_no_activation_w8a8_or_old_model_retest(self):
        for route in ("gpu", "hybrid"):
            rows = requests(Path("/output"), route, Path("/manifest.json"))
            self.assertEqual(len(rows), 4)
            for request in rows:
                self.assertEqual(request["model"], "flux2-klein-4b")
                self.assertEqual(request["warmup_iterations"], 0)
                self.assertEqual(request["compile_gpu"], route == "gpu")
                self.assertEqual("ane_manifest" in request, route == "hybrid")
                self.assertNotIn("loras", request)
                self.assertNotIn("w8a8", request)


if __name__ == "__main__":
    unittest.main()
