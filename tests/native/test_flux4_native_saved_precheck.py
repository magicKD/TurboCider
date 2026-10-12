from pathlib import Path
import hashlib
import json
import statistics
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from flux4_native_saved_precheck import validate_runs


class Flux4NativePrecheckTests(unittest.TestCase):
    def rows(self):
        return [dict(request_wall_including_export=1., native_event_counts=dict(transformer_block=100),
            metrics=dict(model="flux2-klein-4b", width=512, height=512, steps=4, seed=42,
                text_tokens=512, actual_denoise_steps=4, lora_strategy="none", prompt_cache_hit=index == 3)) for index in range(4)]

    def test_actual_4b_cold_two_fresh_repeat_gpu_sequence(self):
        validate_runs(self.rows())
        with self.assertRaises(ValueError):
            validate_runs(self.rows()[:1])

    def test_9b_wrong_block_count_hybrid_or_cache_cannot_pass_as_packed_4b(self):
        for field, value in (("model", "flux2-klein-9b"), ("steps", 1), ("actual_denoise_steps", 1),
                             ("lora_strategy", "runtime"), ("prompt_cache_hit", True), ("hybrid", {"calls": 80})):
            rows = self.rows()
            rows[0]["metrics"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_runs(rows)
        for change in (dict(transformer_block=128), dict(compiled_single_gpu_block=80)):
            rows = self.rows()
            rows[0]["native_event_counts"].update(change)
            with self.assertRaises(ValueError):
                validate_runs(rows)

    def test_missing_or_invalid_wall_is_not_a_sample(self):
        for value in (True, 0, float("nan"), None):
            rows = self.rows()
            rows[0]["request_wall_including_export"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_runs(rows)

    def test_real_native_precheck_keeps_actual_events_and_is_not_a_paired_campaign(self):
        path = "docs/design/validation/flux4-native-saved-q4-precheck-20261012.json"
        record = json.loads((ROOT / path).read_text())
        self.assertEqual(record["status"], "complete_precheck_only")
        self.assertIs(record["latency_comparison_valid"], False)
        self.assertIs(record["performance_verified"], False)
        self.assertTrue(record["memory"]["verified"]["complete"])
        self.assertEqual([row["actual_transformer_blocks"] for row in record["runs"]], [100]*4)
        self.assertEqual([row["actual_compiled_single_blocks"] for row in record["runs"]], [0]*4)
        self.assertEqual(record["one_process_two_fresh_median_seconds"], statistics.median(
            row["request_wall_including_export"] for row in record["runs"][1:3]))
        catalog = json.loads((ROOT / "docs/benchmark/results-20261011.json").read_text())
        self.assertNotIn(path, [row["evidence"]["path"] for row in catalog["campaigns"]])
        raw = ROOT / record["evidence"]
        if raw.is_file():
            self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(), record["evidence_sha256"])
            validate_runs(json.loads(raw.read_text())["report"]["runs"])


if __name__ == "__main__":
    unittest.main()
