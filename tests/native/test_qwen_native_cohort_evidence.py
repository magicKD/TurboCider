import copy
import json
from pathlib import Path
import statistics
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from qwen_native_cohort_evidence import summarize


class QwenNativeCohortTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "docs/design/validation/qwen21-native-cohort-current-20261011.json").read_text())

    def test_valid_native_abba_samples_and_real_successful_work(self):
        self.assertEqual(self.data["status"], "complete_native_cohort")
        self.assertEqual([trial["route"] for trial in self.data["trials"]],
                         ["native-gpu", "native-frozen", "native-frozen", "native-gpu"])
        for route in self.data["routes"]:
            self.assertEqual(len(route["fresh_warm_seconds"]), 4)
            self.assertEqual(route["fresh_warm_median"], statistics.median(route["fresh_warm_seconds"]))
        for trial in self.data["trials"]:
            self.assertIs(trial["memory"]["complete"], True)
            for index, row in enumerate(trial["runs"]):
                self.assertIs(row["prompt_cache_hit"], index == 3)
                if trial["route"] == "native-frozen":
                    self.assertEqual(row["hybrid"]["runtime_calls_session_total"], (index + 1) * 1248)
                    self.assertIs(row["hybrid"]["checkpoint_sha256_verified"], True)
                    self.assertIs(row["hybrid"]["runtime_failed"], False)

    def test_invalid_noise_reference_is_not_a_fast_framework_denominator(self):
        self.assertIs(self.data["reference_latency_ranking_valid"], False)
        self.assertEqual({route["route"] for route in self.data["routes"]}, {"native-gpu", "native-frozen"})
        self.assertIn("random initialization", self.data["invalid_reference_reason"])
        self.assertTrue(all(route["ranking_valid"] is False for route in self.data["invalid_reference_diagnostics"]))

    def test_incomplete_or_valid_reference_is_not_this_diagnostic(self):
        for value in (dict(status="running", reference_latency_ranking_valid=False),
                      dict(status="complete_native_diagnostic_invalid_reference", reference_latency_ranking_valid=True)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize(value)

    def test_portable_and_ratio_not_minimum_single_sample(self):
        a, b = (row["fresh_warm_median"] for row in self.data["routes"])
        self.assertEqual(self.data["native_gpu_over_frozen"], a / b)
        self.assertEqual(self.data["frozen_reduction_percent"], 100 * (1 - b / a))
        self.assertNotIn("/Users/", json.dumps(self.data))
        self.assertIs(self.data["goal_complete"], False)


if __name__ == "__main__":
    unittest.main()
