"""Host-only graph/trace, strong-prefix, seed and fresh-condition contracts."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/validation"))
import qwen_student_graph_screen as screen


class StudentGraphScreenTests(unittest.TestCase):
    def test_only_archived_research_binary_is_admitted(self):
        with patch.object(screen, "sha256_file", return_value=screen.ARCHIVED_LIBRARY_SHA256):
            screen.require_archived_library(Path("archived.dylib"))
        with patch.object(screen, "sha256_file", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "graph retention is retired"):
                screen.require_archived_library(Path("current.dylib"))

    def report(self):
        runs = []
        for i, (_, case, _, enabled, seed, hit, prefix) in enumerate(screen.points(73)):
            selection = "gpu; joint BF16 LoRA A/B operands"
            if case == "edit":
                selection += "; edit prefix KV snapshot " + ("hit" if prefix else "miss")
            if enabled:
                state = "miss" if i in (2, 6, 14) else "hit"
                count = (32 if i == 14 else 64) if state == "miss" else 0
                selection += f"; student graph cache {state}; block wrappers built={count}; block C++ traces={count}; request values cleared before VAE"
            m = dict(model="qwen-image-2.1", operation="image." + case, steps=6,
                actual_denoise_steps=6, width=512, height=512, seed=seed, prompt_cache_hit=hit,
                lora_applied_projections=227, lora_strategy="inference_time", acceleration_selection=selection)
            runs.append(dict(metrics=m, request_wall_including_export=8.))
        return dict(runs=runs)

    def test_real_construction_and_trace_observations_are_required(self):
        observations = screen.validate(self.report(), 73)
        self.assertTrue(observations)
        for change in ("student graph cache hit; block wrappers built=1; block C++ traces=0; request values cleared before VAE",
                       "student graph cache hit; block wrappers built=0; block C++ traces=1; request values cleared before VAE",
                       "student graph cache declined; block wrappers built=64; block C++ traces=64; request values cleared before VAE",
                       "student graph cache miss; block wrappers built=0; block C++ traces=0; request values cleared before VAE"):
            bad = self.report()
            bad["runs"][3]["metrics"]["acceleration_selection"] = "joint BF16 LoRA A/B operands; " + change
            with self.subTest(change=change), self.assertRaises(ValueError):
                screen.validate(bad, 73)

    def test_no_hidden_temporal_ane_or_missing_adapter_work(self):
        for key, value in (("hybrid", {"active": True}), ("student_ffn_reuse", {"enabled": True}),
                           ("qwen21_dbcache", {"cached_steps": 1}), ("lora_applied_projections", 226),
                           ("actual_denoise_steps", 5), ("prompt_cache_hit", False)):
            bad = self.report()
            bad["runs"][3]["metrics"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                screen.validate(bad, 73)

    def test_fresh_conditions_and_strong_prefix_controls_are_matched(self):
        points = screen.points(73)
        self.assertFalse(points[5][5])
        self.assertFalse(points[8][5])
        self.assertEqual(points[3][4], points[1][4])
        self.assertEqual(points[4][4], points[11][4])
        self.assertTrue(all(points[i][6] for i in (13, 14, 15, 16, 17, 18)))
        bad = self.report()
        bad["runs"][15]["metrics"]["acceleration_selection"] = bad["runs"][15]["metrics"]["acceleration_selection"].replace(
            "edit prefix KV snapshot hit", "edit prefix KV snapshot miss")
        with self.assertRaises(ValueError):
            screen.validate(bad, 73)

    def test_return_to_off_cannot_still_report_graph_enabled(self):
        bad = self.report()
        bad["runs"][10]["metrics"]["acceleration_selection"] = bad["runs"][3]["metrics"]["acceleration_selection"]
        with self.assertRaises(ValueError):
            screen.validate(bad, 73)


if __name__ == "__main__":
    unittest.main()
