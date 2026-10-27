"""Startup report math is independent of model weights and native loading."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("image_first_step", ROOT / "tools/validation/image_first_step.py")
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class ImageFirstStepTests(unittest.TestCase):
    def test_step_boundaries_ignore_duplicates_and_separate_startup(self):
        def event(phase, completed, elapsed):
            return dict(phase=phase, completed=completed, elapsed_seconds=elapsed)
        result = BENCHMARK.summarize([
            event("load", 0, 0), event("load", 1, 2), event("denoise", 0, 3),
            event("denoise", 1, 8), event("denoise", 1, 8.1), event("denoise", 2, 9),
            event("denoise", 3, 12), event("export", 1, 13)], 14)
        self.assertEqual(result["time_to_denoise_seconds"], 3)
        self.assertEqual(result["first_step_seconds"], 5)
        self.assertEqual(result["later_step_median_seconds"], 2)
        self.assertEqual(result["first_to_later_ratio"], 2.5)
        self.assertEqual(result["request_wall_seconds"], 14)
        self.assertEqual(result["phase_spans"]["load"], [0, 2])

    def test_missing_boundaries_are_not_invented(self):
        for events in ([], [dict(phase="denoise", completed=2, elapsed_seconds=10)]):
            result = BENCHMARK.summarize(events, 12)
            self.assertIsNone(result["first_step_seconds"])
            self.assertIsNone(result["later_step_median_seconds"])
            self.assertIsNone(result["first_to_later_ratio"])


if __name__ == "__main__":
    unittest.main()
