from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from qwen21_gguf_unsloth_screen import measured_time


class UnslothTimingTests(unittest.TestCase):
    def test_original_timing_boundaries_and_ambiguity(self):
        log = "get_learned_condition completed, taking 3.37s\nsampling completed, taking 109.95s\ngenerate_image completed in 131.68s\n"
        self.assertEqual(measured_time(log, "get_learned_condition completed"), 3.37)
        self.assertEqual(measured_time(log, "sampling completed"), 109.95)
        self.assertEqual(measured_time(log, "generate_image completed"), 131.68)
        for invalid in ("", log + log, "sampling completed, taking 0s", "sampling completed, taking nans"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                measured_time(invalid, "sampling completed")


if __name__ == "__main__":
    unittest.main()
