import unittest

from tools.validation.qwen21_generation_parity import timing_summary


class GenerationTimingTests(unittest.TestCase):
    def test_single_step_has_no_cached_step(self):
        report = timing_summary([4.0], [5.0], 1)
        self.assertIsNone(report["native_median_step_seconds"])
        self.assertIsNone(report["mflux_median_step_seconds"])
        self.assertEqual(report["native_first_step_seconds"], 4.0)
        self.assertEqual(report["denoise_speedup_mflux_over_native"], 1.25)

    def test_prefill_excluded_only_from_cached_median(self):
        report = timing_summary([10.0, 2.0, 4.0], [20.0, 4.0, 8.0], 3)
        self.assertEqual(report["native_median_step_seconds"], 3.0)
        self.assertEqual(report["native_denoise_seconds"], 16.0)
        self.assertEqual(report["denoise_speedup_mflux_over_native"], 2.0)

    def test_rejects_missing_steps(self):
        for a, b, count in [([], [], 0), ([1.0], [], 1), ([1.0], [1.0], 2)]:
            with self.subTest(count=count, a=a, b=b), self.assertRaises(ValueError):
                timing_summary(a, b, count)

    def test_rejects_invalid_timings(self):
        for value in [0.0, -1.0, float("nan"), float("inf")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                timing_summary([value], [1.0], 1)
            with self.subTest(value=value), self.assertRaises(ValueError):
                timing_summary([1.0], [value], 1)


if __name__ == "__main__":
    unittest.main()
