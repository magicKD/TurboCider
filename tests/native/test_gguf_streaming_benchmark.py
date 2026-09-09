#!/usr/bin/env python3
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))
SPEC = importlib.util.spec_from_file_location(
    "turbocider_gguf_streaming_benchmark",
    ROOT / "tools/native/benchmark_z_image_gguf_streaming.py",
)
BENCHMARK = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BENCHMARK)


class GGUFStreamingBenchmarkTests(unittest.TestCase):
    def report(self, peak: int) -> dict:
        return {"memory": {"peak_physical_footprint_bytes": peak}}

    def test_order_requires_balanced_repeated_samples(self):
        self.assertEqual(BENCHMARK.validate_order("abba", 2, 4), "ABBA")
        for order, runs, samples in [
            ("AAB", 2, 2), ("ABC", 2, 2), ("AB", 1, 2), ("", 2, 2),
        ]:
            with self.subTest(order=order, runs=runs, samples=samples):
                with self.assertRaises(ValueError):
                    BENCHMARK.validate_order(order, runs, samples)

    def test_component_root_can_be_separate_from_gguf_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gguf_root = root / "gguf"
            component_root = root / "components"
            gguf_root.mkdir()
            component_root.mkdir()
            self.assertEqual(
                BENCHMARK.resolve_component_root(gguf_root, component_root),
                component_root.resolve(),
            )

            with self.assertRaises(SystemExit):
                BENCHMARK.resolve_component_root(gguf_root, root / "missing")

    def test_comparison_separates_performance_memory_and_parity(self):
        parity = [{"shape_equal": True, "finite": True,
                   "pixel_exact": True, "correlation": 1.0,
                   "cosine": 1.0, "mae_255": 0.0}] * 4
        value = BENCHMARK.comparison(
            [10.0, 10.1, 9.9, 10.0],
            [10.1, 10.0, 10.2, 10.1],
            [self.report(15_000)] * 4,
            [self.report(9_000)] * 4,
            parity,
            maximum_performance_ratio=1.02,
            minimum_footprint_reduction=0.25,
            minimum_samples=4,
            minimum_correlation=0.99,
            minimum_cosine=0.995,
            maximum_mae_255=5.0,
            require_pixel_exact=False,
        )
        self.assertTrue(value["passed"])
        self.assertFalse(value["strict_no_slowdown_passed"])
        self.assertAlmostEqual(value["streaming_footprint_reduction"], 0.4)

        slow = BENCHMARK.comparison(
            [10.0] * 4, [10.3] * 4,
            [self.report(15_000)] * 4, [self.report(9_000)] * 4, parity,
            maximum_performance_ratio=1.02,
            minimum_footprint_reduction=0.25,
            minimum_samples=4,
            minimum_correlation=0.99,
            minimum_cosine=0.995,
            maximum_mae_255=5.0,
            require_pixel_exact=False,
        )
        self.assertFalse(slow["passed"])
        self.assertFalse(slow["gates"]["performance_passed"])

        inaccurate = BENCHMARK.comparison(
            [10.0] * 4, [10.0] * 4,
            [self.report(15_000)] * 4, [self.report(9_000)] * 4,
            [{"shape_equal": True, "finite": True,
              "pixel_exact": False, "correlation": 0.98,
              "cosine": 0.999, "mae_255": 1.0}] * 4,
            maximum_performance_ratio=1.02,
            minimum_footprint_reduction=0.25,
            minimum_samples=4,
            minimum_correlation=0.99,
            minimum_cosine=0.995,
            maximum_mae_255=5.0,
            require_pixel_exact=False,
        )
        self.assertFalse(inaccurate["passed"])
        self.assertFalse(inaccurate["gates"]["quality_passed"])

        approximate = BENCHMARK.comparison(
            [10.0] * 4, [10.0] * 4,
            [self.report(15_000)] * 4, [self.report(9_000)] * 4,
            [{"shape_equal": True, "finite": True,
              "pixel_exact": False, "correlation": 0.998,
              "cosine": 0.9998, "mae_255": 2.2}] * 4,
            maximum_performance_ratio=1.02,
            minimum_footprint_reduction=0.25,
            minimum_samples=4,
            minimum_correlation=0.99,
            minimum_cosine=0.995,
            maximum_mae_255=5.0,
            require_pixel_exact=False,
        )
        self.assertTrue(approximate["passed"])
        self.assertFalse(approximate["gates"]["decoded_pixel_exact_required"])

        exact = BENCHMARK.comparison(
            [10.0] * 4, [10.0] * 4,
            [self.report(15_000)] * 4, [self.report(9_000)] * 4,
            [{"shape_equal": True, "finite": True,
              "pixel_exact": False, "correlation": 0.998,
              "cosine": 0.9998, "mae_255": 2.2}] * 4,
            maximum_performance_ratio=1.02,
            minimum_footprint_reduction=0.25,
            minimum_samples=4,
            minimum_correlation=0.99,
            minimum_cosine=0.995,
            maximum_mae_255=5.0,
            require_pixel_exact=True,
        )
        self.assertFalse(exact["passed"])

    def test_missing_or_invalid_pair_cannot_pass(self):
        valid = {
            "shape_equal": True, "finite": True, "pixel_exact": False,
            "correlation": 0.998, "cosine": 0.9998, "mae_255": 2.2,
        }
        common = dict(
            maximum_performance_ratio=1.02,
            minimum_footprint_reduction=0.25,
            minimum_samples=4,
            minimum_correlation=0.99,
            minimum_cosine=0.995,
            maximum_mae_255=5.0,
            require_pixel_exact=False,
        )
        missing = BENCHMARK.comparison(
            [10.0] * 4, [10.0] * 4,
            [self.report(15_000)] * 4, [self.report(9_000)] * 4,
            [valid] * 3,
            **common,
        )
        self.assertFalse(missing["passed"])
        self.assertFalse(missing["gates"]["paired_output_count_passed"])

        nonfinite = dict(valid, finite=False)
        invalid = BENCHMARK.comparison(
            [10.0] * 4, [10.0] * 4,
            [self.report(15_000)] * 4, [self.report(9_000)] * 4,
            [valid, valid, valid, nonfinite],
            **common,
        )
        self.assertFalse(invalid["passed"])
        self.assertFalse(invalid["gates"]["quality_passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
