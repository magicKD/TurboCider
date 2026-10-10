from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from z_base_formats_evidence import best_routes, controls


class ZBaseEvidenceTests(unittest.TestCase):
    def test_gpu_control_is_best_measured_gpu_not_slow_stock_denominator(self):
        rows = [dict(name=name, format="q4", mode=mode, fresh_warm_median=seconds)
                for name, mode, seconds in (("stock", "gpu", 10), ("compiled", "fast", 8),
                                           ("private", "private", 9), ("public", "public", 11))]
        result = best_routes(rows, "q4")
        self.assertEqual(result["gpu"], "compiled")
        self.assertEqual(result["hybrid"], "private")
        self.assertLess(result["gpu_over_hybrid"], 1)
        self.assertLess(result["hybrid_reduction_percent"], 0)

    def test_gpu_only_format_has_no_invented_hybrid(self):
        result = best_routes([dict(name="nvfp4-gpu", format="nvfp4", mode="gpu", fresh_warm_median=8)], "nvfp4")
        self.assertIsNone(result["hybrid"])
        self.assertIsNone(result["gpu_over_hybrid"])

    def test_never_borrow_other_source_winner(self):
        rows = [dict(name="bf16", format="bf16", mode="gpu", fresh_warm_median=2),
                dict(name="q8", format="q8", mode="gpu", fresh_warm_median=10)]
        self.assertEqual(best_routes(rows, "q8")["gpu_fresh_seconds"], 10)

    def test_controls_are_portable_not_machine_paths(self):
        actual = controls({"model": str(ROOT / "models/test.bin"), "option": "1"})
        self.assertEqual(actual, {"model": "models/test.bin", "option": "1"})


if __name__ == "__main__":
    unittest.main()
