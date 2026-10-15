"""Host-only checks for the matched merge screen; no inference."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/validation"))
import branch_merge_benchmark as screen


class MergeBenchmarkTests(unittest.TestCase):
    def trial(self, branch, cold, first, second, pixel="same"):
        return dict(model="qwen", branch=branch, route="native-gpu",
            report={"runs": [dict(request_wall_including_export=value,
                metrics={"timings_seconds": {"denoise": value - 1}})
                for value in (cold, first, second)]},
            memory={"verified": {"tree_peak_phys_footprint_bytes": 1024, "swap_out_bytes": 0}},
            pngs=[dict(rgba_sha256=pixel)] * 3)

    def test_counterbalanced_branch_order(self):
        for route, expected in (("native-gpu", ["before", "after", "after", "before"]),
                                ("native-frozen", ["after", "before", "before", "after"])):
            self.assertEqual([branch for branch, value in screen.ORDER if value == route], expected)

    def test_full_request_median_excludes_cold_and_keeps_memory_scope(self):
        trials = [self.trial("before", 100, 10, 12), self.trial("after", 200, 8, 10),
                  self.trial("after", 300, 7, 9), self.trial("before", 400, 11, 13)]
        result = screen.aggregate(trials)["qwen/native-gpu"]
        self.assertEqual(result["before"]["fresh_warm_samples"], [10, 12, 11, 13])
        self.assertEqual(result["before"]["median_seconds"], 11.5)
        self.assertEqual(result["after"]["median_seconds"], 8.5)
        self.assertEqual(result["after"]["process_tree_peak_bytes"], [1024, 1024])
        self.assertEqual(result["after"]["cold_seconds"], [200, 300])
        self.assertEqual(result["speedup"], 11.5 / 8.5)
        self.assertTrue(result["all_before_after_rgba_exact"])
        changed = copy.deepcopy(trials)
        changed[0]["pngs"][0]["rgba_sha256"] = "changed"
        self.assertFalse(screen.aggregate(changed)["qwen/native-gpu"]["all_before_after_rgba_exact"])


if __name__ == "__main__":
    unittest.main()
