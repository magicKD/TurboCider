"""Synthetic survey accounting and independent packing controls; no GPU work."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("engine_probe", ROOT / "labs/metal/engine_probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class SurveyAccountingTests(unittest.TestCase):
    def test_abba_uses_paired_block_ratios(self):
        samples = [{"arm": a, "ms": t} for a, t in zip(
            ["baseline", "candidate", "candidate", "baseline"] * 2,
            [4, 2, 2, 4, 8, 2, 2, 8])]
        result = probe.summarize(samples)
        self.assertEqual(result["paired_ratios"], [2, 4])
        self.assertEqual(result["paired_median_speedup"], 3)

    def test_incomplete_invalid_and_nonfinite_samples_fail(self):
        block = [{"arm": arm, "ms": 1} for arm in ["baseline", "candidate", "candidate", "baseline"]]
        for bad in (block[:3], list(reversed(block[:2])) + block[2:],
                    [{**x, "ms": float("nan")} for x in block],
                    [{**x, "ms": 0} for x in block]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                probe.summarize(bad)

    def test_registry_pins_sources_without_runtime_overclaim(self):
        registry = json.loads((ROOT / "metal/research-sources.json").read_text())
        self.assertGreaterEqual(len(registry["repositories"]), 12)
        self.assertEqual(len(registry["pull_requests"]), 8)
        for repo in registry["repositories"]:
            self.assertRegex(repo["commit"], r"^[0-9a-f]{40}$")
            self.assertTrue(repo["files"])
            for source in repo["files"]:
                self.assertRegex(source["sha256"], r"^[0-9a-f]{64}$")
                self.assertIn(repo["commit"], source["url"])
                self.assertTrue(source["review_scope"])
        for pr in registry["pull_requests"]:
            self.assertRegex(pr["diff_sha256"], r"^[0-9a-f]{64}$")
            self.assertFalse(pr["locally_benchmarked"])

    def test_capsule_covers_both_seeds_all_samples_and_current_source(self):
        import hashlib
        capsule = json.loads((ROOT / "metal/experiment-evidence-20261016.json").read_text())
        self.assertFalse(capsule["production_qualified"])
        self.assertEqual([r["seed"] for r in capsule["accepted_runs"]], [17, 29])
        sha = hashlib.sha256((ROOT / "labs/metal/engine_probe.py").read_bytes()).hexdigest()
        for report in capsule["accepted_runs"]:
            self.assertEqual(report["source_sha256"], sha)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(len(report["runs"]), 14)
            for run in report["runs"]:
                self.assertEqual(len(run["samples"]), 32)
                self.assertEqual(probe.summarize(run["samples"])["paired_ratios"], run["paired_ratios"])
                for value in run["checks"].values():
                    self.assertTrue(value["finite"])
                    self.assertLessEqual(value["relative_l2"], run["relative_l2_limit"])
                self.assertTrue(all(s["relative_l2"] <= run["relative_l2_limit"] for s in run["samples"]))


class IndependentPackingTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("optional NumPy only; no GPU required")
        self.np = np

    def test_affine_nibble_order_scales_and_biases(self):
        np = self.np
        out = probe.affine_oracle(np, [[0x76543210, 0xFEDCBA98]], [[0.5, 2.0]], [[-1.0, 3.0]], 4, group_size=8)
        expected = [[i * 0.5 - 1 if i < 8 else i * 2 + 3 for i in range(16)]]
        np.testing.assert_array_equal(out, expected)
        wrong = np.asarray(expected)[:, ::-1]
        self.assertGreater(probe.error_metrics(np, expected, wrong)["relative_l2"], 0.1)

    def test_affine_bad_groups_and_nonfinite_metrics_fail(self):
        np = self.np
        with self.assertRaises(ValueError):
            probe.affine_oracle(np, [[0]], [[1, 2]], [[0, 0]], 4, group_size=8)
        for bad in ([float("nan")], [float("inf")], [1, 2]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                probe.error_metrics(np, [1], bad)


if __name__ == "__main__":
    unittest.main()
