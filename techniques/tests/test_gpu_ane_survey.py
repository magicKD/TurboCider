"""Offline GPU/ANE research receipt, provenance and idle-admission contracts."""
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import unittest

ROOT = Path(__file__).resolve().parents[1]
SURVEY = ROOT / "gpu_ane"


def module(name):
    spec = importlib.util.spec_from_file_location("survey_" + name, SURVEY / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class IdleAdmissionTests(unittest.TestCase):
    def test_nested_utilization_uses_maximum(self):
        probe = module("guarded_run")
        self.assertEqual(probe.utilization([{"IORegistryEntryChildren": [
            {"PerformanceStatistics": {"Device Utilization %": 0}},
            {"PerformanceStatistics": {"Device Utilization %": 7}}]}]), 7)

    def test_missing_and_invalid_counters_fail_closed(self):
        probe = module("guarded_run")
        for value in (True, -1, 101, float("nan"), float("inf"), "0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                probe.utilization({"PerformanceStatistics": {"Device Utilization %": value}})
        with self.assertRaises(ValueError):
            probe.utilization({"PerformanceStatistics": {}})


class ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((SURVEY / "experiment-evidence.json").read_text())
        cls.curator = module("curate_experiments")

    def test_no_inherited_hardware_or_product_claim(self):
        for field in ("production_qualified", "physical_overlap_proven", "whole_request_tested", "integer_mac_observed"):
            self.assertIs(self.data[field], False)
        content = (SURVEY / "experiment-evidence.json").read_text()
        self.assertNotIn("/Users/", content)
        self.assertNotIn("Hardware UUID", content)

    def test_all_gpu_pairs_recompute(self):
        self.assertEqual([r["seed"] for r in self.data["gpu_runs"]], [53, 61])
        total = 0
        for process in self.data["gpu_runs"]:
            self.assertEqual(process["admission"]["preflight_percent"], [0] * 5)
            self.assertEqual(len(process["runs"]), 14)
            for run in process["runs"]:
                self.assertEqual(len(run["samples"]), 32)
                stats = self.curator.paired(run["samples"])
                self.assertEqual(stats["paired_ratios"], run["paired_ratios"])
                self.assertAlmostEqual(stats["paired_median_speedup"], run["paired_median_speedup"], places=12)
                for sample in run["samples"]:
                    self.assertTrue(math.isfinite(sample["relative_l2"]))
                    self.assertLessEqual(sample["relative_l2"], run["relative_l2_limit"])
                total += len(run["samples"])
        self.assertEqual(total, 896)

    def test_all_compression_calls_and_medians(self):
        self.assertEqual(len(self.data["compression_runs"]), 2)
        for run in self.data["compression_runs"]:
            self.assertEqual(len(run["samples"]), 65)
            self.assertEqual(set(run["arms"]), {"fp16", "w8", "lut4", "w8a8", "mlx-fp16"})
            for name, arm in run["arms"].items():
                samples = [s for s in run["samples"] if s["arm"] == name]
                self.assertEqual(sorted(s["trial"] for s in samples), list(range(-3, 10)))
                median = statistics.median(s["host_seconds"] for s in samples if s["phase"] == "timed") * 1000
                self.assertAlmostEqual(median, arm["median_host_ms"], places=12)
                self.assertEqual(max(s["metrics"]["relative_l2"] for s in samples), arm["maximum_relative_l2"])
                self.assertEqual(arm["physical_ane_placement"], "unverified")
            # "component-pass" deliberately does not hide LUT4's large error.
            self.assertGreater(run["arms"]["lut4"]["maximum_relative_l2"], 0.2)

    def test_invalid_pair_and_timing_controls(self):
        c = self.curator
        for value in (True, 0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                c.positive(value)
        rows = [{"arm": a, "block": 0, "ms": t} for a, t in zip(
            ("baseline", "candidate", "candidate", "baseline"), (4, 1, 3, 8))]
        self.assertEqual(c.paired(rows)["paired_median_speedup"], 3)
        for broken in ([], rows[:3], [rows[1], rows[0], *rows[2:]],
                       [{**r, "block": 9} for r in rows]):
            with self.assertRaises(ValueError):
                c.paired(broken)


class DocumentationTests(unittest.TestCase):
    def test_source_identity_and_pr_states(self):
        data = json.loads((SURVEY / "sources.json").read_text())
        ids = [r["id"] for r in data["repositories"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 16)
        for repo in data["repositories"]:
            self.assertRegex(repo["commit"], r"^[a-f0-9]{40}$")
            for source in repo["files"]:
                self.assertRegex(source["sha256"], r"^[a-f0-9]{64}$")
                self.assertFalse(Path(source["path"]).is_absolute())
                self.assertNotIn("..", Path(source["path"]).parts)
                self.assertTrue(source["read_ranges"])
                for first, last in source["read_ranges"]:
                    self.assertTrue(0 < first <= last)
        self.assertEqual(len(data["pull_requests"]), 6)
        self.assertEqual(sum(p["state_observed"] == "merged" for p in data["pull_requests"]), 3)

    def test_survey_pages_reachable_and_local_links_exist(self):
        index = (SURVEY / "README.md").read_text()
        for page in SURVEY.glob("*.md"):
            self.assertTrue(page.read_text().startswith("# "))
            if page.name != "README.md":
                self.assertIn("(" + page.name + ")", index)
            outside_fences, fenced = [], False
            for line in page.read_text().splitlines():
                if line.startswith("```"):
                    fenced = not fenced
                elif not fenced:
                    outside_fences.append(line)
            for target in re.findall(r"\]\(([^\s)]+)\)", "\n".join(outside_fences)):
                if "://" in target or target.startswith("#"):
                    continue
                path = (page.parent / target.split("#")[0]).resolve()
                self.assertTrue(path.is_relative_to(ROOT), target)
                self.assertTrue(path.exists(), f"{page.name}: {target}")


if __name__ == "__main__":
    unittest.main()
