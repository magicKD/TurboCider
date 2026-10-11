import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from base_model_benchmark_evidence import measured


class BaseBenchmarkReportTests(unittest.TestCase):
    def setUp(self):
        self.report = json.loads((ROOT / "docs/benchmark/results-20261011.json").read_text())

    def test_expanded_goal_remains_active_without_promoting_previous_stage(self):
        self.assertEqual(self.report["status"], "partial_expanded_evidence_catalog")
        self.assertIs(self.report["goal_complete"], False)
        self.assertTrue((ROOT / self.report["completion_audit"]).is_file())
        self.assertIn("not release", self.report["goal_completion_scope"])
        self.assertIs(self.report["qualification_passed"], False)
        self.assertTrue((ROOT / self.report["expansion_status"]).is_file())
        self.assertTrue(self.report["missing_current_comparisons"])
        self.assertTrue(self.report["unmeasured_or_inapplicable"])
        self.assertGreaterEqual(len(self.report["campaigns"]), 18)

    def test_complete_actual_sample_arrays_and_medians(self):
        for campaign in self.report["campaigns"]:
            self.assertTrue(campaign["timing_scope"] if "timing_scope" in campaign else campaign["note"])
            for route in campaign["routes"]:
                samples = route["warm_seconds"]
                self.assertEqual(route["warm_sample_count"], len(samples))
                self.assertTrue(all(type(value) in (float, int) and math.isfinite(value) and value > 0 for value in samples))
                self.assertAlmostEqual(route["warm_median_seconds"], statistics.median(samples), places=12)

    def test_bound_raw_evidence_when_locally_available(self):
        for campaign in self.report["campaigns"]:
            binding = campaign["evidence"]
            path = ROOT / binding["path"]
            self.assertRegex(binding["sha256"], r"^[a-f0-9]{64}$")
            if path.is_file():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), binding["sha256"], str(path))
            elif binding["path"].startswith("docs/"):
                self.fail("committed evidence missing: " + str(path))
            # Raw outputs are deliberately not distributed with the repo.

    def test_failed_quality_and_busy_campaign_not_promoted(self):
        ltx = next(c for c in self.report["campaigns"] if c["model"] == "ltx-2.5-distilled")
        self.assertEqual({r["quality_status"] for r in ltx["routes"] if r["route"] != "gpu"},
                         {"experimental-quality-fail"})
        busy = next(c for c in self.report["campaigns"] if c["evidence"]["status"] == "complete_busy_diagnostic")
        self.assertIn("both CPU load checks failed", busy["note"])

    def test_docs_local_links_and_no_machine_specific_paths(self):
        for doc in sorted((ROOT / "docs/benchmark").glob("*.md")):
            for link in re.findall(r"\]\(([^)]+)\)", doc.read_text()):
                if "://" not in link:
                    self.assertTrue((doc.parent / link.split("#")[0]).exists(), (str(doc), link))
            self.assertNotIn("/Users/", doc.read_text())
        for path in (ROOT / "docs/benchmark/results-20261011.json",
                     ROOT / "docs/benchmark/local-model-inventory-20261011.json",
                     ROOT / "docs/design/validation/ltx-base-current-20261011.json"):
            self.assertNotIn("/Users/", path.read_text())

    def test_inventory_component_availability_never_grants_runtime_proof(self):
        data = json.loads((ROOT / "docs/benchmark/local-model-inventory-20261011.json").read_text())
        self.assertEqual(len(data["rows"]), 15)
        for row in data["rows"]:
            self.assertIs(row["runtime_support_verified"], False)
            if row["missing_components"]:
                self.assertIs(row["assets_present"], False)

    def test_missing_or_invalid_latency_is_not_a_success_sample(self):
        for cold, warm in (([], [1]), ([1], []), ([True], [1]), ([1], [float("nan")]), ([1], [0])):
            with self.subTest(cold=cold, warm=warm), self.assertRaises(ValueError):
                measured("test", cold, warm)


if __name__ == "__main__":
    unittest.main()
