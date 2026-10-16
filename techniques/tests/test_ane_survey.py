"""Offline provenance/gate negative controls, not ANE hardware tests."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs/ane"))
import survey_evidence as evidence
import training_math


class AneSurveyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.capsule = json.loads((ROOT / "ane/experiments-20261016.json").read_text())
        cls.small = next(r for r in cls.capsule["compression_runs"] if r["run_id"] == "compression-c256-r256-seed37-v2")

    def report(self):
        run = copy.deepcopy(self.small)
        return {**run, "schema": "ane-compression-survey-v1", "status": "finished",
                "physical_ane_placement": "unverified", "integer_mac_observed": False}

    def test_all_retained_compression_receipts_recompute(self):
        self.assertEqual(len(self.capsule["compression_runs"]), 4)
        self.assertEqual(len(self.capsule["nonqualifying_runs"]), 1)
        self.assertFalse(self.capsule["production_qualified"])
        self.assertEqual(sum(len(r["samples"]) for r in self.capsule["compression_runs"]), 360)
        self.assertEqual(self.capsule["verifier_sha256"], hashlib.sha256((ROOT / "labs/ane/survey_evidence.py").read_bytes()).hexdigest())
        for run in self.capsule["compression_runs"]:
            report = {**run, "schema": "ane-compression-survey-v1", "status": "finished",
                      "physical_ane_placement": "unverified", "integer_mac_observed": False}
            # Capsule has deliberately summarized plan devices, not raw plans.
            # Reconstitute only for the arithmetic contract, never claim trace.
            for arm, info in report["arms"].items():
                if arm != "mlx-fp16":
                    info["compute_plan"] = {"operations": [{"preferred_device": None if device == "unknown" else device}
                        for device, count in info["plan_preferred_counts"].items() for _ in range(count)]}
            result = evidence.compression(report)
            for arm in evidence.ARMS:
                self.assertEqual(result[arm]["pair_comparison"], run["arms"][arm]["pair_comparison"])

    def reconstructed(self):
        report = self.report()
        for arm, info in report["arms"].items():
            if arm != "mlx-fp16":
                info["compute_plan"] = {"operations": [{"preferred_device": None if device == "unknown" else device}
                    for device, count in info["plan_preferred_counts"].items() for _ in range(count)]}
        return report

    def test_rejects_missing_duplicate_or_incorrect_input(self):
        for kind in ("missing", "duplicate", "generation", "order"):
            report = self.reconstructed()
            if kind == "missing": report["samples"].pop()
            if kind == "duplicate": report["samples"][-1] = copy.deepcopy(report["samples"][0])
            if kind == "generation": report["samples"][0]["input_sha256"] = "0" * 64
            if kind == "order": report["samples"][0]["order"] = ["fp16"] * 5
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                evidence.compression(report)

    def test_rejects_nonfinite_boolean_or_failed_gate(self):
        for bad in (True, 0, -1, float("nan"), float("inf")):
            report = self.reconstructed()
            report["samples"][0]["host_seconds"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                evidence.compression(report)
        for bad in (float("nan"), True, .99):
            report = self.reconstructed()
            report["samples"][0]["metrics"]["relative_l2"] = bad
            with self.subTest(error=bad), self.assertRaises(ValueError):
                evidence.compression(report)

    def test_rejects_promotion_changed_graph_or_weaker_gate(self):
        for kind in ("physical", "integer", "gate", "graph", "summary"):
            report = self.reconstructed()
            if kind == "physical": report["physical_ane_placement"] = "verified"
            if kind == "integer": report["integer_mac_observed"] = True
            if kind == "gate": report["gates_relative_l2"]["lut4"] = .99
            if kind == "graph": report["arms"]["w8a8"]["mil_operator_counts"]["quantize"] = 0
            if kind == "summary": report["arms"]["fp16"]["median_host_seconds"] *= .5
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                evidence.compression(report)

    def test_training_receipts_and_negative_updates(self):
        self.assertEqual(len(self.capsule["training_runs"]), 3)
        for run in self.capsule["training_runs"]:
            report = {**run["summary"], "calls": run["calls"], "steps": run["steps"]}
            self.assertEqual(evidence.training(report)["final_loss"], run["summary"]["final_loss"])
            report["steps"] = [report["steps"][0]] * 12
            with self.assertRaises(ValueError): evidence.training(report)

    def test_gradient_and_adam_epsilon_mathematical_negatives(self):
        value = training_math.witness()
        self.assertLess(value["maximum_gradient_absolute_error"], 1e-9)
        self.assertAlmostEqual(value["adam_first_step_original"], value["adam_scaled_gradient_scaled_epsilon"])
        self.assertGreater(abs(value["adam_first_step_original"] - value["adam_scaled_gradient_same_epsilon"]), .5)

    def test_import_acquisition_is_read_only(self):
        spec = importlib.util.spec_from_file_location("ane_source_fetch", ROOT / "labs/ane/research_sources.py")
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        self.assertIn("maderix/ANE", value.RECIPES)

    def test_manifest_immutable_pins_and_declared_reading_scope(self):
        manifest = json.loads((ROOT / "ane/research-manifest.json").read_text())
        self.assertEqual(len(manifest["repos"]), 18)
        self.assertEqual(sum(len(r["files"]) for r in manifest["repos"]), 37)
        self.assertFalse(manifest["upstream_source_distributed"])
        self.assertFalse(manifest["fork_snapshot_relations"][0]["independent_evidence"])
        self.assertIn("not a URL redirect", manifest["fork_snapshot_relations"][0]["relation"])
        for repo in manifest["repos"]:
            self.assertRegex(repo["commit"], r"^[a-f0-9]{40}$")
            for file in repo["files"]:
                self.assertTrue(evidence.digest(file["sha256"]))
                self.assertIn("/blob/" + repo["commit"] + "/", file["url"])
                for lo, hi in file["read_ranges"]:
                    self.assertLessEqual(1, lo)
                    self.assertLessEqual(lo, hi)
                    self.assertLessEqual(hi, file["lines"])

    def test_planned_cpu_and_ane_remain_different_from_physical_proof(self):
        for run in self.capsule["compression_runs"]:
            preferred = "MLCPUComputeDevice" if run["shape"][1] == 256 else "MLNeuralEngineComputeDevice"
            for arm in ("fp16", "w8", "lut4", "w8a8"):
                self.assertEqual(set(run["arms"][arm]["plan_preferred_counts"]), {"unknown", preferred})
                self.assertEqual(run["arms"][arm]["plan_kind"], "predicted-not-physical")
        for run in self.capsule["training_runs"]:
            if run["summary"]["mode"] == "public-coreml":
                self.assertEqual(run["summary"]["plan_preferred_counts"]["MLCPUComputeDevice"], 1)
                self.assertEqual(run["summary"]["physical_ane_placement"], "unverified")


if __name__ == "__main__":
    unittest.main()
