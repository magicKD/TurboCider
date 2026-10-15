"""Offline identity/gate/evidence regressions. Never import ANEForge or MLX."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs"))
import search_contracts as contracts
SPEC = importlib.util.spec_from_file_location("aneforge_data_verifier", ROOT / "scripts/curate_aneforge_campaign.py")
curator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(curator)


class SearchContracts(unittest.TestCase):
    def identities(self, **changes):
        args = {"structure": {"graph": "ffn", "layout": "row-major"}, "environment": {"source": "v1", "os": "fixture"},
                "weights_sha256": "a" * 64, "input_sha256": "b" * 64, "limit": .01}
        args.update(changes)
        return contracts.identities(**args)

    def test_weight_source_does_not_inherit_structural_qualification(self):
        a, b = self.identities(), self.identities(weights_sha256="c" * 64)
        self.assertEqual(a["structural_search_hint"], b["structural_search_hint"])
        self.assertNotEqual(a["compile_asset"], b["compile_asset"])
        self.assertNotEqual(a["numerical_qualification"], b["numerical_qualification"])

    def test_input_budget_and_environment_invalidate_the_right_identity(self):
        base = self.identities()
        for change in ({"input_sha256": "c" * 64}, {"limit": .001}):
            value = self.identities(**change)
            self.assertEqual(value["compile_asset"], base["compile_asset"])
            self.assertNotEqual(value["numerical_qualification"], base["numerical_qualification"])
        value = self.identities(environment={"source": "v1", "os": "new"})
        self.assertNotEqual(value["compile_asset"], base["compile_asset"])

    def test_keys_require_real_content_hashes_and_finite_budget(self):
        for changes in ({"weights_sha256": "not-a-hash"}, {"weights_sha256": "x" * 64},
                        {"limit": True}, {"limit": float("nan")}, {"limit": 0}, {"environment": {}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.identities(**changes)

    def test_error_budget_is_strict_and_typed(self):
        for value in (True, None, float("nan"), float("inf"), -.01, .01001):
            self.assertFalse(contracts.numeric_pass({"finite": True, "relative_l2": value}, .01))
        self.assertFalse(contracts.numeric_pass({"finite": 1, "relative_l2": 0.}, .01))
        self.assertFalse(contracts.numeric_pass({"finite": True, "relative_l2": 0.}, True))
        self.assertTrue(contracts.numeric_pass({"finite": True, "relative_l2": .01}, .01))


class ANEForgeEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = json.loads((ROOT / "review/aneforge-evidence-20261016.json").read_text())
        cls.by_id = {item["id"]: item for item in cls.corpus["runs"]}

    def report(self):
        return copy.deepcopy(self.by_id["ffn-exp-confirm32"]["report"])

    def test_all_experiments_failure_and_lineage_retained(self):
        self.assertEqual(len(self.corpus["runs"]), 15)
        self.assertEqual(self.corpus["curator_sha256"], hashlib.sha256((ROOT / "scripts/curate_aneforge_campaign.py").read_bytes()).hexdigest())
        measured = [item for item in self.corpus["runs"] if item["report"]["status"] == "measured"]
        self.assertEqual(len(measured), 13)
        self.assertEqual(sum(len(item["report"]["samples"]) for item in measured), 351)
        for item in measured:
            self.assertEqual(curator.verify_device(item["report"]), item["report"]["comparisons"])
            self.assertTrue(item["child_exit_confirmed"])
            self.assertTrue(all(s["local_snapshot_available"] for s in item["source_snapshots"]))
        failed = self.by_id["cpu"]["report"]
        self.assertEqual(failed["errors"][0]["type"], "TypeError")
        self.assertIn("unexpected keyword", failed["errors"][0]["message"])
        self.assertTrue(all(not s["local_snapshot_available"] for s in self.by_id["cpu"]["source_snapshots"]))

    def test_upstream_cpu_control_is_not_a_device_benchmark(self):
        value = self.by_id["cpu-v2"]["report"]["contracts"]
        self.assertTrue(value["same_structure_key"])
        self.assertTrue(value["different_weight_blob_and_content_key"])
        self.assertEqual(value["cache_hit"]["measurement_calls"], 0)
        self.assertEqual(value["cache_hit"]["build_calls"], 1)
        self.assertFalse(value["dylib_loaded"])
        self.assertEqual(value["mocked_precision_unmet_budget"]["chosen_error"], .2)
        self.assertIn("NO variant met", value["mocked_precision_unmet_budget"]["reason"])
        self.assertEqual(value["route_guard"]["causal_candidates"], [])
        self.assertTrue(value["route_guard"]["unmasked_candidates"])

    def test_rejected_baseline_cannot_become_a_speedup_denominator(self):
        value = self.report()
        self.assertIsNone(value["comparisons"][0]["median_paired_host_ratio"])
        value["comparisons"][0]["median_paired_host_ratio"] = 2.
        with self.assertRaises(ValueError):
            curator.verify_device(value)

    def test_first_warmup_failure_is_not_hidden_by_good_timed_rows(self):
        for phase in ("first", "warmup", "measured"):
            value = self.report()
            row = next(r for r in value["samples"] if r["arm"] == "ane-candidate" and r["phase"] == phase)
            row["metrics"]["relative_l2"] = .1
            result = contracts.paired_result(value["samples"], "gpu", "ane-candidate")
            self.assertFalse(result["all_calls_numeric_pass"])
            self.assertIsNone(result["median_paired_host_ratio"])

    def test_pair_boundary_dtype_hash_time_and_phase_tampering_refused(self):
        for change in ("duplicate", "missing-first", "input", "boundary", "dtype", "time", "hash", "order"):
            value = self.report()
            if change == "duplicate": value["samples"][-1] = value["samples"][-2]
            elif change == "missing-first": value["samples"] = value["samples"][3:]
            elif change == "input": value["samples"][-1]["input_sha256"] = "a" * 64
            elif change == "boundary": value["samples"][-1]["boundary"] = "execute-only"
            elif change == "dtype": value["samples"][-1]["output_dtype"] = "float16"
            elif change == "time": value["samples"][-1]["wall_ns"] = True
            elif change == "hash": value["samples"][-1]["input_sha256"] = "z" * 64
            else: value["samples"][-3:] = value["samples"][-3:][::-1]
            with self.subTest(change=change), self.assertRaises(ValueError):
                curator.verify_device(value)

    def test_exp_rewrite_seed_confirmation_and_scalar_counterexample(self):
        one, two = self.by_id["ffn-exp-gpu32"]["report"], self.report()
        self.assertNotEqual(one["seed"], two["seed"])
        self.assertNotEqual(one["source_weight_sha256"], two["source_weight_sha256"])
        self.assertTrue(two["comparisons"][-1]["all_calls_numeric_pass"])
        self.assertEqual(two["independent_oracle_sha256"], hashlib.sha256((ROOT / "labs/ane/precision_oracle.py").read_bytes()).hexdigest())
        self.assertFalse(two["comparisons"][-1]["production_qualified"])
        scalar = self.by_id["activation-scalar"]["report"]["activation_scalar_witness"]
        self.assertEqual(len(scalar), 6)
        zero = next(row for row in scalar if row["input"] == 0.)
        self.assertEqual(zero["fp64_expected"], 0.)
        self.assertEqual(zero["ane-candidate"]["maximum"], 0.)
        self.assertNotEqual(zero["ane-baseline"]["maximum"], 0.)
        for row in scalar:
            expected = row["input"] / (1 + math.exp(-row["input"]))
            self.assertAlmostEqual(expected, row["fp64_expected"], places=15)

    def test_alive_views_and_resident_diagnostics_do_not_claim_lifetime_or_product_qualification(self):
        value = self.report()
        view = value["alive_view_control"]
        self.assertTrue(view["view_updated_by_next_call"])
        self.assertTrue(view["model_result_copy_stable"])
        self.assertTrue(view["poison_fully_overwritten"])
        self.assertFalse(view["read_after_release_performed"])
        self.assertFalse(value["resident_gpu_diagnostic"]["formal_performance_score"])
        self.assertEqual(value["physical_ane_placement"], "not independently observed")


if __name__ == "__main__":
    unittest.main()
