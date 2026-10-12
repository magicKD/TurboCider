import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from flux4_reference_warm_screen import validate_trial
from flux4_reference_warm_evidence import audit
from base_model_benchmark_evidence import flux4_warm_reference_campaign, matched_comparison


class FluxWarmReferenceTests(unittest.TestCase):
    def data(self):
        return json.loads((ROOT / "docs/design/validation/flux4-reference-warm-current-20261012.json").read_text())

    def test_complete_reference_only_actual_steps_and_no_native_retest(self):
        data = self.data()
        self.assertEqual(data["status"], "complete_reference_only")
        self.assertIs(data["native_retested"], False)
        self.assertIs(data["same_window_native_pair"], False)
        for trial in data["trials"]:
            validate_trial(trial)
            self.assertEqual(len(trial["report"]["runs"]), 4)
        self.assertNotIn("/Users/", json.dumps(data))

    def test_generating_stage_not_decoding_is_the_four_step_proof(self):
        trial = copy.deepcopy(self.data()["trials"][1])
        self.assertEqual(trial["report"]["runs"][0]["denoise_progress"], [[i,4] for i in range(1,5)])
        for progress in ([], [[4,4]], [[i,4] for i in range(1,5)] + [[4,4]]):
            trial["report"]["runs"][0]["denoise_progress"] = progress
            with self.assertRaises(ValueError):
                validate_trial(trial)

    def test_failed_memory_or_nan_wall_cannot_be_recovered(self):
        for edit in (lambda t:t["memory"]["verified"].update(complete=False),
                     lambda t:t["report"]["runs"][0].update(request_wall_including_export=float("nan"))):
            trial = copy.deepcopy(self.data()["trials"][0])
            edit(trial)
            with self.assertRaises(ValueError):
                validate_trial(trial)

    def test_known_postvalidation_recovery_keeps_failed_original_binding(self):
        data = self.data()
        self.assertEqual(data["collector_record_status"], "failed")
        self.assertIs(data["collector_stage_name_recovered"], True)
        path = ROOT / data["evidence"]
        if path.is_file():
            raw = json.loads(path.read_text())
            self.assertEqual(raw["status"], "failed")
            self.assertEqual(raw["error"], "actual four denoise completion events required")
            checked = audit(path.parent)
            self.assertEqual(checked["routes"], data["routes"])

    def test_reference_warm_does_not_upgrade_native_precheck_or_paired_headline(self):
        campaign = flux4_warm_reference_campaign("docs/design/validation/flux4-reference-warm-current-20261012.json")
        self.assertEqual({r["warm_sample_count"] for r in campaign["routes"]}, {2})
        campaign["in_current_scope"] = True
        self.assertIs(campaign["native_comparison_valid"], False)
        with self.assertRaises(ValueError):
            matched_comparison(campaign, "native-gpu", "mlxserve-q4", "not paired")

    def test_preserved_current_configuration_uses_exact_existing_precheck(self):
        cfg = json.loads((ROOT / "docs/benchmark/best-local-configs-20261011.json").read_text())["current_flux4_available_configuration"]
        native = json.loads((ROOT / cfg["native_evidence"]).read_text())
        self.assertEqual(cfg["two_fresh_precheck_seconds"], native["one_process_two_fresh_median_seconds"])
        self.assertIsNone(cfg["gpu_ane_current_q4_result"])
        self.assertEqual(cfg["request"]["execution"], "gpu")
        self.assertEqual(cfg["request"]["steps"], 4)


if __name__ == "__main__":
    unittest.main()
