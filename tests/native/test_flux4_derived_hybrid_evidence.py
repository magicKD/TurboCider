import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from flux4_derived_hybrid_evidence import validate_screen
from base_model_benchmark_evidence import best_known_flux4_comparisons, flux4_derived_hybrid_campaign, matched_comparison

EVIDENCE = "docs/design/validation/flux4-derived-hybrid-current-20261013.json"


class Flux4DerivedHybridEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / EVIDENCE).read_text())

    def test_actual_source_branch_calls_media_and_memory_are_complete(self):
        validate_screen(self.data)
        self.assertEqual(self.data["source_inputs_verified"], 514)
        self.assertEqual(self.data["source_derivation"]["budget"]["derived_tensors"], 307)
        self.assertEqual(len(self.data["source_derivation"]["conversions"]), 40)
        self.assertEqual(sum(len(t["pngs"]) for t in self.data["trials"]), 16)
        self.assertIs(self.data["original_packed_native_retested"], False)
        self.assertIs(self.data["external_retested"], False)
        self.assertNotIn("/Users/", (ROOT / EVIDENCE).read_text())
        raw = ROOT / self.data["evidence"]
        if raw.is_file():
            self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(), self.data["evidence_sha256"])
            validate_screen(json.loads(raw.read_text()))

    def test_label_only_median_incomplete_memory_missing_image_or_calls_rejected(self):
        for edit in (
            lambda d: d["routes"][0].update(fresh_warm_median=.001),
            lambda d: d["trials"][0]["memory"]["verified"].update(complete=False),
            lambda d: d["trials"][0]["pngs"].pop(),
            lambda d: d["trials"][1]["report"]["runs"][0]["metrics"]["hybrid"].update(runtime_calls_session_total=79),
            lambda d: d.update(original_packed_native_retested=True),
        ):
            wrong = copy.deepcopy(self.data)
            edit(wrong)
            with self.assertRaises(ValueError):
                validate_screen(wrong)

    def test_full_png_channels_preserve_exact_controls_and_nonzero_hybrid_delta(self):
        self.assertEqual(len(self.data["image_comparisons"]), 32)
        comparisons = self.data["image_comparisons"]
        controls = [r for r in comparisons if r["kind"] in ("same_route_forward_reverse", "prior_packed_vs_gpu")]
        self.assertEqual(len(controls), 16)
        self.assertTrue(all(r["png_pixel_exact"] for r in controls))
        hybrid = [r for r in comparisons if r["kind"] == "derived_gpu_vs_hybrid"]
        self.assertEqual(len(hybrid), 8)
        self.assertGreater(max(r["metrics"]["rgb_mae_u8"] for r in hybrid), 2.5)
        self.assertGreater(min(r["metrics"]["rgb_ssim_gaussian11_valid"] for r in hybrid), .98)
        for row in comparisons:
            self.assertIn(row["png_mode"], ("RGB", "RGBA"))
            self.assertEqual(row["png_channels"], 3 if row["png_mode"] == "RGB" else 4)
            if row["png_channels"] == 4:
                self.assertTrue(row["metrics"]["alpha"]["pixel_exact"])
            else:
                self.assertNotIn("alpha", row["metrics"])
        self.assertTrue(all(r["qualification_passed"] is False for r in comparisons))

    def test_native_pair_cannot_be_promoted_to_external_framework_headline(self):
        campaign = flux4_derived_hybrid_campaign(EVIDENCE)
        self.assertIs(campaign["native_comparison_valid"], True)
        self.assertIs(campaign["reference_comparison_valid"], False)
        campaign["in_current_scope"] = True
        with self.assertRaises(ValueError):
            matched_comparison(campaign, "hybrid", "gpu", "not an external framework pair")
        wrong = copy.deepcopy(self.data)
        wrong["source_derivation"]["all_saved_tensor_bitpatterns_verified"] = False
        with mock.patch("base_model_benchmark_evidence.source", return_value=(wrong, {})):
            with self.assertRaises(ValueError):
                flux4_derived_hybrid_campaign(EVIDENCE)

    def test_preserved_reference_comparisons_bind_coefficients_without_false_pair(self):
        rows = best_known_flux4_comparisons()
        self.assertEqual([row["reference_route"] for row in rows], ["mflux-q4", "mlxserve-q4"])
        self.assertTrue(all(row["same_window_pair"] is False for row in rows))
        self.assertTrue(all(row["original_coefficient_identity_verified"] for row in rows))
        self.assertAlmostEqual(rows[0]["speedup"], 1.4801759028121846)
        self.assertAlmostEqual(rows[1]["speedup"], 1.6720673673385917)
        original_source = __import__("base_model_benchmark_evidence").source

        def wrong_source(path, statuses):
            data, binding = original_source(path, statuses)
            if "warm-current" in path:
                data["model_sha256"][next(iter(data["model_sha256"]))] = "0" * 64
            return data, binding

        with mock.patch("base_model_benchmark_evidence.source", side_effect=wrong_source):
            with self.assertRaises(ValueError):
                best_known_flux4_comparisons()


if __name__ == "__main__":
    unittest.main()
