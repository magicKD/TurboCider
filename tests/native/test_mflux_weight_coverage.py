from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from mflux_weight_coverage import verify_required_sources


class ReferenceWeightCoverageTests(unittest.TestCase):
    def test_fused_source_is_not_proof_that_separate_linears_were_loaded(self):
        header = {"gate_up.weight": dict(shape=[4, 2])}
        mapping = {"gate.weight": [("gate_layer.weight", None)], "up.weight": [("proj.weight", None)]}
        with self.assertRaisesRegex(ValueError, "total 2"):
            verify_required_sources(header, mapping)

    def test_all_required_targets_can_use_alternative_source_names(self):
        mapping = {"old.weight": [("linear.weight", None)], "new.weight": [("linear.weight", None)]}
        result = verify_required_sources({"new.weight": {}}, mapping)
        self.assertEqual(result, dict(required_targets=1, covered_targets=1, missing=[]))

    def test_one_to_many_source_covers_every_required_target(self):
        mapping = {"fused.weight": [("a.weight", None), ("b.weight", None)]}
        self.assertEqual(verify_required_sources({"fused.weight": {}}, mapping)["covered_targets"], 2)

    def test_empty_target_set_is_not_vacuous_coverage(self):
        with self.assertRaises(ValueError):
            verify_required_sources({"unused": {}}, {})


if __name__ == "__main__":
    unittest.main()
