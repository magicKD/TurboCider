from pathlib import Path
import hashlib
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from flux4_mflux_saved_precheck import saved_coverage


class Flux4SavedCoverageTests(unittest.TestCase):
    def test_stored_f32_rope_constant_is_verified_not_discarded(self):
        values = {"weight": dict(dtype="U32", shape=[4, 8]), "rotary_emb.inv_freq": dict(dtype="F32", shape=[64])}
        self.assertEqual(saved_coverage(values, values)["required_targets"], 2)

    def test_generated_absent_constant_does_not_require_fake_saved_tensor(self):
        header = {"weight": dict(dtype="BF16", shape=[4, 8])}
        actual = dict(header, **{"rotary_emb.inv_freq": dict(dtype="F32", shape=[64])})
        self.assertEqual(saved_coverage(header, actual)["covered_targets"], 1)

    def test_missing_learned_tensor_and_wrong_extent_are_rejected(self):
        with self.assertRaises(ValueError):
            saved_coverage({}, {"weight": dict(dtype="BF16", shape=[4, 8])})
        with self.assertRaises(ValueError):
            saved_coverage({"weight": dict(dtype="U32", shape=[4, 4])}, {"weight": dict(dtype="U32", shape=[4, 8])})

    def test_f32_exception_is_only_for_stored_rope_with_matching_extent(self):
        for key in ("weight", "rotary_emb.inv_freq"):
            header = {key: dict(dtype="F32", shape=[64])}
            actual = header if key == "weight" else {key: dict(dtype="F32", shape=[32])}
            with self.subTest(key=key), self.assertRaises(ValueError):
                saved_coverage(header, actual)

    def test_portable_compatible_prechecks_are_never_promoted_to_a_speed_campaign(self):
        evidence_path = "docs/design/validation/flux4-mflux-saved-q4-precheck-20261012.json"
        data = json.loads((ROOT / evidence_path).read_text())
        self.assertEqual(data["status"], "complete_precheck_only")
        self.assertIs(data["latency_comparison_valid"], False)
        self.assertIs(data["performance_verified"], False)
        self.assertEqual(data["mflux"]["actual_steps"], [0, 1, 2, 3])
        self.assertTrue(data["mflux"]["runtime_verified"])
        self.assertFalse(data["mflux"]["independent_process_tree_memory_verified"])
        self.assertEqual(data["mlx_serve"]["returned_png"]["dimensions"], [512, 512])
        self.assertEqual(data["mlx_serve"]["http_status"], 200)
        self.assertIs(data["mlx_serve"]["server_terminal"], True)
        self.assertIs(data["mlx_serve"]["latency_comparison_valid"], False)
        for runner in ("mflux", "mlx_serve"):
            path = ROOT / data[runner]["evidence"]
            if path.is_file():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), data[runner]["evidence_sha256"])
        report = json.loads((ROOT / "docs/benchmark/results-20261011.json").read_text())
        self.assertNotIn(evidence_path, [row["evidence"]["path"] for row in report["campaigns"]])


if __name__ == "__main__":
    unittest.main()
