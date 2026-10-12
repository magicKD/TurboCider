import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from ltx_full_config_evidence import audit, checked_cohort, frame_metrics, motion_energy, portable_diagnostics
from ltx_full_config_screen import CASES


def fixture():
    name = "gpu-quality-batch"
    return dict(status="complete_diagnostic", order=[name, name], trials=[dict(index=index, name=name,
        configured=CASES[name], rows=[dict(timings_seconds=dict(request_wall=10))] * 2) for index in range(2)])


class LtxFullConfigEvidenceTests(unittest.TestCase):
    def test_vectorized_offline_stats_keep_original_full_frame_definition(self):
        sys.path.insert(0, str(ROOT / "tools/native"))
        import video_quality_gate as gate
        for left, right in ((bytes(range(256)) * 17, bytes(reversed(range(256))) * 17),
                (bytes([0]) * 300, bytes([255]) * 300), (bytes([31]) * 300, bytes([31]) * 300)):
            actual, expected = frame_metrics(left, right), gate._rgb_metrics(left, right)
            for key in expected:
                self.assertAlmostEqual(actual[key], expected[key], places=12)
            self.assertEqual(motion_energy(left, right), gate._motion_energy(left, right))
        for left, right in ((b"", b""), (b"abc", b"ab")):
            with self.assertRaises(ValueError):
                frame_metrics(left, right)

    def test_complete_actual_configuration_order_not_a_partial_window(self):
        with patch("ltx_full_config_evidence.validate_rows"):
            self.assertEqual(checked_cohort(fixture()), ["gpu-quality-batch"] * 2)
            for key, value in (("status", "running"), ("order", ["gpu-quality-batch"]), ("trials", [])):
                data = fixture()
                data[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    checked_cohort(data)

    def test_actual_recipe_or_finite_request_time_cannot_be_forged(self):
        with patch("ltx_full_config_evidence.validate_rows"):
            data = fixture()
            data["trials"][0]["configured"] = CASES["gpu-quality-exec"]
            with self.assertRaises(ValueError):
                checked_cohort(data)
            data = fixture()
            data["trials"][0]["rows"][0]["timings_seconds"]["request_wall"] = float("nan")
            with self.assertRaises(ValueError):
                checked_cohort(data)

    def test_e5_diagnostic_keeps_cache_identity_without_user_home_path(self):
        values = [dict(kind="coreml_e5_missing_bundle_resource", recompile_required=True,
            resource_path="/Users/example/Library/Caches/turbocider/com.apple.e5rt.e5bundlecache/25G83/hash/model.anehash", stdout_offset=0)]
        result = portable_diagnostics(values)
        self.assertTrue(result[0]["recompile_required"])
        self.assertEqual(result[0]["resource_path"], "<USER_CACHE>/turbocider/com.apple.e5rt.e5bundlecache/25G83/hash/model.anehash")
        self.assertNotIn("/Users/", json.dumps(result))
        self.assertEqual(result[0]["stdout_offset"], 0)

    def test_failed_or_running_screen_never_reads_model_payloads(self):
        with tempfile.TemporaryDirectory(prefix="tc-ltx-config-evidence-") as raw:
            path = Path(raw)
            (path / "summary.json").write_text(json.dumps(dict(status="failed")))
            with self.assertRaises(ValueError):
                audit(path, path / "missing-build")


if __name__ == "__main__":
    unittest.main()
