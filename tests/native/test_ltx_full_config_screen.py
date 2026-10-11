from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from ltx_full_config_screen import make_request, request_seconds, validate_rows


class LtxFullConfigTests(unittest.TestCase):
    def test_approximate_candidates_keep_actual_full_media_task(self):
        actual = make_request("int8-text256-batch", Path("video.mp4"), Path("existing-profile.json"))
        self.assertEqual((actual["width"], actual["height"], actual["frames"], actual["steps"], actual["audio"]), (704, 448, 97, 11, True))
        self.assertTrue(actual["allow_approximation"])
        self.assertTrue(actual["ltx_sol_stage2"])
        self.assertFalse(actual["ltx_sol_stage1"])
        self.assertEqual(actual["ltx_stage2_text_rows"], 256)
        self.assertNotIn("loras", actual)

    def test_quality_baseline_has_no_hidden_algorithm_approximation(self):
        actual = make_request("gpu-quality-batch", Path("video.mp4"))
        self.assertFalse(actual["allow_approximation"])
        self.assertNotIn("ltx_sol_stage2", actual)
        self.assertNotIn("ltx_stage2_text_rows", actual)

    def test_exec_candidate_cannot_relabel_spawned_helper_as_clean_exec(self):
        rows = [dict(model="ltx-2.5-distilled", operation="video.generate", width=704, height=448,
                     frames=97, audio=True, execution="gpu", conditioning_cache_hit=index == 1, video_vae_isolation="process") for index in range(2)]
        validate_rows(rows, "gpu-quality-batch")
        with self.assertRaises(ValueError):
            validate_rows(rows, "gpu-quality-exec")
        for row in rows:
            row["video_vae_isolation"] = "exec"
        validate_rows(rows, "gpu-quality-exec")

    def test_hybrid_profile_and_matching_cache_lifecycle_are_actual(self):
        rows = [dict(model="ltx-2.5-distilled", operation="video.generate", width=704, height=448,
                     frames=97, audio=True, execution="gpu", conditioning_cache_hit=index == 1) for index in range(2)]
        with self.assertRaises(ValueError):
            validate_rows(rows, "int8-quality-batch")
        with self.assertRaises(ValueError):
            make_request("int8-quality-batch", Path("video.mp4"))

    def test_finalizer_estimate_is_not_relabelled_as_persistent_request_wall(self):
        row = dict(timings_seconds=dict(request_wall=10., request_wall_estimate=12.))
        self.assertEqual(request_seconds(row, False), 10.)
        self.assertEqual(request_seconds(row, True), 12.)
        with self.assertRaises(ValueError):
            request_seconds(dict(timings_seconds=dict(request_wall=float("nan"))), False)


if __name__ == "__main__":
    unittest.main()
