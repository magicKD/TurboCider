from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from ltx_full_config_screen import make_request, parse_native_stdout, request_seconds, validate_rows


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
                     frames=97, audio=True, execution="gpu", conditioning_cache_hit=index == 1, video_vae_isolation="process",
                     plan=dict(ltx_sol_stage2=False, ltx_stage2_text_rows=0)) for index in range(2)]
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

    def test_known_coreml_diagnostics_glued_to_json_are_retained_not_hidden(self):
        warning = ("Missing E5 bundle resource required for loading ExecutionStreamOperation. Must re-compile the E5. Resource path = "
            "/example cache/turbocider/com.apple.e5rt.e5bundlecache/25G83/model.anehash @ GetANEFModel")
        rows, diagnostics = parse_native_stdout(warning + warning + '{"execution":"gpu_ane"}\n{"execution":"gpu_ane"}\n')
        self.assertEqual(rows, [dict(execution="gpu_ane")] * 2)
        self.assertEqual(len(diagnostics), 2)
        self.assertTrue(all(row["recompile_required"] for row in diagnostics))
        self.assertEqual(diagnostics[0]["resource_path"], "/example cache/turbocider/com.apple.e5rt.e5bundlecache/25G83/model.anehash")

    def test_unknown_or_truncated_stdout_is_not_silently_filtered(self):
        for text in ('unexpected warning\n{}', 'Missing E5 bundle resource\n{}', '{"receipt":', '[]', '{} junk {}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_native_stdout(text)
        self.assertEqual(parse_native_stdout(' \n{"index":0}\n{"index":1}\n'), ([dict(index=0), dict(index=1)], []))

    def test_sol_controls_are_validated_at_actual_native_plan_location(self):
        plan = dict(ltx_sol_stage1=False, ltx_sol_stage2=True, ltx_sol_tau=1.0,
            ltx_sol_dense_edge_blocks=1, ltx_sol_dense_edge_steps=0, ltx_stage2_text_rows=256)
        rows = [dict(model="ltx-2.5-distilled", operation="video.generate", width=704, height=448,
            frames=97, audio=True, execution="gpu", conditioning_cache_hit=index == 1, plan=plan.copy()) for index in range(2)]
        validate_rows(rows, "gpu-text256-batch")
        for name, value in (("ltx_sol_tau", 0.5), ("ltx_stage2_text_rows", 0), ("ltx_sol_stage2", False)):
            rows[0]["plan"] = {**plan, name: value}
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_rows(rows, "gpu-text256-batch")
        rows[0]["plan"] = plan.copy()
        rows[0].pop("plan")
        rows[0].update(plan)
        with self.assertRaises(ValueError):
            validate_rows(rows, "gpu-text256-batch")


if __name__ == "__main__":
    unittest.main()
