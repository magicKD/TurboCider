"""Host-only composition/counter checks; no native library or model loads."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/validation"))
import qwen_cache_hybrid_screen as screen


class QwenCacheHybridScreenTests(unittest.TestCase):
    def test_joint_report_supports_generation_only_and_rejects_mismatched_cases(self):
        reports = {route: dict(runs=[dict(case="generate")]) for route in ("gpu", "frozen")}
        self.assertEqual(screen.comparison_cases(reports), ("generate",))
        reports["frozen"]["runs"].append(dict(case="edit"))
        with self.assertRaises(ValueError):
            screen.comparison_cases(reports)
        reports["gpu"]["runs"].append(dict(case="edit"))
        self.assertEqual(screen.comparison_cases(reports), ("generate", "edit"))
        reports["gpu"]["runs"].append(dict(case="unknown"))
        with self.assertRaises(ValueError):
            screen.comparison_cases(reports)

    def row(self, mode="off", saved=0, prior=0, route="frozen"):
        args = SimpleNamespace(child=route, seed=29, output=Path("/tmp/qwen-test"),
                               reference=Path("/tmp/reference.png"), manifest=Path("/tmp/manifest.json"))
        request = screen.make_request(args, "generate", mode, 1)
        metrics = dict(model="qwen-image-2.1", operation="image.generate", width=512, height=512,
                       steps=40, seed=29, actual_denoise_steps=40, prompt_cache_hit=True)
        if mode != "off":
            threshold, maximum = screen.PRESETS[mode]
            metrics["qwen21_dbcache"] = dict(mode="diagnostic", front_blocks=8, back_blocks=0,
                warmup_steps=8, threshold=threshold, max_consecutive=maximum,
                cached_steps=saved // 24, saved_middle_blocks=saved)
        if route == "frozen":
            metrics["hybrid"] = dict(runtime_failed=False, runtime_failures_session_total=0,
                checkpoint_sha256_verified=True, runtime_weight=None, ane_mlp_range=[0, 6144],
                compute_units="cpuAndNeuralEngine", runtime_calls_session_total=prior + 1248 - saved)
        return dict(position=1, mode=mode, request=request, metrics=metrics, request_wall_including_export=20.)

    def test_sequence_keeps_warmup_and_two_controls(self):
        self.assertEqual(screen.sequence(["conservative", "fast"]),
                         ("off", "off", "conservative", "fast", "off"))
        for modes in ([], ["off"], ["fast", "fast"], ["unknown"]):
            with self.assertRaises(ValueError):
                screen.sequence(modes)

    def test_explicit_off_and_diagnostic_candidates_are_distinct(self):
        self.assertEqual(self.row()["request"]["qwen21_dit_cache"], "off")
        self.assertNotIn("qwen21_dit_cache", self.row("fast", 600)["request"])

    def test_named_preset_uses_request_field_and_requires_actual_preset_receipt(self):
        args = SimpleNamespace(child="frozen", seed=29, output=Path("/tmp/qwen-test"),
            reference=Path("/tmp/reference.png"), manifest=Path("/tmp/manifest.json"), cache_api="preset")
        request = screen.make_request(args, "generate", "conservative", 1)
        self.assertEqual(request["qwen21_dit_cache"], "conservative")
        row = self.row("conservative", 360)
        row["cache_api"] = "preset"
        with self.assertRaises(ValueError):
            screen.validate_row(row, "frozen", 0)
        row["metrics"]["qwen21_dbcache"]["mode"] = "conservative"
        self.assertEqual(screen.validate_row(row, "frozen", 0), 888)

    def test_actual_call_delta_subtracts_only_skipped_middle_blocks(self):
        self.assertEqual(screen.validate_row(self.row(), "frozen", 0), 1248)
        row = self.row("conservative", 360, prior=1248)
        self.assertEqual(screen.validate_row(row, "frozen", 1248), 2136)
        row["metrics"]["hybrid"]["runtime_calls_session_total"] += 1
        with self.assertRaises(ValueError):
            screen.validate_row(row, "frozen", 1248)

    def test_failed_or_wrong_hybrid_is_not_speed_evidence(self):
        for key, value in (("runtime_failed", True), ("runtime_failures_session_total", 1),
                           ("runtime_failures_session_total", False),
                           ("checkpoint_sha256_verified", False), ("ane_mlp_range", [0, 4096]),
                           ("compute_units", "all"), ("runtime_weight", {})):
            row = self.row()
            row["metrics"]["hybrid"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                screen.validate_row(row, "frozen", 0)

    def test_cache_coverage_and_parameters_are_not_inferred_from_label(self):
        base = self.row("balanced", 504)
        for key, value in (("mode", "balanced"), ("front_blocks", 1), ("back_blocks", 1),
                           ("warmup_steps", 4), ("threshold", .5), ("max_consecutive", 4),
                           ("cached_steps", 0), ("cached_steps", True), ("saved_middle_blocks", 503)):
            row = copy.deepcopy(base)
            row["metrics"]["qwen21_dbcache"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                screen.validate_row(row, "frozen", 0)

    def test_control_rejects_cache_and_gpu_rejects_hybrid(self):
        row = self.row()
        row["metrics"]["qwen21_dbcache"] = {"cached_steps": 1}
        with self.assertRaises(ValueError):
            screen.validate_row(row, "frozen", 0)
        with self.assertRaises(ValueError):
            screen.validate_row(self.row(), "gpu", 0)
        self.assertEqual(screen.validate_row(self.row(route="gpu"), "gpu", 0), 0)

    def test_incomplete_steps_false_cache_state_and_bad_walls_fail(self):
        for key, value in (("actual_denoise_steps", 39), ("seed", 42), ("prompt_cache_hit", False)):
            row = self.row()
            row["metrics"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                screen.validate_row(row, "frozen", 0)
        for value in (0, -1, float("nan"), float("inf"), True):
            row = self.row()
            row["request_wall_including_export"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                screen.validate_row(row, "frozen", 0)


if __name__ == "__main__":
    unittest.main()
