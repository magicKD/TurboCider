"""CPU-only checks for sparse benchmark reporting."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
import json
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))
spec = importlib.util.spec_from_file_location("sparse_benchmark", ROOT / "tools/native/benchmark_ltx_sparse_stage2.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
evaluation_spec = importlib.util.spec_from_file_location(
    "sparse_evaluation", ROOT / "tools/native/evaluate_ltx_sparse_run.py")
evaluation = importlib.util.module_from_spec(evaluation_spec)
evaluation_spec.loader.exec_module(evaluation)
import summarize_ltx_sparse as summary
import benchmark_ltx_qkv_replay as replay
import benchmark_ltx_attention as core_benchmark
import io


class SparseBenchmarkTests(unittest.TestCase):
    def test_ane_profile_isolates_stage2_mlp(self):
        valid = dict(schema="turbocider-ltx-ane-v1", mlp_stage_mask=2, kv_stage_mask=0)
        module.validate_ane_stage2_profile(valid)
        for patch_value in ({"mlp_stage_mask": 3}, {"mlp_stage_mask": 2.0},
                            {"kv_stage_mask": False}, {"kv_stage_mask": 1},
                            {"qkv_stage2": "model"}, {"v2a_stage1": "model"},
                            {"schema": "other"}):
            with self.subTest(patch_value=patch_value), self.assertRaises(ValueError):
                module.validate_ane_stage2_profile(dict(valid, **patch_value))

    def test_replay_budgets_include_matched_block_fractions(self):
        self.assertEqual(replay.replay_keep_budgets(5376), [16, 21, 32, 42, 63, 64, 128])
        self.assertEqual(replay.replay_keep_budgets(14080), [16, 32, 55, 64, 110, 128, 165])
        self.assertEqual(replay.replay_keep_budgets(1), [1, 16, 32, 64, 128])
        self.assertLessEqual(max(replay.replay_keep_budgets(16384)), 256)
        for invalid in (0, 16385, True, 1.5):
            with self.assertRaises(ValueError):
                replay.replay_keep_budgets(invalid)

    def test_core_runner_preserves_existing_report(self):
        with patch.object(sys, "argv", ["benchmark", "--report", __file__]), \
                patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                core_benchmark.main()
        self.assertEqual(error.exception.code, 2)

    def test_core_runner_rejects_invalid_runs_before_execution(self):
        with patch.object(sys, "argv", ["benchmark", "--report", "unused.json", "--runs", "0"]), \
                patch.object(Path, "exists", return_value=False), \
                patch.object(sys, "stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                core_benchmark.main()
        self.assertEqual(error.exception.code, 2)

    def test_per_head_reconstructs_energy_weighted_error(self):
        row = {"heads": 2, "relative_l2": (2 / 101) ** .5, "per_head": [
            {"head": 0, "squared_error": 1., "reference_squared_norm": 100.,
             "relative_l2": .1, "max_abs": 1., "exact_block_fraction": .5},
            {"head": 1, "squared_error": 1., "reference_squared_norm": 1.,
             "relative_l2": 1., "max_abs": 1., "exact_block_fraction": 1.}]
        }
        replay.validate_head_metrics(row)
        wrong = deepcopy(row)
        wrong["relative_l2"] = .55
        with self.assertRaisesRegex(ValueError, "reconstruct"):
            replay.validate_head_metrics(wrong)
        wrong = deepcopy(row)
        wrong["per_head"][1]["head"] = 0
        with self.assertRaisesRegex(ValueError, "duplicated"):
            replay.validate_head_metrics(wrong)

    def test_zero_energy_head_has_undefined_relative_error(self):
        row = {"heads": 1, "relative_l2": 0., "per_head": [
            {"head": 0, "squared_error": 0., "reference_squared_norm": 0.,
             "relative_l2": None, "max_abs": 0., "exact_block_fraction": 1.}]}
        replay.validate_head_metrics(row)
        row["per_head"][0]["relative_l2"] = 0.
        with self.assertRaisesRegex(ValueError, "per-head relative"):
            replay.validate_head_metrics(row)

    def test_missing_head_metrics_reject_stale_probe(self):
        with self.assertRaisesRegex(ValueError, "rebuild"):
            replay.validate_head_metrics({"heads": 32})

    def test_summary_never_promotes_missing_quality(self):
        report = dict(schema="ltx-sparse-stage2-paired-v1", complete=True,
                      workload={}, candidate={}, method="ABBA", library_sha256="library",
                      shader_sha256="shader", quality=[], runs=[
                          dict(variant="baseline", result={"timings_seconds": {"stage2": 40.0}}),
                          dict(variant="candidate", result={"timings_seconds": {"stage2": 32.0}})])
        result = summary.summarize(report)
        self.assertEqual(result["rgb_quality"], "unverified")
        self.assertFalse(result["stage2_input_isolated"])
        self.assertEqual(result["stage2_speedup"], 1.25)
        self.assertIsNone(result["request_wall_seconds"])
        self.assertIsNone(result["request_wall_speedup"])
        with_wall = deepcopy(report)
        with_wall["runs"][0]["result"]["timings_seconds"]["request_wall"] = 80.0
        with_wall["runs"][1]["result"]["timings_seconds"]["request_wall"] = 72.0
        self.assertAlmostEqual(summary.summarize(with_wall)["request_wall_speedup"], 80 / 72)
        for field in ("stage2", "request_wall"):
            for value in (0, -1, float("nan"), float("inf"), True, "40"):
                invalid = deepcopy(with_wall)
                invalid["runs"][0]["result"]["timings_seconds"][field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "invalid"):
                    summary.summarize(invalid)
        incomplete = deepcopy(report)
        incomplete["complete"] = False
        self.assertIsNone(summary.summarize(incomplete))
        self.assertEqual(summary.summarize(report, {"complete": False, "comparisons": [
            {"rgb_gate_passed": True}]})["rgb_quality"], "unverified")
        self.assertEqual(summary.summarize(report, {"complete": True, "comparisons": [
            {"rgb_gate_passed": False}]})["rgb_quality"], "failed")

    def test_audio_isolation_and_stage2_drift_are_reported(self):
        def metrics(left, right):
            return {"byte_exact": left.name != "stage2_audio.bf16"}
        with patch.object(evaluation, "bf16_metrics", side_effect=metrics):
            result = evaluation.tensor_metrics(Path("reference"), Path("candidate"), True)
        self.assertTrue(result["stage1_audio.bf16"]["byte_exact"])
        self.assertFalse(result["stage2_audio.bf16"]["byte_exact"])

    def test_stage1_audio_drift_rejects_isolation(self):
        def metrics(left, right):
            return {"byte_exact": left.name != "stage1_audio.bf16"}
        with patch.object(evaluation, "bf16_metrics", side_effect=metrics):
            with self.assertRaisesRegex(RuntimeError, "Stage-2 isolation failed"):
                evaluation.tensor_metrics(Path("reference"), Path("candidate"), True)

    def test_preflight_validates_candidate_before_gpu_work(self):
        library = Mock()
        library.tc_plan_json.side_effect = [0, 1]
        base = {"model": "ltx-2.5-distilled"}
        candidate = {"ltx_sparse_mode": 5, "ltx_sparse_keep_blocks": 32}
        with patch.object(module, "consume", side_effect=['{}', None, None, 'unsupported mode']):
            with self.assertRaisesRegex(RuntimeError, 'candidate preflight failed: unsupported mode'):
                module.validate_requests(library, base, candidate)
        requests = [json.loads(call.args[0]) for call in library.tc_plan_json.call_args_list]
        self.assertEqual(requests, [base, dict(base, **candidate)])
        self.assertEqual(base, {"model": "ltx-2.5-distilled"})

    def test_preflight_retains_native_resolved_plans(self):
        library = Mock()
        library.tc_plan_json.return_value = 0
        with patch.object(module, "consume", side_effect=['{"ltx_sparse_mode":0}', None,
                                                         '{"ltx_sparse_mode":5}', None]):
            plans = module.validate_requests(library, {}, {"ltx_sparse_mode": 5})
        self.assertEqual(plans["baseline"]["ltx_sparse_mode"], 0)
        self.assertEqual(plans["candidate"]["ltx_sparse_mode"], 5)

    def test_profile_records_preserve_stage_and_overlapping_branches(self):
        records = module.parse_native_profile(
            "unrelated native log\n"
            "ltx_c_profile stage=1 video_self_ms=100.25 audio_self_ms=4.25 parallel_stream_wall_ms=105.0\n"
            "ltx_c_profile stage=2 video_self_ms=230.125 video_ffn_ms=1.25e2 parallel_ffn_wall_ms=126\n")
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["stage"], 1)
        self.assertEqual(records[1]["stage"], 2)
        self.assertEqual(records[1]["video_self_ms"], 230.125)
        self.assertEqual(records[1]["video_ffn_ms"], 125.0)
        self.assertEqual(records[0]["parallel_stream_wall_ms"], 105.0)

    def test_missing_profile_is_not_zero_timing(self):
        self.assertEqual(module.parse_native_profile("ordinary output"), [])


if __name__ == "__main__":
    unittest.main()
