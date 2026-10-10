"""CPU-only evidence checks: a request choice cannot stand in for execution."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "qwen_native_pair", ROOT / "tools/validation/run_qwen_native_e2e.py")
PAIR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAIR)


class PairEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.hybrid = PAIR.pair_routes("hybrid-vs-native", Path("/tmp/fresh-pair"))[1]
        self.result = dict(actual_denoise_steps=20,
                           runtime_backend="native_metal_graph+private_ane_w1+mlx_encoder_vae",
                           native_denoiser_hybrid=dict(enabled=True, weight_staging="generic", successful_ffns=608,
                                                       completed_hybrid_steps=19),
                           native_denoiser_memory_phase=dict(memory_admission=dict(
                               execution_state="succeeded", worker_quarantined=False, tainted=False,
                               ledger_storage_count=0, ledger_storage_bytes=0, ledger_reserved_bytes=0,
                               ledger_pending_release_bytes=0, ledger_unknown_bytes=0)))

    def test_only_complete_executed_hybrid_receipt_passes(self):
        PAIR.verify_native_result(self.result, self.hybrid)
        for key, value in (("enabled", False), ("successful_ffns", 607),
                           ("completed_hybrid_steps", 18)):
            broken = copy.deepcopy(self.result)
            broken["native_denoiser_hybrid"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                PAIR.verify_native_result(broken, self.hybrid)
        for key in ("native_denoiser_hybrid", "native_denoiser_memory_phase", "actual_denoise_steps"):
            broken = copy.deepcopy(self.result)
            del broken[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                PAIR.verify_native_result(broken, self.hybrid)

    def test_physical_cleanup_cannot_be_excused_by_valid_image_or_counts(self):
        for key in ("ledger_storage_count", "ledger_storage_bytes", "ledger_reserved_bytes",
                    "ledger_pending_release_bytes", "ledger_unknown_bytes", "worker_quarantined", "tainted"):
            broken = copy.deepcopy(self.result)
            broken["native_denoiser_memory_phase"]["memory_admission"][key] = 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                PAIR.verify_native_result(broken, self.hybrid)
        broken = copy.deepcopy(self.result)
        broken["runtime_backend"] = "native_metal_graph+mlx_encoder_vae"
        with self.assertRaises(ValueError):
            PAIR.verify_native_result(broken, self.hybrid)

    def test_comparison_profiles_keep_dense_and_hybrid_distinct(self):
        routes = PAIR.pair_routes("hybrid-vs-native", Path("/tmp/fresh-pair"))
        self.assertEqual([r["execution"] for r in routes], ["gpu", "gpu_ane"])
        self.assertTrue(all(r["allow_approximation"] for r in routes))
        self.assertNotIn("qwen21_native_ane_cache", routes[0])
        self.assertEqual(routes[1]["qwen21_native_ane_cache"], "/tmp/fresh-pair/owned-ane-cache")
        oracle = PAIR.pair_routes("native-vs-mlx", Path("/tmp/oracle-pair"))
        self.assertEqual([r["backend"] for r in oracle], ["native-metal-graph", "mlx"])
        self.assertTrue(all(r["execution"] == "gpu" for r in oracle))
        register = PAIR.pair_routes("hybrid-vs-native", Path("/tmp/register"), "register")[1]
        with self.assertRaises(ValueError):
            PAIR.verify_native_result(self.result, register)
        result = copy.deepcopy(self.result)
        result["native_denoiser_hybrid"]["weight_staging"] = "register"
        PAIR.verify_native_result(result, register)

    def test_timeout_exit_zero_race_remains_failed_and_reaped(self):
        process = Mock(pid=101)
        process.wait.side_effect = [subprocess.TimeoutExpired("owned-child", 1), 0]
        with patch.object(PAIR.os, "killpg", side_effect=ProcessLookupError):
            code, timed_out = PAIR.wait_bounded(process, 1)
        self.assertEqual(code, 0)
        self.assertTrue(timed_out)

    def test_scale_cache_requires_actual_fills_hits_and_payload(self):
        route = PAIR.pair_routes("hybrid-vs-native", Path("/tmp/cache-pair"), "register", True)[1]
        result = copy.deepcopy(self.result)
        result["native_denoiser_hybrid"]["weight_staging"] = "register"
        with self.assertRaises(ValueError):
            PAIR.verify_native_result(result, route)
        cache = dict(enabled=True, entries=32, payload_bytes=1048576, hits=576, misses=32)
        result["native_denoiser_hybrid"]["weight_scale_cache"] = cache
        PAIR.verify_native_result(result, route)
        for key, value in (("enabled", False), ("enabled", 1), ("entries", 31),
                           ("payload_bytes", 33554432), ("hits", 575), ("hits", 576.0),
                           ("misses", 33)):
            broken = copy.deepcopy(result)
            broken["native_denoiser_hybrid"]["weight_scale_cache"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                PAIR.verify_native_result(broken, route)

    def test_cache_execution_cannot_replace_a_disabled_experiment(self):
        result = copy.deepcopy(self.result)
        result["native_denoiser_hybrid"]["weight_scale_cache"] = dict(
            enabled=False, entries=0, payload_bytes=0, hits=0, misses=0)
        PAIR.verify_native_result(result, self.hybrid)
        for key in ("entries", "payload_bytes", "hits", "misses"):
            broken = copy.deepcopy(result)
            broken["native_denoiser_hybrid"]["weight_scale_cache"][key] = 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                PAIR.verify_native_result(broken, self.hybrid)
        with self.assertRaises(ValueError):
            PAIR.pair_routes("native-vs-mlx", Path("/tmp/gpu-only"), weight_scale_cache=True)

    def test_unobserved_termination_cannot_be_success(self):
        process = Mock(pid=102)
        process.wait.side_effect = subprocess.TimeoutExpired("owned-child", 1)
        with patch.object(PAIR.os, "killpg") as kill:
            code, timed_out = PAIR.wait_bounded(process, 1)
        self.assertIsNone(code)
        self.assertTrue(timed_out)
        self.assertEqual(kill.call_count, 2)


if __name__ == "__main__":
    unittest.main()
