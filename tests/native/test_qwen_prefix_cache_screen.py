"""Host-only schedule/receipt and native-observer hook checks."""
import copy
import ctypes as C
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/validation"))
import image_framework_native as native
import qwen_prefix_cache_screen as screen


class PrefixCacheScreenTests(unittest.TestCase):
    def report(self, route="gpu"):
        rows = []
        calls = 0
        for i, (_, mode, flag, seed, hit) in enumerate(screen.points(73)):
            selection = ("edit prefix KV snapshot " + ("hit" if hit else "miss")) if flag else "gpu"
            m = dict(model="qwen-image-2.1", operation="image.edit", width=512, height=512,
                steps=40, actual_denoise_steps=40, seed=seed, prompt_cache_hit=i != 0,
                acceleration_selection=selection)
            if mode != "off":
                m["qwen21_dbcache"] = dict(mode=mode, front_blocks=8, back_blocks=0, warmup_steps=8,
                    max_consecutive=1, threshold=.15, cached_steps=15, saved_middle_blocks=360)
                if route == "frozen":
                    calls += 888
                    m["hybrid"] = dict(checkpoint_sha256_verified=True, runtime_failed=False,
                        runtime_failures_session_total=0, runtime_weight=None, ane_mlp_range=[0, 6144],
                        compute_units="cpuAndNeuralEngine", runtime_calls_session_total=calls)
            rows.append(dict(index=i, metrics=m, request_wall_including_export=40.))
        return {"runs": rows}

    def test_schedule_keeps_seed_matched_controls_and_strong_gpu_baseline(self):
        p = screen.points(73)
        self.assertEqual(len(p), 9)
        self.assertEqual(p[3][3], p[1][3])
        self.assertEqual(p[4][3], p[5][3])
        self.assertNotEqual(p[3][3], p[4][3])
        self.assertEqual(p[-1][1:], ("off", 1, 73, True))
        args = SimpleNamespace(seed=73, prompt="edit", reference=Path("/tmp/ref.png"), output=Path("/tmp/results"))
        for request in screen.requests(args):
            self.assertEqual(request["inputs"], [dict(kind="image", role="reference", path="/tmp/ref.png")])
            self.assertNotIn("qwen21_reference_size", request)
            self.assertEqual(request["steps"], 40)

    def test_actual_receipts_are_required(self):
        screen.validate(self.report(), 73)
        for key, value in (("seed", 74), ("steps", 39), ("actual_denoise_steps", 39),
                           ("prompt_cache_hit", False), ("hybrid", {"enabled": True}),
                           ("acceleration_selection", "edit prefix KV snapshot miss")):
            bad = self.report()
            bad["runs"][3]["metrics"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                screen.validate(bad, 73)
        bad = self.report()
        bad["runs"].pop()
        with self.assertRaises(ValueError):
            screen.validate(bad, 73)

    def test_frozen_request_coverage_keeps_step_zero_gpu_and_last_controls_gpu(self):
        report = self.report("frozen")
        screen.validate(report, 73, "frozen")
        args = SimpleNamespace(seed=73, prompt="edit", reference=Path("/tmp/ref.png"),
            output=Path("/tmp/results"), route="frozen", manifest=Path("/tmp/manifest.json"))
        values = screen.requests(args)
        self.assertTrue(all(r["execution"] == "gpu_ane" and r["qwen21_w8a8"] for r in values[:7]))
        self.assertTrue(all(r["execution"] == "gpu" and "ane_manifest" not in r and
                            "qwen21_w8a8" not in r for r in values[7:]))
        bad = copy.deepcopy(report)
        bad["runs"][3]["metrics"]["hybrid"]["runtime_calls_session_total"] += 32
        with self.assertRaisesRegex(ValueError, "step zero"):
            screen.validate(bad, 73, "frozen")
        for key, value in (("checkpoint_sha256_verified", False), ("runtime_failed", True),
                           ("runtime_failures_session_total", False), ("runtime_failures_session_total", 1),
                           ("runtime_weight", {}), ("ane_mlp_range", [0, 4096]), ("compute_units", "all")):
            bad = copy.deepcopy(report)
            bad["runs"][3]["metrics"]["hybrid"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                screen.validate(bad, 73, "frozen")
        bad = copy.deepcopy(report)
        bad["runs"][-1]["metrics"]["hybrid"] = report["runs"][0]["metrics"]["hybrid"]
        with self.assertRaisesRegex(ValueError, "pure GPU"):
            screen.validate(bad, 73, "frozen")

    def test_temporal_cache_is_not_inferred_from_prefix_hit(self):
        for key, value in (("mode", "off"), ("front_blocks", 1), ("warmup_steps", 4),
                           ("threshold", .25), ("max_consecutive", 4), ("cached_steps", 0),
                           ("saved_middle_blocks", 359)):
            bad = self.report()
            bad["runs"][3]["metrics"]["qwen21_dbcache"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                screen.validate(bad, 73)
        bad = self.report()
        bad["runs"][-1]["metrics"]["qwen21_dbcache"] = copy.deepcopy(bad["runs"][3]["metrics"]["qwen21_dbcache"])
        with self.assertRaises(ValueError):
            screen.validate(bad, 73)

    def test_hook_runs_before_each_inference_and_cleanup_survives_hook_failure(self):
        lib = MagicMock()
        lib.tc_engine_create_model.return_value = 0
        events, buffers = [], []
        def infer(engine, request, callback, context, result, error):
            events.append(("infer", json.loads(request)["index"]))
            buffer = C.create_string_buffer(b'{"ok":true}')
            buffers.append(buffer)
            C.cast(result, C.POINTER(C.c_void_p))[0] = C.cast(buffer, C.c_void_p)
            return 0
        lib.tc_engine_generate.side_effect = infer
        with tempfile.TemporaryDirectory() as directory, patch.object(native.C, "CDLL", return_value=lib):
            output = Path(directory)
            requests = [{"index": 0}, {"index": 1}]
            result = native.generate_requests(Path("lib"), "qwen-image-2.1", Path("model"), requests, output,
                "test", before_request=lambda i, r: events.append(("before", i)))
            self.assertEqual(events, [("before", 0), ("infer", 0), ("before", 1), ("infer", 1)])
            self.assertEqual(len(result["runs"]), 2)
            self.assertEqual(lib.tc_engine_free.call_count, 1)
            def fail(index, request):
                raise ValueError("invalid experiment configuration")
            with self.assertRaisesRegex(ValueError, "invalid experiment"):
                native.generate_requests(Path("lib"), "qwen-image-2.1", Path("model"), requests, output,
                    "test", before_request=fail)
            self.assertEqual(lib.tc_engine_generate.call_count, 2)
            self.assertEqual(lib.tc_engine_free.call_count, 2)

    def test_observation_gates_do_not_grant_quality_from_speed(self):
        gates = dict(minimum_speed_reduction_percent=5., minimum_incremental_rgb_ssim=.98,
                     maximum_incremental_mae_255=3., maximum_incremental_alpha_abs=4)
        a = dict(observations=[dict(reduction_percent=10., quality=dict(
            rgb=dict(ssim=1., mae_255=0.), alpha_max_abs=0))],
            control_restoration=dict(rgba_equal=True))
        self.assertTrue(all(screen.evaluate_observation_gates(a, gates).values()))
        a["observations"][0]["quality"]["rgb"]["ssim"] = .9
        actual = screen.evaluate_observation_gates(a, gates)
        self.assertTrue(actual["speed_passed"])
        self.assertFalse(actual["incremental_quality_passed"])
        a["control_restoration"]["rgba_equal"] = False
        self.assertFalse(screen.evaluate_observation_gates(a, gates)["control_restoration_passed"])


if __name__ == "__main__":
    unittest.main()
