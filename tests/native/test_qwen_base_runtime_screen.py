"""Portable base-only request/policy guards; no model/device inference."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/validation"))
import qwen_base_runtime_screen as screen


class BaseRuntimeScreenTests(unittest.TestCase):
    def test_all_routes_keep_original_base_forty_step_work(self):
        for route in ("gpu", "runtime-off", "runtime-surface"):
            for case in ("generate", "edit"):
                args = SimpleNamespace(route=route, case=case, output=Path("/synthetic-output"),
                                       manifest=Path("/synthetic-manifest"), reference=Path("/synthetic-reference"))
                requests = screen.requests(args)
                self.assertEqual([r["seed"] for r in requests], [73, 74, 73])
                for request in requests:
                    self.assertEqual((request["steps"], request["width"], request["height"]), (40, 512, 512))
                    self.assertNotIn("loras", request)
                    self.assertEqual(request["qwen21_dit_cache"], "off")
                    self.assertEqual(request["execution"], "gpu" if route == "gpu" else "gpu_ane")
                    if case == "edit":
                        self.assertEqual(request["qwen21_reference_size"], 1024)
                        self.assertEqual(len(request["inputs"]), 1)
                    else:
                        self.assertNotIn("inputs", request)

    def test_native_surface_cache_is_the_only_runtime_arm_difference(self):
        off, on = screen.environment("runtime-off"), screen.environment("runtime-surface")
        changed = {key for key in off if off[key] != on[key]}
        self.assertEqual(changed, {"TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_BYTES", "TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_MODE"})
        self.assertEqual(on["TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_BYTES"], str(2 << 30))
        self.assertFalse(any("LORA" in key or "STUDENT" in key for key in on))
        self.assertEqual(screen.environment("gpu"), {})

    def test_incomplete_run_cannot_claim_full_base_request(self):
        with self.assertRaises(ValueError):
            screen.validate(dict(runs=[]), "gpu", "generate")

    def test_adapter_and_partial_steps_rejected_before_generic_backend_validation(self):
        metrics = dict(model="qwen-image-2.1", operation="image.generate", width=512, height=512,
                       steps=40, actual_denoise_steps=40, seed=73, prompt_cache_hit=False,
                       acceleration_selection="ordinary base GPU", lora_applied_projections=0)
        for key, value in (("actual_denoise_steps", 39), ("lora_applied_projections", 227),
                           ("acceleration_selection", "Viggle six-step student")):
            m = {**metrics, key: value}
            report = dict(runs=[dict(metrics=m, request_wall_including_export=40.)] * 3)
            with self.subTest(key=key), self.assertRaises(ValueError):
                screen.validate(report, "gpu", "generate")


if __name__ == "__main__":
    unittest.main()
