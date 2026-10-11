from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from z_mflux_framework_screen import native_case, requests, route_environment, validate_runs


class ZFrameworkScreenTests(unittest.TestCase):
    def test_native_routes_keep_existing_real_backend_recipe(self):
        self.assertEqual(native_case("native-private"), dict(format="bf16", mode="private"))
        with self.assertRaises(ValueError):
            native_case("mflux-q4")
        actual = requests(Path("output"), "native-private", 8, 4)
        self.assertEqual(actual[2]["prompt"], actual[3]["prompt"])
        self.assertTrue(all(row["model"] == "z-image-turbo" and row["hybrid_mlp_mode"] == "runtime" and
                            row["steps"] == 8 and row["seed"] == 42 for row in actual))

    def reference(self):
        return dict(runs=[dict(request_wall_including_export=8., actual_steps=list(range(8)),
                               actual_quantization=None, loras=[], prompt_cache_hit=False) for _ in range(4)])

    def test_reference_does_not_invent_repeat_conditioning_cache_hit(self):
        report = self.reference()
        validate_runs(report, "mflux-bf16", 8, 4)
        report["runs"][3]["prompt_cache_hit"] = True
        with self.assertRaises(ValueError):
            validate_runs(report, "mflux-bf16", 8, 4)

    def test_actual_precision_steps_and_finite_latency_are_required(self):
        for field, value in (("actual_steps", [0, 1]), ("actual_quantization", 4), ("loras", ["adapter"]),
                             ("request_wall_including_export", float("nan"))):
            report = self.reference()
            report["runs"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_runs(report, "mflux-bf16", 8, 4)

    def test_affine_routes_are_not_equal_to_original_bf16(self):
        report = self.reference()
        for row in report["runs"]:
            row["actual_quantization"] = 8
        validate_runs(report, "mflux-q8", 8, 4)
        with self.assertRaises(ValueError):
            validate_runs(report, "mflux-q4", 8, 4)
        with self.assertRaises(ValueError):
            validate_runs(dict(runs=report["runs"][:1]), "mflux-q8", 8, 4)

    def test_channel_exploration_does_not_leak_to_other_routes(self):
        candidate = route_environment("native-private", 6144, "f16")
        self.assertEqual(candidate["TURBOCIDER_PRIVATE_ANE_CHANNELS"], "6144")
        self.assertEqual(candidate["TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN"], "0")
        control = route_environment("native-gpu", 6144, "f16")
        self.assertNotIn("TURBOCIDER_PRIVATE_ANE_CHANNELS", control)
        self.assertNotIn("TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN", control)
        with self.assertRaises(ValueError):
            route_environment("native-private", 10240)


if __name__ == "__main__":
    unittest.main()
