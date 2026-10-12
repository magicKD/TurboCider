import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from qwen21_mflux_framework_screen import requests, validate_runs


class QwenFrameworkScreenTests(unittest.TestCase):
    def test_full_base_geometry_and_repeat_separated(self):
        actual = requests(Path("output"), 40, 4, "native-gpu")
        self.assertEqual(len(actual), 4)
        self.assertEqual(len({row["prompt"] for row in actual}), 3)
        self.assertEqual(actual[2]["prompt"], actual[3]["prompt"])
        for row in actual:
            self.assertEqual((row["model"], row["operation"], row["steps"], row["width"], row["height"], row["seed"]),
                             ("qwen-image-2.1", "image.generate", 40, 512, 512, 29))
            self.assertNotIn("loras", row)
        frozen = requests(Path("output"), 40, 4, "native-frozen", Path("existing-manifest.json"))
        self.assertTrue(all(row["qwen21_w8a8"] and row["execution"] == "gpu_ane" for row in frozen))

    def reference(self):
        return dict(runs=[dict(request_wall_including_export=10., actual_steps=list(range(40)),
                               actual_quantization=None, loras=[], prompt_cache_hit=index == 3) for index in range(4)])

    def test_actual_reference_steps_quantization_and_cache_not_relabelled(self):
        validate_runs(self.reference(), "mflux-bf16", 40, 4)
        for field, value in (("actual_steps", list(range(4))), ("actual_quantization", 4), ("loras", ["adapter"]),
                             ("prompt_cache_hit", True), ("request_wall_including_export", float("nan"))):
            report = self.reference()
            report["runs"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_runs(report, "mflux-bf16", 40, 4)

    def test_full_request_sequence_and_explicit_affine_recipes(self):
        report = self.reference()
        for row in report["runs"]:
            row["actual_quantization"] = 4
        validate_runs(report, "mflux-q4", 40, 4)
        with self.assertRaises(ValueError):
            validate_runs(report, "mflux-q8", 40, 4)
        with self.assertRaises(ValueError):
            validate_runs(dict(runs=report["runs"][:1]), "mflux-q4", 40, 4)

    def test_partial_float16_requires_real_parameters_and_generation_metadata(self):
        report = self.reference()
        report.update(compute_precision_name="float16", text_embedding_dtype="mlx.core.bfloat16",
                      transformer_floating_parameter_dtypes={"mlx.core.float16": 100, "mlx.core.bfloat16": 20})
        for row in report["runs"]:
            row["generation_parameters"] = dict(compute_precision="float16")
        validate_runs(report, "mflux-fp16", 40, 4)
        with self.assertRaisesRegex(ValueError, "BF16 control"):
            validate_runs(report, "mflux-bf16", 40, 4)
        altered = copy.deepcopy(report)
        altered["transformer_floating_parameter_dtypes"] = {"mlx.core.bfloat16": 120}
        with self.assertRaisesRegex(ValueError, "actual upstream partial float16"):
            validate_runs(altered, "mflux-fp16", 40, 4)
        report["runs"][0]["generation_parameters"] = {}
        with self.assertRaisesRegex(ValueError, "actual upstream partial float16"):
            validate_runs(report, "mflux-fp16", 40, 4)


if __name__ == "__main__":
    unittest.main()
