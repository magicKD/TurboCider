import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from z_base_formats_screen import cases, environment, request, validate_rows, PUBLIC, PUBLIC_CONVROT


class ZBaseFormatsTests(unittest.TestCase):
    def test_nvfp4_has_only_gpu_not_a_fake_hybrid(self):
        self.assertEqual([name for name, case in cases().items() if case["format"] == "nvfp4"], ["nvfp4-gpu"])

    def test_original_convrot_basis_requires_its_own_public_template(self):
        case = cases()["convrot-public"]
        task = request(case, "fox", Path("/tmp/no-write.png"), 8)
        self.assertEqual(task["ane_manifest"], str(PUBLIC_CONVROT))
        self.assertNotEqual(task["ane_manifest"], str(PUBLIC))
        self.assertEqual(environment(case)["TURBOCIDER_Z_RUNTIME_CONVROT"], "1")

    def test_gpu_recipes_cannot_leak_into_runtime_recipe(self):
        fast = environment(cases()["q4-fast"])
        mixed = environment(cases()["q4-private"])
        self.assertEqual(fast["TURBOCIDER_Z_GGUF_COMPUTE"], "qmm_f16_down64")
        self.assertEqual(fast["TURBOCIDER_Z_GGUF_COMPILE_PACKED"], "1")
        self.assertNotIn("TURBOCIDER_Z_GGUF_COMPUTE", mixed)
        self.assertNotIn("TURBOCIDER_Z_GGUF_COMPILE_PACKED", mixed)
        self.assertEqual(mixed["TURBOCIDER_Z_GGUF_RETAIN_PACKED"], "1")
        self.assertNotIn("TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN", mixed)

    def test_private_convrot_uses_measured_bf16_partial_not_f32_mpp(self):
        env = environment(cases()["convrot-private"])
        self.assertEqual(env["TURBOCIDER_PRIVATE_ANE_DATA_PATH"], "convrot_w8a8")
        self.assertEqual(env["TURBOCIDER_Z_CONVROT_BF16_PARTIAL"], "1")
        self.assertEqual(env["TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN"], "1")
        self.assertNotIn("TURBOCIDER_Z_CONVROT_FP32_MPP", env)

    def test_actual_gpu_source_and_prompt_cache_not_just_labels(self):
        case = cases()["nvfp4-gpu"]
        rows = [dict(model="z-image-turbo", operation="image.generate", width=512, height=512,
                     seed=42, steps=8, prompt_cache_hit=index == 3,
                     runtime_backend="mlx_cpp_metal_nvfp4_w4a16", runtime_precision="nvfp4_weights_bf16_activations",
                     timings_seconds=dict(request_wall=2., denoise=1., vae_decode=.2), hybrid=None) for index in range(4)]
        validate_rows(rows, case, 8)
        for fault in (lambda row: row.update(runtime_backend="mlx_cpp_metal"),
                      lambda row: row.update(prompt_cache_hit=True),
                      lambda row: row.update(lora_applied_projections=238)):
            broken = copy.deepcopy(rows); fault(broken[0])
            with self.assertRaises(ValueError):
                validate_rows(broken, case, 8)


if __name__ == "__main__":
    unittest.main()
