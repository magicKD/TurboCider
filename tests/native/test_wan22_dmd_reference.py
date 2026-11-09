"""Scheduler contract shared by the ComfyUI and native Wan2.2 work."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from tools.validation.wan22_schedule import warped_sigmas


class Wan22DMDReferenceTests(unittest.TestCase):
    def test_fastwan_warped_schedule(self):
        sigmas = warped_sigmas()
        self.assertEqual(len(sigmas), 3)
        self.assertAlmostEqual(sigmas[0], 1.0)
        self.assertAlmostEqual(sigmas[1], 0.9396722939424031)
        self.assertAlmostEqual(sigmas[2], 0.8452072538860105)
        self.assertGreater(sigmas[0], sigmas[1])
        self.assertGreater(sigmas[1], sigmas[2])
        for invalid in ((1000, 1000, 522), (1001, 757, 522), (1000, 0, 1)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                warped_sigmas(invalid)

    def test_comfy_sampler_matches_explicit_renoising(self):
        import torch

        # The sampler math has no ComfyUI dependency; supply an isolated stub
        # just for its node registration import.
        original_comfy = sys.modules.get("comfy")
        original_samplers = sys.modules.get("comfy.samplers")
        comfy = types.ModuleType("comfy")
        samplers = types.ModuleType("comfy.samplers")
        comfy.samplers = samplers
        sys.modules["comfy"] = comfy
        sys.modules["comfy.samplers"] = samplers
        try:
            path = Path(__file__).resolve().parents[2] / "tools/validation/wan22_comfy_dmd_node.py"
            spec = importlib.util.spec_from_file_location("tc_wan22_dmd_test_node", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            seed = 42
            generator = torch.Generator(device="cpu").manual_seed(seed)
            initial = torch.randn((1, 2, 3), generator=generator)
            fresh1 = torch.randn(initial.shape, generator=generator)
            fresh2 = torch.randn(initial.shape, generator=generator)
            sigmas = torch.tensor([*warped_sigmas(), 0.0])
            seen = []
            def model(x, sigma, **kwargs):
                seen.append(float(sigma[0]))
                return x * 0.25 + sigma.reshape(-1, 1, 1) * 0.1
            actual = module.sample_dmd(model, initial, sigmas, {"seed": seed})
            first = initial * 0.25 + sigmas[0] * 0.1
            state = (1 - sigmas[1]) * first + sigmas[1] * fresh1
            second = state * 0.25 + sigmas[1] * 0.1
            state = (1 - sigmas[2]) * second + sigmas[2] * fresh2
            expected = state * 0.25 + sigmas[2] * 0.1
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(len(seen), 3)
        finally:
            if original_comfy is None: sys.modules.pop("comfy", None)
            else: sys.modules["comfy"] = original_comfy
            if original_samplers is None: sys.modules.pop("comfy.samplers", None)
            else: sys.modules["comfy.samplers"] = original_samplers


if __name__ == "__main__":
    unittest.main()
