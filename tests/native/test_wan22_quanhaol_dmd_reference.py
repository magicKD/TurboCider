"""Exercise the diagnostic four-step Comfy sampler without a Comfy server."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch


class Wan22QuanhaolDMDReferenceTests(unittest.TestCase):
    def test_four_steps_and_renoising(self):
        original_comfy, original_samplers = sys.modules.get("comfy"), sys.modules.get("comfy.samplers")
        comfy = types.ModuleType("comfy")
        comfy.samplers = types.ModuleType("comfy.samplers")
        sys.modules["comfy"], sys.modules["comfy.samplers"] = comfy, comfy.samplers
        try:
            path = Path(__file__).resolve().parents[2] / "tools/validation/wan22_comfy_dmd_node.py"
            spec = importlib.util.spec_from_file_location("tc_quanhaol_sampler_test", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            seed = 42
            generator = torch.Generator(device="cpu").manual_seed(seed)
            initial = torch.randn((1, 2, 3), generator=generator)
            fresh = [torch.randn(initial.shape, generator=generator).to(torch.bfloat16) for _ in range(3)]
            sigmas = torch.tensor([1., .96, 8 / 9, 8 / 11, 0.])
            model_calls, callbacks = [], []

            def model(x, sigma, **kwargs):
                model_calls.append(float(sigma[0]))
                return x - sigma.reshape(-1, 1, 1) * .125

            result = module.sample_quanhaol_dmd(model, initial, sigmas, {"seed": seed}, callbacks.append)
            expected = initial.to(torch.bfloat16)
            for step in range(4):
                clean = (expected.float() - sigmas[step] * .125).to(torch.bfloat16)
                if step < 3:
                    sigma = sigmas[step + 1].to(torch.bfloat16)
                    expected = (1. - sigma) * clean + sigma * fresh[step]
                else:
                    expected = clean
            self.assertTrue(torch.equal(result, expected))
            for actual, expected_t in zip(model_calls, (1., .96, .888, .728)):
                self.assertAlmostEqual(actual, expected_t, places=6)
            self.assertEqual([entry["i"] for entry in callbacks], [0, 1, 2, 3])
            with self.assertRaises(ValueError):
                module.sample_quanhaol_dmd(model, initial, torch.tensor([1., .9, .8, 0.]), {"seed": seed})
        finally:
            if original_comfy is None: sys.modules.pop("comfy", None)
            else: sys.modules["comfy"] = original_comfy
            if original_samplers is None: sys.modules.pop("comfy.samplers", None)
            else: sys.modules["comfy.samplers"] = original_samplers


if __name__ == "__main__":
    unittest.main()
