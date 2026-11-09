"""First-frame mask and Euler update parity with ComfyUI's inpaint math."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROBE = Path(os.environ.get("TURBOCIDER_LTX098_SCHEDULER_PROBE", "/nonexistent")).resolve()


@unittest.skipUnless(PROBE.is_file(), "native LTX scheduler probe required")
class SchedulerTests(unittest.TestCase):
    def test_strength_and_euler_match_reference(self):
        import torch
        from safetensors.torch import load_file, save_file
        torch.manual_seed(17)
        state = torch.randn((1, 3, 2, 2, 8), dtype=torch.float32)
        first = torch.randn((1, 1, 2, 2, 8), dtype=torch.float32)
        velocity = torch.randn(state.shape, dtype=torch.float32)
        with tempfile.TemporaryDirectory(prefix="tc-ltx098-scheduler-") as temporary:
            root = Path(temporary)
            save_file({"state": state, "first": first, "velocity": velocity}, str(root / "input.safetensors"))
            for strength in (0., .25, .75, 1.):
                for sigma, next_sigma in ((1., .9937), (.725, .4219), (.4219, 0.)):
                    with self.subTest(strength=strength, sigma=sigma):
                        mask = torch.ones((1, 3, 1, 1, 1), dtype=torch.float32)
                        mask[:, :1] = 1 - strength
                        latent = torch.cat((first, torch.zeros_like(state[:, 1:])), dim=1)
                        model_input = state * mask + latent * (1 - mask)
                        denoised = (model_input - velocity * sigma) * mask + latent * (1 - mask)
                        expected = state + ((state - denoised) / sigma) * (next_sigma - sigma)
                        run = subprocess.run([str(PROBE), str(root / "input.safetensors"), str(root / "output.safetensors"),
                                              str(strength), str(sigma), str(next_sigma)], capture_output=True, text=True, timeout=30)
                        self.assertEqual(run.returncode, 0, run.stderr)
                        actual = load_file(str(root / "output.safetensors"))
                        self.assertTrue(torch.allclose(actual["model_input"], model_input, rtol=1e-6, atol=1e-6))
                        self.assertTrue(torch.allclose(actual["output"], expected, rtol=1e-6, atol=1e-6))
                        cap = 1.0 - strength
                        original_expected = state + velocity * (next_sigma - sigma)
                        original_expected[:, :1] = state[:, :1] + velocity[:, :1] * (min(next_sigma, cap) - min(sigma, cap))
                        self.assertTrue(torch.allclose(actual["original_input"], model_input, rtol=1e-6, atol=1e-6))
                        self.assertTrue(torch.allclose(actual["original_output"], original_expected, rtol=1e-6, atol=1e-6))
                        self.assertAlmostEqual(float(actual["original_time_strength"]), 1 - min(sigma, cap) / sigma, places=6)


if __name__ == "__main__":
    unittest.main()
