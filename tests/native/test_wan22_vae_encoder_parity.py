"""Optional full-weight first-frame Wan2.2 VAE encoder parity fixture."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


def configured_path(name):
    value = os.environ.get(name)
    return Path(value).resolve() if value else None


PROBE = configured_path("TURBOCIDER_WAN22_VAE_ENCODER_PROBE")
CHECKPOINT = configured_path("TURBOCIDER_WAN22_VAE_CHECKPOINT")
CAPTURE = configured_path("TURBOCIDER_WAN22_VAE_CAPTURE")


@unittest.skipUnless(all(path is not None and path.is_file() for path in (PROBE, CHECKPOINT, CAPTURE)),
                     "native VAE probe, full checkpoint and ComfyUI capture required")
class Wan22VAEEncoderParityTests(unittest.TestCase):
    def test_first_frame_against_comfyui(self):
        import torch
        from safetensors.torch import load_file

        fixture = load_file(str(CAPTURE))
        pixels, expected = fixture["pixels"], fixture["expected"].float()
        self.assertEqual(tuple(pixels.shape[:3]), (1, 3, 1))
        self.assertEqual(tuple(expected.shape[:3]), (1, 48, 1))
        with tempfile.TemporaryDirectory(prefix="tc-wan22-vae-") as temporary:
            output = Path(temporary) / "native.safetensors"
            result = subprocess.run([str(PROBE), str(CHECKPOINT), str(CAPTURE), str(output)],
                                    capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = load_file(str(output))["output"].float()
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(torch.isfinite(expected).all())
        relative_l2 = (actual - expected).norm() / expected.norm()
        cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
        self.assertLess(relative_l2.item(), .02)
        self.assertGreater(cosine.item(), .9998)


if __name__ == "__main__":
    unittest.main()
