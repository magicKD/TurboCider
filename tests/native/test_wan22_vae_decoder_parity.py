"""Optional full-weight, multi-frame Wan2.2 VAE decoder parity fixture."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


def configured_path(name):
    value = os.environ.get(name)
    return Path(value).resolve() if value else None


PROBE = configured_path("TURBOCIDER_WAN22_VAE_DECODER_PROBE")
CHECKPOINT = configured_path("TURBOCIDER_WAN22_VAE_CHECKPOINT")
CAPTURE = configured_path("TURBOCIDER_WAN22_VAE_DECODE_CAPTURE")


@unittest.skipUnless(all(path is not None and path.is_file() for path in (PROBE, CHECKPOINT, CAPTURE)),
                     "native VAE probe, full checkpoint and ComfyUI decode capture required")
class Wan22VAEDecoderParityTests(unittest.TestCase):
    def test_video_against_comfyui(self):
        import torch
        from safetensors.torch import load_file

        fixture = load_file(str(CAPTURE))
        latents, expected = fixture["latents"], fixture["expected"].float()
        self.assertEqual(tuple(latents.shape[:2]), (1, 48))
        self.assertEqual(expected.shape[2], 4 * latents.shape[2] - 3)
        with tempfile.TemporaryDirectory(prefix="tc-wan22-decode-") as temporary:
            output = Path(temporary) / "native.safetensors"
            result = subprocess.run([str(PROBE), str(CHECKPOINT), str(CAPTURE), str(output)],
                                    capture_output=True, text=True, timeout=900)
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = load_file(str(output))["output"].float()
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(torch.isfinite(expected).all())
        relative_l2 = (actual - expected).norm() / expected.norm().clamp_min(1e-30)
        self.assertLess(relative_l2.item(), .02)
        # Check both the clean first frame and the final chunk, where stale
        # or incorrectly reset temporal history would otherwise go unnoticed.
        for frame in (0, actual.shape[2] - 1):
            reference = expected[:, :, frame]
            error = (actual[:, :, frame] - reference).norm() / reference.norm().clamp_min(1e-30)
            self.assertLess(error.item(), .03, f"frame {frame}")


if __name__ == "__main__":
    unittest.main()
