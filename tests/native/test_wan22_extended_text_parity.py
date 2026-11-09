"""Optional real-weight extended UMT5 conditioning against ComfyUI."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


def configured(name):
    value = os.environ.get(name)
    return Path(value).resolve() if value else None


PROBE = configured("TURBOCIDER_WAN22_TEXT_PROBE")
CHECKPOINT = configured("TURBOCIDER_UMT5_CHECKPOINT")
TOKENIZER = configured("TURBOCIDER_UMT5_TOKENIZER_ROOT")
CAPTURE = configured("TURBOCIDER_WAN22_TEXT_CAPTURE")


@unittest.skipUnless(PROBE and PROBE.is_file() and CHECKPOINT and CHECKPOINT.is_file() and
                     TOKENIZER and TOKENIZER.is_dir() and CAPTURE and CAPTURE.is_file(),
                     "native probe, UMT5 checkpoint/tokenizer and Comfy capture required")
class Wan22ExtendedTextParityTests(unittest.TestCase):
    def test_extended_real_conditioning(self):
        import torch
        from safetensors.torch import load_file

        fixture = load_file(str(CAPTURE))
        expected = fixture["expected"].float()
        self.assertGreater(fixture["ids"].numel(), 512)
        self.assertLessEqual(fixture["ids"].numel(), 1024)
        with tempfile.TemporaryDirectory(prefix="tc-wan22-long-text-") as temporary:
            output = Path(temporary) / "native.safetensors"
            result = subprocess.run([str(PROBE), str(CHECKPOINT), str(TOKENIZER), str(CAPTURE), str(output)],
                                    capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = load_file(str(output))["output"].float()
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.isfinite(actual).all())
        relative = (actual - expected).norm() / expected.norm().clamp_min(1e-30)
        self.assertLess(relative.item(), .03)


if __name__ == "__main__":
    unittest.main()
