"""Original Lightricks post-upscale latent AdaIN versus PyTorch."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROBE = Path(os.environ.get("TURBOCIDER_LTX098_ADAIN_PROBE", "/nonexistent")).resolve()


@unittest.skipUnless(PROBE.is_file(), "native LTX AdaIN probe required")
class AdaINTests(unittest.TestCase):
    def test_bf16_statistics_and_operators(self):
        import torch
        from safetensors.torch import load_file, save_file

        generator = torch.Generator().manual_seed(100)
        with tempfile.TemporaryDirectory(prefix="tc-ltx098-adain-") as temporary:
            root = Path(temporary)
            for f, h, w in ((2, 1, 1), (4, 3, 5), (16, 10, 10)):
                with self.subTest(frames=f, height=h, width=w):
                    reference = torch.randn((1, f, h, w, 128), generator=generator).to(torch.bfloat16)
                    upsampled = torch.randn((1, f, h * 2, w * 2, 128), generator=generator).to(torch.bfloat16)
                    save_file({"reference": reference, "upsampled": upsampled}, str(root / "input.safetensors"))
                    run = subprocess.run([str(PROBE), str(root / "input.safetensors"), str(root / "output.safetensors")],
                                         capture_output=True, text=True, timeout=30)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    actual = load_file(str(root / "output.safetensors"))["value"].float()
                    ref_std, ref_mean = torch.std_mean(reference.float(), dim=(1, 2, 3), keepdim=True, correction=1)
                    up_std, up_mean = torch.std_mean(upsampled.float(), dim=(1, 2, 3), keepdim=True, correction=1)
                    expected = ((upsampled - up_mean.bfloat16()) / up_std.bfloat16()) * ref_std.bfloat16() + ref_mean.bfloat16()
                    delta = actual - expected.float()
                    self.assertLess((delta.norm() / expected.float().norm()).item(), .008)
                    self.assertTrue(torch.isfinite(actual).all())


if __name__ == "__main__":
    unittest.main()
