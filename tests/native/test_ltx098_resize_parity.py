"""LTX multiscale pixel resize versus PyTorch bilinear interpolation."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROBE = Path(os.environ.get("TURBOCIDER_LTX098_RESIZE_PROBE", "/nonexistent")).resolve()


@unittest.skipUnless(PROBE.is_file(), "native LTX resize probe required")
class ResizeTests(unittest.TestCase):
    def test_bilinear_sizes_and_dtypes(self):
        import torch
        import torch.nn.functional as F
        from safetensors.torch import load_file, save_file

        torch.manual_seed(42)
        with tempfile.TemporaryDirectory(prefix="tc-ltx098-resize-") as temporary:
            root = Path(temporary)
            for shape, output in [((1, 2, 7, 9, 3), (13, 5)), ((1, 2, 20, 20, 3), (16, 16)),
                                  ((1, 1, 4, 6, 3), (4, 6)), ((1, 1, 1, 1, 3), (3, 5))]:
                for dtype in (torch.float32, torch.bfloat16):
                    with self.subTest(shape=shape, output=output, dtype=dtype):
                        pixels = torch.randn(shape, dtype=dtype)
                        save_file({"pixels": pixels}, str(root / "input.safetensors"))
                        run = subprocess.run([str(PROBE), str(root / "input.safetensors"), str(root / "output.safetensors"),
                                              str(output[0]), str(output[1])], capture_output=True, text=True, timeout=30)
                        self.assertEqual(run.returncode, 0, run.stderr)
                        actual = load_file(str(root / "output.safetensors"))["pixels"]
                        expected = F.interpolate(pixels.reshape(-1, shape[2], shape[3], 3).permute(0, 3, 1, 2),
                                                 size=output, mode="bilinear", align_corners=False)
                        expected = expected.permute(0, 2, 3, 1).reshape(1, shape[1], *output, 3)
                        self.assertEqual(actual.shape, expected.shape)
                        error = (actual.float() - expected.float()).abs().max().item()
                        self.assertTrue(torch.allclose(actual.float(), expected.float(), atol=.016 if dtype == torch.bfloat16 else 4e-6, rtol=0),
                                        f"maximum absolute difference: {error}")


if __name__ == "__main__":
    unittest.main()
