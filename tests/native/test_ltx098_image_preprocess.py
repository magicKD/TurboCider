"""LTX image crop/resize parity for opaque RGB images, without model weights."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


PROBE = Path(os.environ.get("TURBOCIDER_LTX098_IMAGE_PROBE", "/nonexistent")).resolve()


@unittest.skipUnless(PROBE.is_file(), "native LTX image probe required")
class ImagePreprocessTests(unittest.TestCase):
    def test_comfy_center_crop_and_half_pixel_bilinear(self):
        import numpy as np
        from PIL import Image
        from safetensors.torch import load_file
        import torch
        import torch.nn.functional as F

        rng = np.random.default_rng(941)
        # Includes landscape/portrait, enlargement, downsampling, non-square
        # output and odd dimensions where crop rounding is observable.
        cases = [(517, 261, 128, 128), (261, 517, 128, 128),
                 (79, 51, 128, 96), (63, 111, 64, 96), (127, 127, 127, 127)]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for index, (w, h, width, height) in enumerate(cases):
                with self.subTest(source=(w, h), target=(width, height)):
                    source = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
                    image, output = directory / f"{index}.png", directory / f"{index}.safetensors"
                    Image.fromarray(source, "RGB").save(image)
                    process = subprocess.run([str(PROBE), str(image), str(width), str(height), str(output)],
                                             capture_output=True, text=True, timeout=60)
                    self.assertEqual(process.returncode, 0, process.stderr)
                    value = load_file(str(output))["pixels"]
                    old_aspect, new_aspect = w / h, width / height
                    x = round((w - w * (new_aspect / old_aspect)) / 2) if old_aspect > new_aspect else 0
                    y = round((h - h * (old_aspect / new_aspect)) / 2) if old_aspect < new_aspect else 0
                    source_tensor = torch.from_numpy(source.copy()).float().permute(2, 0, 1)[None] / 255.
                    cropped = source_tensor[:, :, y:h-y, x:w-x]
                    expected = F.interpolate(cropped, size=(height, width), mode="bilinear", align_corners=False)
                    expected = expected.permute(0, 2, 3, 1) * 2 - 1
                    self.assertEqual(tuple(value.shape), tuple(expected.shape))
                    self.assertLess((value - expected).abs().max().item(), 3e-5)


if __name__ == "__main__":
    unittest.main()
