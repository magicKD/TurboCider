"""CPU-only image contract tests; run separately from timed inference."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
SPEC = importlib.util.spec_from_file_location(
    "image_compare", ROOT / "tools/validation/runtime_ane_image_compare.py")
COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARE)


class ImageCompareTests(unittest.TestCase):
    def test_identical_nonconstant_and_constant_images(self):
        rng = np.random.default_rng(7)
        for value in (rng.integers(0, 256, (17, 19, 3), dtype=np.uint8),
                      np.full((11, 11, 3), 128, np.uint8)):
            report = COMPARE.rgb_metrics(value, value)
            self.assertTrue(report["pixel_exact"])
            self.assertEqual(report["rmse_u8"], 0)
            self.assertIsNone(report["psnr_db"])
            self.assertAlmostEqual(report["rgb_ssim_gaussian11_valid"], 1, places=10)

    def test_constant_brightness_change_uses_declared_ssim_luminance_term(self):
        a, b = (np.full((13, 15, 3), value, np.uint8) for value in (100, 120))
        report = COMPARE.rgb_metrics(a, b)
        c1 = (0.01 * 255) ** 2
        self.assertAlmostEqual(report["rgb_ssim_gaussian11_valid"],
                               (2 * 100 * 120 + c1) / (100 ** 2 + 120 ** 2 + c1), places=9)
        self.assertEqual(report["rmse_u8"], 20)
        self.assertIsNone(report["rgb_correlation"])

    def test_geometry_dtype_and_small_images_are_rejected(self):
        a = np.zeros((17, 19, 3), np.uint8)
        for candidate in (a[:10], a[:, :11], a[:, :, :1], a.astype(np.float32)):
            with self.assertRaises(ValueError):
                COMPARE.rgb_metrics(a, candidate)
        with self.assertRaises(ValueError):
            COMPARE.rgb_metrics(a[:10], a[:10])

    def test_png_alpha_is_reported_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = np.full((13, 13, 4), 128, np.uint8)
            b = a.copy()
            b[:, :, 3] = 0
            Image.fromarray(a).save(root / "a.png")
            Image.fromarray(b).save(root / "b.png")
            report = COMPARE.compare_png(root / "a.png", root / "b.png")
            self.assertTrue(report["metrics"]["pixel_exact"])
            self.assertFalse(report["metrics"]["alpha"]["pixel_exact"])
            self.assertFalse(report["png_pixel_exact"])
            self.assertEqual(report["metrics"]["alpha"]["max_abs_u8"], 128)
            self.assertFalse(report["qualification_passed"])


if __name__ == "__main__":
    unittest.main()
