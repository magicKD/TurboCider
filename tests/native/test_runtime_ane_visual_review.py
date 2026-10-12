"""Visual evidence preserves originals; it never automatically qualifies them."""
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
    "visual_review", ROOT / "tools/validation/runtime_ane_visual_review.py")
REVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REVIEW)


class VisualReviewTests(unittest.TestCase):
    def test_whole_and_shared_coordinate_crops_preserve_rgb_and_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for channels in (3, 4):
                rng = np.random.default_rng(channels)
                a = rng.integers(0, 256, (280, 300, channels), dtype=np.uint8)
                b = rng.integers(0, 256, a.shape, dtype=np.uint8)
                paths = [root / f"a{channels}.png", root / f"b{channels}.png"]
                for path, values in zip(paths, (a, b)):
                    Image.fromarray(values).save(path)
                original = [path.read_bytes() for path in paths]
                output = root / f"review{channels}"
                report = REVIEW.prepare_review(*paths, output)
                self.assertFalse(report["qualification_passed"])
                self.assertEqual(report["review"]["status"], "pending")
                self.assertIsNone(report["review"]["visually_very_similar"])
                self.assertEqual(len(report["sheets"]), 4)
                for sheet in report["sheets"]:
                    x, y, right, bottom = sheet["source_crop_xyxy"]
                    with Image.open(output / sheet["file"]) as source:
                        image = np.array(source)
                    for origin, values in zip((sheet["left_origin"], sheet["right_origin"]), (a, b)):
                        ox, oy = origin
                        np.testing.assert_array_equal(image[oy:oy+bottom-y, ox:ox+right-x], values[y:bottom, x:right])
                    self.assertEqual(sheet["scale"], 1)
                self.assertEqual([path.read_bytes() for path in paths], original)
                with self.assertRaisesRegex(ValueError, "already exists"):
                    REVIEW.prepare_review(*paths, output)

    def test_geometry_mode_format_and_mutation_rejection(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, candidate = root / "a.png", root / "b.png"
            Image.new("RGB", (20, 20)).save(reference)
            for mode, size in (("RGB", (21, 20)), ("RGBA", (20, 20)), ("L", (20, 20)), ("RGB", (10, 10))):
                Image.new(mode, size).save(candidate)
                with self.assertRaises(ValueError):
                    REVIEW.prepare_review(reference, candidate, root / "invalid")
                self.assertFalse((root / "invalid").exists())
            Image.new("RGB", (20, 20)).save(candidate, format="JPEG")
            with self.assertRaisesRegex(ValueError, "PNG originals"):
                REVIEW.prepare_review(reference, candidate, root / "invalid")
            Image.new("RGB", (20, 20)).save(candidate)
            with patch.object(REVIEW, "sha256_file", side_effect=["A", "B", "changed", "B"]):
                with self.assertRaisesRegex(ValueError, "changed"):
                    REVIEW.prepare_review(reference, candidate, root / "mutated")
            self.assertFalse((root / "mutated").exists())


if __name__ == "__main__":
    unittest.main()
