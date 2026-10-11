import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from qwen21_mflux_framework_evidence import audit, compare_rgba, rgba


class QwenFrameworkEvidenceTests(unittest.TestCase):
    def test_generated_alpha_is_retained_not_silently_dropped(self):
        a = np.full((512, 512, 4), 100, dtype=np.uint8)
        b = a.copy()
        a[:, :, 3] = 253
        b[:, :, 3] = 255
        result = compare_rgba(a, b)
        self.assertIs(result["raw_rgb_metrics"]["pixel_exact"], True)
        self.assertIs(result["rgba_pixel_exact"], False)
        self.assertEqual(result["alpha"]["mae_255"], 2)
        self.assertEqual(result["alpha"]["reference_nonopaque_pixels"], 512 * 512)
        self.assertIs(result["white_composited_rgb_metrics"]["pixel_exact"], False)

    def test_rgb_implicit_alpha_is_explicit_and_no_resize(self):
        with tempfile.TemporaryDirectory(prefix="tc-qwen-reference-png-") as raw:
            path = Path(raw) / "image.png"
            Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8)).save(path)
            self.assertTrue(np.all(rgba(path)[:, :, 3] == 255))
            Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(path)
            with self.assertRaises(ValueError):
                rgba(path)

    def test_running_screen_is_never_promoted(self):
        with tempfile.TemporaryDirectory(prefix="tc-qwen-evidence-") as raw:
            root = Path(raw)
            (root / "summary.json").write_text(json.dumps(dict(status="running", trials=[], order=[])))
            with self.assertRaises(ValueError):
                audit(root, root, root / "missing-build.json")


if __name__ == "__main__":
    unittest.main()
