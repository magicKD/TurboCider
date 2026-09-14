import json
import sys
import tempfile
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/native"))
from evaluate_ltx_lpips import report_pairs, rgb_tensor, validate_pair


class LPIPSInputTests(unittest.TestCase):
    def test_rgb_order_range_and_layout(self):
        value = rgb_tensor(bytes([255, 0, 0, 0, 255, 255]), 2, 1)
        self.assertEqual(tuple(value.shape), (1, 3, 1, 2))
        torch.testing.assert_close(value, torch.tensor([[[[1., -1.]], [[-1., 1.]], [[-1., 1.]]]]))
        with self.assertRaises(ValueError):
            rgb_tensor(bytes(5), 2, 1)

    def test_pair_requires_known_matching_geometry_and_fps(self):
        meta = dict(width=768, height=448, fps=24.)
        validate_pair(meta, meta)
        for change in (dict(width=1280), dict(fps=0), dict(fps=25), dict(fps=float('nan'))):
            with self.assertRaises(ValueError):
                validate_pair(meta, {**meta, **change})

    def test_vae_tiling_report_requires_exact_isolation(self):
        report = {
            "schema": "ltx-vae-tiling-quality-v1", "complete": True,
            "comparisons": [{
                "run": 0, "videos": ["untiled.mp4", "tiled.mp4"],
                "video_sha256": ["left", "right"],
                "tensor_checks": {"stage2_video": {"byte_exact": True}},
                "metrics": {"frame_count_compared": 97,
                            "reference": {"width": 704, "height": 448, "fps": 24.0}},
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps(report))
            pairs = report_pairs(path)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0]["expected_frames"], 97)
            self.assertEqual(pairs[0]["workload"]["operation"], "vae_decode")
            self.assertEqual(pairs[0]["candidate"]["variant"], "spatial_tiling")
            report["comparisons"][0]["tensor_checks"]["stage2_video"]["byte_exact"] = False
            path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                report_pairs(path)
