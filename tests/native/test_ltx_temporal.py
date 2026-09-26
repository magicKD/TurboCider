import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))
import evaluate_ltx_temporal as temporal


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        self.zero = np.zeros((8, 8, 2), np.float32)

    def metrics(self, a, b, c, d, forward=None, backward=None, candidate=None):
        return temporal.temporal_metrics(a, b, c, d,
            self.zero if forward is None else forward,
            self.zero if backward is None else backward,
            self.zero if candidate is None else candidate)

    def test_identity_temporal_error_zero(self):
        a, b = self.frame, np.flip(self.frame, axis=0).copy()
        result = self.metrics(a, b, a, b)
        self.assertEqual(result["residual_difference_mae_255"], 0.)
        self.assertEqual(result["flow_endpoint_difference_px"], 0.)

    def test_candidate_flicker_is_detected(self):
        a = np.zeros_like(self.frame)
        result = self.metrics(a, a, a, np.full_like(a, 20))
        self.assertEqual(result["residual_difference_mae_255"], 20.)
        self.assertEqual(result["residual_difference_rmse_255"], 20.)

    def test_constant_offset_not_temporal_quality(self):
        # A static brightness error is intentionally invisible: spatial metrics
        # must be reported alongside the temporal diagnostic.
        a = np.zeros_like(self.frame)
        c = np.full_like(a, 20)
        self.assertEqual(self.metrics(a, a, c, c)["residual_difference_mae_255"], 0.)

    def test_forward_warp_sign_and_border_mask(self):
        a = self.frame
        b = np.roll(a, 1, axis=1)
        forward = self.zero.copy()
        forward[..., 0] = 1
        result = self.metrics(a, b, a, b, forward, -forward, forward)
        self.assertEqual(result["reference_warp_mae_255"], 0.)
        self.assertEqual(result["valid_pixels"], 6 * 7)

    def test_candidate_flow_does_not_change_mask(self):
        candidate = self.zero.copy()
        candidate[..., 0] = 100
        result = self.metrics(*([self.frame] * 4), candidate=candidate)
        self.assertEqual(result["valid_pixels"], 49)
        self.assertEqual(result["flow_endpoint_difference_px"], 100.)

    def test_inconsistent_reference_flow_fails_closed(self):
        backward = self.zero.copy()
        backward[..., 0] = 2
        with self.assertRaisesRegex(ValueError, "no flow-consistent"):
            self.metrics(*([self.frame] * 4), backward=backward)

    def test_invalid_data_rejected(self):
        with self.assertRaises(ValueError):
            self.metrics(self.frame.astype(np.float32), *([self.frame] * 3))
        bad = self.zero.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.metrics(*([self.frame] * 4), forward=bad)

    def test_decode_identity_and_frame_count(self):
        class Process:
            def __init__(self, payload):
                self.stdout, self.stderr = io.BytesIO(payload), io.BytesIO()
            def poll(self):
                return 0
            def wait(self):
                return 0
        meta = dict(width=8, height=8, fps=24.)
        data = self.frame.tobytes() * 3
        for expected, should_raise in ((3, False), (4, True)):
            with patch.object(temporal, "_command", return_value="ffmpeg"), \
                 patch.object(temporal, "_probe", return_value=meta), \
                 patch.object(temporal, "_decode_process", side_effect=[Process(data), Process(data)]), \
                 patch.object(temporal, "flow", return_value=self.zero):
                if should_raise:
                    with self.assertRaisesRegex(ValueError, "decoded length"):
                        temporal.compare(Path("a"), Path("b"), expected, 24, 640)
                else:
                    result = temporal.compare(Path("a"), Path("b"), expected, 24, 640)
                    self.assertEqual(result["transitions"], 2)
                    self.assertEqual(result["summary"]["residual_difference_mae_255"]["weighted_mean"], 0.)
