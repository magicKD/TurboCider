import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "z_image_w8a8_layers", ROOT / "tools/validation/z_image_w8a8_layers.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ZImageA8LayersTests(unittest.TestCase):
    def test_zero_quantization_error_for_exactly_representable_values(self):
        x = np.array([[1, -2], [2, 1]], dtype=np.float32)
        gate = up = down = np.eye(2, dtype=np.float32)
        metrics = MODULE.measure_a8(x, gate, up, down, np.ones(2, dtype=np.float32),
                                    1, 1 / 1024, 1, np)
        self.assertEqual(metrics["input_a8_squared_error"], 0)
        self.assertGreater(metrics["reference_energy"], 0)

    def test_separate_input_and_hidden_ablation(self):
        x = np.array([[0.41, -0.83], [0.89, 0.13]], dtype=np.float32)
        gate = up = down = np.eye(2, dtype=np.float32)
        base = dict(x=x, gate=gate, up=up, down=down,
                    s1=np.ones(2, dtype=np.float32), activation_scale=1, np=np)
        input_loss = MODULE.measure_a8(input_scale=0.5, hidden_scale=1e-6, **base)
        hidden_loss = MODULE.measure_a8(input_scale=0.01, hidden_scale=0.0001, **base)
        self.assertGreater(input_loss["input_a8_squared_error"], 0)
        self.assertGreater(hidden_loss["hidden_a8_squared_error"], 0)
        self.assertGreater(hidden_loss["hidden_clip_fraction"], 0)
        np.testing.assert_array_equal(MODULE.select_rows(x, 2, np), x)
        with self.assertRaisesRegex(ValueError, "scale"):
            MODULE.quantize_dequantize(x, 0, np)

    def test_grouped_hidden_quantization_reduces_outlier_interference(self):
        x = np.array([[.5, .7], [.8, .3]], dtype=np.float32)
        gate = np.eye(2, dtype=np.float32)
        up = np.diag([1., 100.]).astype(np.float32)
        down = np.eye(2, dtype=np.float32)
        args = (x, gate, up, down, np.ones(2, dtype=np.float32), .001, .3, 1., np)
        metrics = MODULE.measure_a8(*args, channel_scales=[.003, .3])
        self.assertLess(metrics["channel_group_a8_squared_error"],
                        metrics["combined_a8_squared_error"])
        with self.assertRaisesRegex(ValueError, "channel-group scales"):
            MODULE.measure_a8(*args, channel_scales=[.003, .3, .3])

    def test_dynamic_row_hidden_a8_is_diagnostic_and_preserves_small_rows(self):
        hidden = np.array([[.7, .2], [70., 20.]], dtype=np.float32)
        down = np.eye(2, dtype=np.float32)
        static = MODULE.quantize_dequantize(hidden, 70 / 127, np)
        dynamic = MODULE.dynamic_hidden_a8(hidden, down, np)
        self.assertLess(np.linalg.norm(dynamic - hidden),
                        np.linalg.norm(static - hidden))
        self.assertGreater(dynamic[0, 1], 0)
        with self.assertRaisesRegex(ValueError, "geometry"):
            MODULE.dynamic_hidden_a8(hidden, down[:1, :1], np)

        x = np.array([[.7, .2], [70., 20.]], dtype=np.float32)
        metrics = MODULE.measure_a8(x, down, down, down, np.ones(2), .01,
                                    1., 1., np, dynamic_hidden=True)
        self.assertIn("dynamic_hidden_a8_squared_error", metrics)

    def test_static_hidden_row_group_scales_use_training_rows_only(self):
        eye = np.eye(2, dtype=np.float32)
        training = [np.array([[.5, .2], [.7, .3], [10., 2.], [11., 3.]],
                             dtype=np.float32)]
        scales = MODULE.calibrate_static_hidden_row_groups(
            eye, eye, eye, np.ones(2), training, .01, 1, 4, 2, np,
            rows_per_group=2)
        output_aware = MODULE.calibrate_static_hidden_row_groups(
            eye, eye, eye, np.ones(2), training, .01, 1, 4, 2, np,
            rows_per_group=2, output_aware=True)
        self.assertEqual(scales.shape, (2,))
        self.assertEqual(output_aware.shape, (2,))
        self.assertTrue(np.isfinite(output_aware).all())
        self.assertGreater(scales[1], scales[0])
        heldout = training[0] * 2
        first = MODULE.measure_a8(heldout, eye, eye, eye, np.ones(2),
                                  .01, 1, 1, np,
                                  static_row_scales=scales[[0, 0, 1, 1], None])
        self.assertIn("static_row_hidden_a8_squared_error", first)
        with self.assertRaisesRegex(ValueError, "calibration"):
            MODULE.calibrate_static_hidden_row_groups(
                eye, eye, eye, np.ones(2), training, .01, 1, 4, 3, np)

    def test_adaptive_hidden_scale_bank_uses_training_samples(self):
        eye = np.eye(2, dtype=np.float32)
        training = [np.array([[.5, .2], [.7, .3], [10., 2.], [11., 3.]],
                             dtype=np.float32)]
        scales = MODULE.calibrate_adaptive_hidden_a8(
            eye, eye, np.ones(2), training, .01, 1, 4, 3, np,
            rows_per_sample=4)
        self.assertGreaterEqual(len(scales), 2)
        self.assertTrue(np.all(np.diff(scales) > 0))
        hidden = np.array([[.7, .2], [70., 20.]], dtype=np.float32)
        candidate = MODULE.adaptive_hidden_a8(hidden, eye, scales, np)
        self.assertEqual(candidate.shape, hidden.shape)
        with self.assertRaisesRegex(ValueError, "geometry or scales"):
            MODULE.adaptive_hidden_a8(hidden, eye, scales[::-1], np)


if __name__ == "__main__":
    unittest.main()
