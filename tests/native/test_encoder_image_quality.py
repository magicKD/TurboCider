#!/usr/bin/env python3
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))
from analyze_z_image_encoder_streaming import compare_arrays, quality_passes, read_tensor

THRESHOLDS = {"rgb_correlation_min": .99, "rgb_cosine_min": .995, "rgb_mae_max": 2.55,
              "final_latent_relative_l2_max": .05, "final_latent_cosine_min": .995}


class EncoderImageQualityTests(unittest.TestCase):
    def test_nonfinite_empty_and_shape_mismatch_cannot_pass(self):
        for left, right in [([1], [np.nan]), ([1], [np.inf]), ([], []), ([[1, 2]], [1, 2])]:
            metric = compare_arrays(left, right)
            self.assertFalse(metric["valid"])
            self.assertFalse(quality_passes(metric, metric, metric, THRESHOLDS))

    def test_latent_error_is_not_hidden_by_perfect_cosine_or_rgb(self):
        x = np.arange(1, 10)
        identical = compare_arrays(x, x)
        good = compare_arrays(x, x * 1.01)
        bad = compare_arrays(x, x * 1.1)
        self.assertAlmostEqual(bad["cosine"], 1.)
        self.assertAlmostEqual(bad["relative_l2"], .1)
        self.assertTrue(quality_passes(identical, identical, good, THRESHOLDS))
        self.assertFalse(quality_passes(identical, identical, bad, THRESHOLDS))

    def test_initial_noise_mismatch_and_zero_reference_are_explicit(self):
        exact = compare_arrays([1, 2, 3], [1, 2, 3])
        changed = compare_arrays([1, 2, 3], [1, 2, 3.0001])
        self.assertFalse(quality_passes(exact, changed, exact, THRESHOLDS))
        zero = compare_arrays([0, 0], [1, 1])
        self.assertIsNone(zero["relative_l2"])
        self.assertFalse(quality_passes(exact, exact, zero, THRESHOLDS))
        self.assertEqual(compare_arrays([0, 0], [0, 0])["relative_l2"], 0)

    def test_bfloat16_dump_decode_and_truncated_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tensor.safetensors"
            header = json.dumps({"tensor": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}).encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<HH", 0x3f80, 0xc000))
            np.testing.assert_array_equal(read_tensor(path), [1., -2.])
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "offsets"):
                read_tensor(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
