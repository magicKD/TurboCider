#!/usr/bin/env python3
"""Numerical comparison contracts used by the H3 audio and video VAE probes."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))

import h3_mlx_audio_vae_parity as audio
import h3_mlx_video_vae_parity as video

OWNERS = (audio, video)


class H3VaeMetricsTests(unittest.TestCase):
    def assert_metrics(self, left, right, expected):
        for owner in OWNERS:
            with self.subTest(owner=owner.__name__):
                actual = owner.metrics(left, right)
                self.assertEqual(set(actual), set(expected))
                for name, value in expected.items():
                    self.assertIsInstance(actual[name], float)
                    self.assertAlmostEqual(actual[name], value)

    def test_identical_readonly_inputs_are_not_modified(self):
        left = np.array([3, 4], dtype=np.float32)
        left.flags.writeable = False
        self.assert_metrics(left, left, dict(max_abs=0, mean_abs=0, rmse=0, cosine=1))
        np.testing.assert_array_equal(left, [3, 4])
        self.assertFalse(left.flags.writeable)

    def test_opposite_vectors_keep_signed_cosine(self):
        self.assert_metrics(np.array([1, 2]), np.array([-1, -2]),
                            dict(max_abs=4, mean_abs=3, rmse=math.sqrt(10), cosine=-1))

    def test_integer_subtraction_uses_float64_before_difference(self):
        self.assert_metrics(np.array([-128], dtype=np.int8), np.array([127], dtype=np.int8),
                            dict(max_abs=255, mean_abs=255, rmse=255, cosine=-1))

    def test_zero_norm_keeps_existing_default_cosine(self):
        self.assert_metrics(np.zeros(2), np.array([1, 2]),
                            dict(max_abs=2, mean_abs=1.5, rmse=math.sqrt(2.5), cosine=1))
        self.assert_metrics(np.zeros(2), np.zeros(2),
                            dict(max_abs=0, mean_abs=0, rmse=0, cosine=1))

    def test_shapes_flatten_and_noncontiguous_inputs_are_accepted(self):
        self.assert_metrics(np.arange(6)[::2], np.array([[0], [2], [4]]),
                            dict(max_abs=0, mean_abs=0, rmse=0, cosine=1))

    def test_scalar_arrays_are_accepted(self):
        self.assert_metrics(np.array(4), np.array(2),
                            dict(max_abs=2, mean_abs=2, rmse=2, cosine=1))

    def test_empty_and_incompatible_lengths_raise(self):
        for owner in OWNERS:
            for left, right in ((np.array([]), np.array([])),
                                (np.ones(2), np.ones(3))):
                with self.subTest(owner=owner.__name__, sizes=(left.size, right.size)):
                    with self.assertRaises(ValueError):
                        owner.metrics(left, right)

    def test_nan_inputs_propagate_nan_results(self):
        for owner in OWNERS:
            with self.subTest(owner=owner.__name__), np.errstate(invalid="ignore"):
                actual = owner.metrics(np.array([np.nan]), np.array([1]))
                self.assertEqual(set(actual), {"max_abs", "mean_abs", "rmse", "cosine"})
                self.assertTrue(all(math.isnan(value) for value in actual.values()))

    def test_infinite_inputs_keep_infinite_error_and_nan_cosine(self):
        for owner in OWNERS:
            with self.subTest(owner=owner.__name__), np.errstate(invalid="ignore"):
                actual = owner.metrics(np.array([np.inf]), np.array([1]))
                for name in ("max_abs", "mean_abs", "rmse"):
                    self.assertEqual(actual[name], math.inf)
                self.assertTrue(math.isnan(actual["cosine"]))


if __name__ == "__main__":
    unittest.main()
