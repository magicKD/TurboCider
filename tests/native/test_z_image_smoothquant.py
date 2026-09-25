import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "z_image_smoothquant", ROOT / "tools/coreml/z_image_smoothquant.py")
SQ = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SQ)


def swiglu(x, gate, up, down):
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        g = x @ gate.T
        u = x @ up.T
        return (g / (1 + np.exp(-g))) * u @ down.T


class ZImageSmoothQuantTests(unittest.TestCase):
    def test_partial_branch_is_algebraically_equivalent(self):
        rng = np.random.default_rng(91)
        gate = rng.normal(0, 0.08, (32, 48)).astype(np.float32)
        up = rng.normal(0, 0.08, (32, 48)).astype(np.float32)
        down = rng.normal(0, 0.08, (48, 32)).astype(np.float32)
        samples = rng.normal(0, 0.2, (3, 24, 48)).astype(np.float16)
        samples[:, :, 7] *= 8  # channel outlier
        g, u, d, s1, s2 = SQ.smooth_partial_ffn(
            gate, up, down, list(samples), 0.6, 0.5, np, hidden_rows=16)
        x = samples[0].astype(np.float32)
        original = swiglu(x, gate, up, down)
        smoothed = swiglu(x / s1, g, u, d)
        self.assertTrue(np.isfinite(original).all() and np.isfinite(smoothed).all())
        np.testing.assert_allclose(original, smoothed, rtol=2e-5, atol=2e-6)
        self.assertGreater(float(s1[7]), float(np.median(s1)))
        self.assertTrue(np.isfinite(s2).all())

    def test_calibration_hash_and_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            block = root / "block0"
            block.mkdir()
            path = block / "step0.npy"
            np.save(path, np.ones((8, 16), dtype=np.float16))
            samples, first = SQ.load_calibration(block, 8, 16, np)
            self.assertEqual(len(samples), 1)
            np.save(path, np.zeros((8, 16), dtype=np.float16))
            self.assertNotEqual(first, SQ.load_calibration(block, 8, 16, np)[1])
            np.save(path, np.full((8, 16), np.nan, dtype=np.float16))
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                SQ.load_calibration(block, 8, 16, np)

    def test_calibration_union_binds_order_and_contents_and_preserves_single_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            first, second = (Path(temporary) / part for part in ("first", "second"))
            first.mkdir()
            second.mkdir()
            np.save(first / "step.npy", np.ones((8, 16), dtype=np.float16))
            np.save(second / "step.npy", np.full((8, 16), 2, dtype=np.float16))
            old_samples, old_digest = SQ.load_calibration(first, 8, 16, np)
            single, single_digest = SQ.load_calibration_union([first], 8, 16, np)
            self.assertEqual(single_digest, old_digest)
            np.testing.assert_array_equal(single[0], old_samples[0])
            samples, digest = SQ.load_calibration_union([first, second], 8, 16, np)
            self.assertEqual(len(samples), 2)
            self.assertEqual(float(samples[0][0, 0]), 1.)
            self.assertEqual(float(samples[1][0, 0]), 2.)
            self.assertNotEqual(digest, old_digest)
            self.assertNotEqual(digest, SQ.load_calibration_union(
                [second, first], 8, 16, np)[1])
            np.save(second / "step.npy", np.full((8, 16), 3, dtype=np.float16))
            self.assertNotEqual(digest, SQ.load_calibration_union(
                [first, second], 8, 16, np)[1])
            with self.assertRaisesRegex(ValueError, "1...16"):
                SQ.load_calibration_union([], 8, 16, np)
            with self.assertRaisesRegex(ValueError, "calibration"):
                SQ.load_calibration_union([first, Path(temporary) / "missing"], 8, 16, np)

    def test_invalid_calibration_and_scales_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            block = Path(temporary)
            np.save(block / "x.npy", np.ones((8, 15), dtype=np.float16))
            with self.assertRaisesRegex(ValueError, "calibration"):
                SQ.load_calibration(block, 8, 16, np)
        with self.assertRaisesRegex(ValueError, "alpha"):
            SQ.channel_scale(np.ones(2), np.ones(2), 1.1, np)

    def test_noise_refiner_can_pad_32_rows_only_when_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            block = Path(temporary)
            np.save(block / "noise.npy", np.ones((32, 16), dtype=np.float16))
            with self.assertRaisesRegex(ValueError, "calibration"):
                SQ.load_calibration(block, 64, 16, np)
            samples, _ = SQ.load_calibration(block, 64, 16, np, pad_rows=True)
            self.assertEqual(samples[0].shape, (64, 16))
            self.assertEqual(float(samples[0][-1].sum()), 0.)

    def test_image_only_calibration_binds_entire_source_but_uses_image_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            block = Path(temporary)
            sample = np.arange(12 * 16, dtype=np.float16).reshape(12, 16)
            path = block / "capture.npy"
            np.save(path, sample)
            rows, digest = SQ.load_calibration_union(
                [block], 8, 16, np, source_rows=12)
            np.testing.assert_array_equal(rows[0], sample[:8])
            sample[11] = -1
            np.save(path, sample)
            rows_again, changed = SQ.load_calibration_union(
                [block], 8, 16, np, source_rows=12)
            np.testing.assert_array_equal(rows_again[0], rows[0])
            self.assertNotEqual(digest, changed)
            with self.assertRaisesRegex(ValueError, "source rows"):
                SQ.load_calibration(block, 12, 16, np, source_rows=8)

    def test_hidden_a8_search_is_finite_and_output_aware(self):
        rng = np.random.default_rng(25)
        gate = rng.normal(0, 0.03, (16, 32)).astype(np.float32)
        up = rng.normal(0, 0.03, (16, 32)).astype(np.float32)
        down = rng.normal(0, 0.03, (32, 16)).astype(np.float32)
        samples = [rng.normal(0, 0.8, (64, 32)).astype(np.float16)]
        scale, trials = SQ.calibrate_hidden_a8(
            gate, up, down, np.ones(32, dtype=np.float32), samples,
            0.04, 1., np, rows_per_sample=32)
        self.assertGreater(float(scale), 0)
        self.assertEqual(len(trials), 7)
        self.assertTrue(np.isfinite([trial["nmse"] for trial in trials]).all())

    def test_hidden_channel_group_scales_cover_exact_nonoverlapping_columns(self):
        rng = np.random.default_rng(31)
        gate = rng.normal(0, .03, (16, 32)).astype(np.float32)
        up = rng.normal(0, .03, (16, 32)).astype(np.float32)
        down = rng.normal(0, .03, (32, 16)).astype(np.float32)
        up[8:] *= 12
        samples = [rng.normal(0, .8, (32, 32)).astype(np.float16)]
        scales = SQ.calibrate_hidden_channel_groups(
            gate, up, down, np.ones(32, dtype=np.float32), samples, .04, 1., 2, np,
            rows_per_sample=16)
        self.assertEqual(len(scales), 2)
        self.assertTrue(all(np.isfinite(s) and s > 0 for s in scales))
        self.assertGreater(float(scales[1]), float(scales[0]))
        with self.assertRaisesRegex(ValueError, "group geometry"):
            SQ.calibrate_hidden_channel_groups(gate, up, down, np.ones(32),
                                               samples, .04, 1., 3, np)

    def test_joint_group_screen_is_deterministic_and_reduces_training_loss(self):
        rng = np.random.default_rng(87)
        gate = rng.normal(0, .07, (16, 32)).astype(np.float32)
        up = rng.normal(0, .07, (16, 32)).astype(np.float32)
        down = rng.normal(0, .07, (32, 16)).astype(np.float32)
        samples = [rng.normal(0, .8, (48, 32)).astype(np.float16)]
        first, metrics = SQ.screen_joint_hidden_channel_groups(
            gate, up, down, np.ones(32, np.float32), samples, .04, 1., 4, np,
            rows_per_sample=16)
        second, repeated = SQ.screen_joint_hidden_channel_groups(
            gate, up, down, np.ones(32, np.float32), samples, .04, 1., 4, np,
            rows_per_sample=16)
        self.assertEqual(first, second)
        self.assertEqual(metrics, repeated)
        self.assertEqual(len(first), 4)
        self.assertTrue(all(np.isfinite(s) and s > 0 for s in first))
        self.assertLessEqual(metrics["joint_nmse"], metrics["initial_nmse"] + 1e-12)
        with self.assertRaisesRegex(ValueError, "joint hidden A8 group geometry"):
            SQ.screen_joint_hidden_channel_groups(gate, up, down, np.ones(32),
                                                   samples, .04, 1., 3, np)

    def test_routed_ane_and_gpu_channels_partition_full_swiglu(self):
        rng = np.random.default_rng(112)
        x = rng.normal(size=(3, 8)).astype(np.float32)
        g = rng.normal(0, .2, (128, 8)).astype(np.float32)
        u = rng.normal(0, .2, (128, 8)).astype(np.float32)
        d = rng.normal(0, .2, (8, 128)).astype(np.float32)
        ane, gpu = SQ.route_channel_indices([1, 3], 32, 128, 64, np)
        np.testing.assert_array_equal(ane, np.r_[32:64, 96:128])
        np.testing.assert_array_equal(gpu, np.r_[0:32, 64:96])
        np.testing.assert_allclose(swiglu(x, g, u, d),
                                   swiglu(x, g[ane], u[ane], d[:, ane]) +
                                   swiglu(x, g[gpu], u[gpu], d[:, gpu]),
                                   rtol=2e-6, atol=2e-6)
        for invalid in ([1, 1], [2, 1], [1, 4], [True, 2]):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                    ValueError, "channel routing"):
                SQ.route_channel_indices(invalid, 32, 128, 64, np)

    def test_input_a8_search_keeps_maximum_fallback(self):
        rng = np.random.default_rng(11)
        gate = rng.normal(0, 0.1, (16, 32)).astype(np.float32)
        up = rng.normal(0, 0.1, (16, 32)).astype(np.float32)
        down = rng.normal(0, 0.1, (32, 16)).astype(np.float32)
        sample = rng.normal(0, 0.4, (32, 32)).astype(np.float16)
        sample[0, 0] = 20
        scale, trials = SQ.calibrate_input_a8(
            gate, up, down, np.ones(32, dtype=np.float32), [sample], np)
        self.assertEqual(len(trials), 7)
        self.assertEqual(float(scale), min(trials, key=lambda trial: trial["nmse"])["scale"])
        self.assertGreaterEqual(trials[-1]["scale"] * 127, 20 - 0.01)
        with self.assertRaisesRegex(ValueError, "calibration"):
            SQ.calibrate_input_a8(gate, up, down, np.ones(32), [], np)

    def test_representative_rows_include_rare_conditioning_outliers(self):
        sample = np.ones((1056, 4), dtype=np.float16)
        sample[1025, 2] = 36
        sample[1026, 3] = 20
        selected = SQ.representative_rows(sample, 32, np)
        self.assertIn(1025, selected)
        self.assertIn(1026, selected)
        self.assertIn(0, selected)
        with self.assertRaisesRegex(ValueError, "row selection"):
            SQ.representative_rows(sample, 0, np)


if __name__ == "__main__":
    unittest.main()
