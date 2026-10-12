"""CPU-only numerical contracts for the isolated dynamic-weight candidate."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/coreml"))
from runtime_ane_int8_candidate import prepare_inputs, specification


class PreparationTests(unittest.TestCase):
    def test_sylvester_channel_order(self):
        # A basis vector distinguishes the specified ordering/signs from a
        # different orthogonal transform that would also preserve a dot product.
        x = np.eye(4, dtype=np.float32)
        values = prepare_inputs(x, x, specification(4, 4, 4, "fp16", 4))
        expected = np.array([[1, 1, 1, 1], [1, -1, 1, -1],
                             [1, 1, -1, -1], [1, -1, -1, 1]], dtype=np.float32) / 2
        for key in ("x", "w"):
            np.testing.assert_array_equal(values[key] * values[key + "_scale"], expected)

    def test_rotation_and_normalization_preserve_projection(self):
        rng = np.random.default_rng(91)
        x = rng.normal(size=(4, 64)).astype(np.float32)
        w = rng.normal(size=(6, 64)).astype(np.float32)
        x[0] = 0
        w[0] = 0
        expected = np.einsum("ik,jk->ij", x.astype(np.float64), w.astype(np.float64))
        for group in (1, 2, 16, 64):
            with self.subTest(group=group):
                values = prepare_inputs(x, w, specification(4, 64, 6, "qdq_int8", group))
                restored = {key: values[key].astype(np.float64) *
                            values[key + "_scale"].astype(np.float64) for key in ("x", "w")}
                actual = np.einsum("ik,jk->ij", restored["x"], restored["w"])
                relative = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
                self.assertLess(relative, 0.001)
                self.assertTrue(np.array_equal(actual[0], np.zeros(6)))
                self.assertTrue(np.array_equal(actual[:, 0], np.zeros(4)))
                self.assertTrue(all(v.dtype == np.float16 and v.flags.c_contiguous
                                    for v in values.values()))

    def test_unrepresentable_scales_fail_instead_of_corrupting_projection(self):
        spec = specification(2, 64, 3, "fp16", 64)
        w = np.ones((3, 64), dtype=np.float32)
        for x in (np.full((2, 64), np.inf), np.full((2, 64), np.nan),
                  np.full((2, 64), 10000.), np.full((2, 64), 1e-12),
                  np.ones((2, 63)), np.ones((2, 64), dtype=np.int32)):
            with self.subTest(shape=x.shape, dtype=x.dtype):
                with self.assertRaises(ValueError):
                    prepare_inputs(x, w, spec)

    def test_zero_operand_and_input_ownership(self):
        x = np.zeros((2, 64), dtype=np.float32)
        w = np.arange(192, dtype=np.float32).reshape(3, 64) - 90
        original = w.copy()
        values = prepare_inputs(x, w, specification(2, 64, 3, "fp16", 64))
        np.testing.assert_array_equal(w, original)
        self.assertTrue(all(np.isfinite(v).all() for v in values.values()))
        self.assertTrue(np.all(values["x"] == 0))
        self.assertTrue(np.all(values["x_scale"] > 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
