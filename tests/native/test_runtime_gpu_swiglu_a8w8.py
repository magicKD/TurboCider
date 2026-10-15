"""CPU-only independent oracles for portable GPU A8W8 arithmetic."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools/metal"), str(ROOT / "tools/validation")]
from runtime_swiglu_a8w8 import VARIANTS, geometry, integer_dot_reference, quantize_reference, reference
from runtime_ane_swiglu_int8_probe import errors, fixtures, unquantized_oracle


class ArithmeticTests(unittest.TestCase):
    def test_fp16_error_metrics_are_finite_and_do_not_underflow_or_overflow(self):
        zero = np.zeros((2, 3), dtype=np.float16)
        with np.errstate(all="raise"):
            self.assertEqual(errors(zero, zero), {"relative_l2": 0., "max_absolute": 0.})
            large = np.full((2, 3), 60000, dtype=np.float16)
            self.assertEqual(errors(large, large)["relative_l2"], 0.)
            self.assertAlmostEqual(errors(zero, large)["relative_l2"], 1.)
            self.assertGreater(errors(np.ones_like(zero), zero)["relative_l2"], 1e12)
        with self.assertRaises(ValueError):
            errors(np.full_like(zero, np.nan), zero)

    def test_signed_integer_dot_and_accumulator_bound(self):
        a = np.array([[127, -127, 0, 1, -1], [-2, 3, -4, 5, -6]], np.int8)
        b = np.array([[-127, 127, 1, -1, 1], [6, -5, 4, -3, 2]], np.int8)
        expected = np.array([[sum(int(x) * int(y) for x, y in zip(row, column))
                              for column in b] for row in a], np.int32)
        np.testing.assert_array_equal(integer_dot_reference(a, b), expected)
        maximum = np.full((1, 768), 127, np.int8)
        self.assertEqual(integer_dot_reference(maximum, maximum)[0, 0], 768 * 127 * 127)
        for bad in (np.zeros((1, 4), np.int8), a.astype(np.int32)):
            with self.assertRaises(ValueError):
                integer_dot_reference(bad, b)
        minimum = np.full((1, 768), -128, np.int8)
        self.assertEqual(integer_dot_reference(minimum, minimum)[0, 0], 768 * 128 * 128)
        unsafe = np.zeros((1, 131072), np.int8)
        with self.assertRaises(ValueError):
            integer_dot_reference(unsafe, unsafe)

    def test_symmetric_quantization_zero_rows_and_owned_inputs(self):
        x = np.array([[1., .5, -.5, -1.], [0., 0., 0., 0.]], np.float32)
        x.flags.writeable = False
        original = x.copy()
        half, codes, scale = quantize_reference(x, 1)
        np.testing.assert_array_equal(codes, [[127, 64, -64, -127], [0, 0, 0, 0]])
        np.testing.assert_array_equal(scale, [[1.], [1.]])
        np.testing.assert_array_equal(x, original)
        self.assertEqual(codes.dtype, np.int8)
        self.assertEqual(half.dtype, np.float16)

    def test_all_full_swiglu_arithmetic_oracles_and_lora_bases(self):
        config = geometry("tiny")
        case = fixtures(5, 64, 96, 4)[0]
        exact = unquantized_oracle(case)
        actuals = {}
        for variant in VARIANTS:
            actuals[variant] = actual = reference(case, config, variant)
            for name in ("h", "y"):
                relative = np.linalg.norm(actual[name].astype(np.float64) - exact[name]) / np.linalg.norm(exact[name])
                self.assertLess(relative, .04)
                self.assertEqual(actual[name].dtype, np.float16)
            _, x, weights, adapters = case
            a, b, scale = adapters[2]
            no_down = ("no_down", x, weights, [*adapters[:2], (a, np.zeros_like(b), scale)])
            base = reference(no_down, config, variant)
            # Independent original-h down branch; never quantized/rotated.
            low = np.einsum("ik,rk->ir", actual["h"].astype(np.float64), a.astype(np.float64))
            correction = np.einsum("ir,or->io", low, b.astype(np.float64)) * scale
            expected = (base["y"].astype(np.float64) + correction).astype(np.float16)
            np.testing.assert_allclose(actual["y"], expected, rtol=.001, atol=.001)
        # These are two distinct valid rounding contracts, not an equality claim.
        self.assertFalse(np.array_equal(actuals["a8w8_integer_alu"]["y"], actuals["a8w8_dequant_fp16"]["y"]))

    def test_bounded_geometry_and_zero_case(self):
        with self.assertRaises(ValueError):
            geometry("large")
        self.assertLess(geometry("medium")["integer_accumulator_max_abs"], np.iinfo(np.int32).max)
        config = geometry("tiny")
        case = fixtures(5, 64, 96, 4)[-1]
        for variant in VARIANTS:
            self.assertTrue(all(np.all(value == 0) for value in reference(case, config, variant).values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
