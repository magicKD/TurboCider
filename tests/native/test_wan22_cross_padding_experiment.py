"""Pure-math checks for the non-production padding-compression probe."""
import math
import struct
import unittest

from tools.validation.wan22_cross_padding import padding_biases


class Wan22CrossPaddingExperimentTests(unittest.TestCase):
    def test_biases_representable_and_reported_mass_error(self):
        for rows, copies in ((1, 1), (459, 1), (459, 16), (462, 16), (511, 64)):
            with self.subTest(rows=rows, copies=copies):
                biases, error = padding_biases(rows, copies)
                self.assertEqual(len(biases), copies)
                for bias in biases:
                    bits = struct.unpack(">I", struct.pack(">f", bias))[0]
                    self.assertEqual(bits & 0xffff, 0, "bias is not exactly BF16 representable")
                actual = abs(sum(math.exp(bias) for bias in biases) / rows - 1)
                self.assertAlmostEqual(actual, error, places=12)
                self.assertLess(error, .01)
        self.assertLess(padding_biases(459, 16)[1], 1e-5)
        self.assertLess(padding_biases(462, 16)[1], 1e-5)

    def test_invalid_geometry_fails(self):
        for rows, copies in ((0, 1), (4, 0), (4, 5), (1025, 1), (512, 65),
                             (-1, 1), (True, 1), (4.0, 1)):
            with self.subTest(rows=rows, copies=copies), self.assertRaises(ValueError):
                padding_biases(rows, copies)


if __name__ == "__main__":
    unittest.main()
