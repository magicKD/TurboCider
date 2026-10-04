"""Host-only strict checks for the Core ML Python FP16 output bridge."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/validation"))
from runtime_ane_swiglu_int8_probe import verify_prediction_boundary


class PythonBridgeTests(unittest.TestCase):
    def fixture(self, dtype=np.float32):
        spec = {"outputs": {"y": [1, 5], "h": [1, 5]}}
        row = {"model_description_outputs": {
            name: {"feature_type": "multiArrayType", "dtype": "FLOAT16", "shape": shape}
            for name, shape in spec["outputs"].items()}, "mil_interface_outputs": {"y": "FLOAT16", "h": "FLOAT16"}}
        actual = {name: np.array([[.5, -.25, 0., -0., 65504.]], dtype=dtype) for name in spec["outputs"]}
        return actual, row, spec

    def test_native_and_exactly_promoted_fp16_are_accepted_without_mutation(self):
        for dtype in (np.float16, np.float32):
            actual, row, spec = self.fixture(dtype)
            originals = {name: value.copy() for name, value in actual.items()}
            verify_prediction_boundary(actual, row, spec, "synthetic")
            for name in actual:
                np.testing.assert_array_equal(actual[name], originals[name])
                self.assertEqual(actual[name].dtype, dtype)
                receipt = row["prediction_boundaries"][0]["outputs"][name]
                self.assertTrue(receipt["fp16_roundtrip_exact"])
                self.assertEqual(receipt["python_bridge_promoted"], dtype == np.float32)

    def test_nonrepresentable_float32_float64_and_nonfinite_values_are_rejected(self):
        actual, row, spec = self.fixture()
        for value in (np.float32(.1), np.float32(np.inf), np.float32(-np.inf), np.float32(np.nan)):
            invalid = {**actual, "y": actual["y"].copy()}
            invalid["y"][0, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                verify_prediction_boundary(invalid, deepcopy(row), spec, "invalid")
        actual, row, spec = self.fixture(np.float64)
        with self.assertRaises(ValueError):
            verify_prediction_boundary(actual, row, spec, "invalid")

    def test_model_description_and_mil_must_both_remain_fp16(self):
        actual, row, spec = self.fixture()
        for location in ("description", "mil"):
            invalid = deepcopy(row)
            if location == "description":
                invalid["model_description_outputs"]["y"]["dtype"] = "FLOAT32"
            else:
                invalid["mil_interface_outputs"]["y"] = "FLOAT32"
            with self.subTest(location=location), self.assertRaises(ValueError):
                verify_prediction_boundary(actual, invalid, spec, "invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
