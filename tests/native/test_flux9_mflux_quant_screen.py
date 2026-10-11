from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from flux9_mflux_quant_screen import validate_report


class FluxQuantScreenTests(unittest.TestCase):
    def report(self, bits=4):
        return dict(quantization=bits, runs=[dict(actual_steps=[0, 1, 2, 3], actual_quantization=bits, loras=[]) for _ in range(4)])

    def test_stock_quantized_reference_is_not_relabelled_bf16(self):
        validate_report(self.report(), "q4")
        with self.assertRaises(ValueError):
            validate_report(self.report(), "bf16")

    def test_missing_steps_adapter_and_incomplete_sequence_are_rejected(self):
        for field, value in (("actual_steps", [0]), ("actual_quantization", 8), ("loras", ["adapter"])):
            report = self.report()
            report["runs"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_report(report, "q4")
        with self.assertRaises(ValueError):
            validate_report(dict(quantization=4, runs=[]), "q4")


if __name__ == "__main__":
    unittest.main()
