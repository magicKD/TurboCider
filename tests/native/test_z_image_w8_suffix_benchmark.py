import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_z_image_w8_suffix", ROOT / "tools/native/benchmark_z_image_w8_suffix.py")
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class FakeReader:
    def __init__(self, values, dtype="BF16"):
        self.values = values
        self.dtype = dtype

    def _tensor_record(self, name):
        data = self.values.tobytes()
        return None, data, {"shape": list(self.values.shape), "dtype": self.dtype,
                            "data_offsets": [0, len(data)]}, 0


class ZImageW8SuffixBenchmarkTests(unittest.TestCase):
    def test_exact_bf16_bits_for_gate_and_down_slices(self):
        # BF16 0x4780 is 65536, beyond FP16's finite range. Ordinary
        # in-range BF16 values are exactly representable by FP16.
        bits = np.array([[0x3f81, 0x4001, 0xbf81, 0x4041],
                         [0x3f80, 0x4000, 0x4780, 0x4040]], dtype=np.uint16)
        reader = FakeReader(bits)
        gate = BENCHMARK.read_bf16_suffix(reader, "gate", (2, 4), 1, False, np)
        down = BENCHMARK.read_bf16_suffix(reader, "down", (2, 4), 2, True, np)
        np.testing.assert_array_equal(gate.view(np.uint32), bits[1:].astype(np.uint32) << 16)
        np.testing.assert_array_equal(down.view(np.uint32), bits[:, 2:].astype(np.uint32) << 16)

    def test_reject_non_bf16_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "expected original BF16"):
            BENCHMARK.read_bf16_suffix(FakeReader(np.zeros((2, 4), np.uint16), "F16"),
                                       "gate", (2, 4), 1, False, np)


if __name__ == "__main__":
    unittest.main()
