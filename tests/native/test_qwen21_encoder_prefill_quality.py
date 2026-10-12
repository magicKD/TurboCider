import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from qwen21_encoder_prefill_quality import difference, load_tensor


class EncoderPrefillQualityTests(unittest.TestCase):
    def test_complete_fp16_bf16_fp32_and_difference(self):
        with tempfile.TemporaryDirectory(prefix="tc-qwen-prefill-quality-") as raw:
            path = Path(raw) / "tensor.safetensors"
            expected = np.array([[0, -.5, 1, 2]], dtype=np.float32)
            for dtype in ("F16", "BF16", "F32"):
                payload = ((expected.view(np.uint32) >> 16).astype("<u2").tobytes() if dtype == "BF16" else
                           expected.astype("<f2" if dtype == "F16" else "<f4").tobytes())
                header = json.dumps(dict(tensor=dict(dtype=dtype, shape=[1, 4], data_offsets=[0, len(payload)]))).encode()
                path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
                actual = load_tensor(path)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(difference(actual, actual)["relative_l2"], 0)
                self.assertGreater(difference(actual, actual + .1)["relative_l2"], 0)

    def test_truncated_nonfinite_geometry_and_cropped_comparison_rejected(self):
        with tempfile.TemporaryDirectory(prefix="tc-qwen-prefill-quality-") as raw:
            path = Path(raw) / "tensor.safetensors"
            for dtype, shape, offsets, payload in (
                ("F32", [1, 2], [0, 8], struct.pack("<f", 0)),
                ("F32", [1, 1], [0, 4], struct.pack("<f", float("nan"))),
                ("F32", [1, 0], [0, 0], b""),
                ("F32", [1, True], [0, 4], struct.pack("<f", 0)),
                ("F16", [1, 1], [1, 3], struct.pack("<e", 0)),
            ):
                header = json.dumps(dict(tensor=dict(dtype=dtype, shape=shape, data_offsets=offsets))).encode()
                path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
                with self.assertRaises(ValueError):
                    load_tensor(path)
            with self.assertRaises(ValueError):
                difference(np.zeros((1, 4)), np.zeros((1, 2)))


if __name__ == "__main__":
    unittest.main()
