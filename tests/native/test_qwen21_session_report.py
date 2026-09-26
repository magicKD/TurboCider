import json
from pathlib import Path
import struct
import tempfile
import unittest

from tools.validation.qwen21_compare_sessions import tensor_identity


class SessionTensorIdentityTests(unittest.TestCase):
    def fixture(self, path, payload=b"\x00\x00\x80\x3f", shape=None, metadata=None):
        header = {"tensor": {"dtype": "F32", "shape": shape or [1], "data_offsets": [0, len(payload)]}}
        if metadata:
            header["__metadata__"] = metadata
        encoded = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)

    def test_identical_tensor_ignores_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = (Path(directory) / name for name in ("a", "b"))
            self.fixture(a)
            self.fixture(b, metadata={"run": "candidate"})
            self.assertEqual(tensor_identity(a), tensor_identity(b))

    def test_rejects_shape_or_content_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = (Path(directory) / name for name in ("a", "b"))
            self.fixture(a)
            self.fixture(b, shape=[1, 1])
            self.assertNotEqual(tensor_identity(a), tensor_identity(b))
            self.fixture(b, payload=b"\x00\x00\x00\x40")
            self.assertNotEqual(tensor_identity(a), tensor_identity(b))

    def test_truncation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid"
            for data in (b"", struct.pack("<Q", 100) + b"{}"):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    tensor_identity(path)
            self.fixture(path)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                tensor_identity(path)


if __name__ == "__main__":
    unittest.main()
