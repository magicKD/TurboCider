"""CPU-only tests for the native GGUF format boundary (no MLX/Python packages)."""
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def string(value):
    data = value.encode()
    return struct.pack('<Q', len(data)) + data


def fixture(types, metadata=b'', count=0):
    data = struct.pack('<IIQQ', 0x46554747, 3, len(types), count) + metadata
    for index, dtype in enumerate(types):
        data += string('weight.' + str(index))
        data += struct.pack('<IQQIQ', 2, 32, 32, dtype, 0)
    return data + bytes(4096)


class NativeGGUFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='tc-native-gguf-')
        cls.root = Path(cls.temporary.name)
        cls.binary = cls.root / 'inspect'
        source = ROOT / 'tests/native/native_gguf_test.cpp'
        subprocess.run(['xcrun', 'clang++', '-std=c++20', '-Wall', '-Wextra',
                        '-Werror', '-I', str(ROOT / 'native/core'), str(source),
                        '-o', str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def inspect(self, data, name='renamed-Q8_0.gguf'):
        path = self.root / name
        path.write_bytes(data)
        return subprocess.run([str(self.binary), str(path)], capture_output=True, text=True)

    def test_supported_tensor_types_not_filename(self):
        for dtype in (0, 1, 2, 3, 8, 30):
            with self.subTest(dtype=dtype):
                self.assertEqual(self.inspect(fixture([dtype]), 'arbitrary.gguf').returncode, 0)

    def test_mixed_and_unsupported_types_fail_even_when_renamed(self):
        for dtype in (4, 6, 7, 10, 11, 12, 13, 14, 15, 16, 20, 255):
            with self.subTest(dtype=dtype):
                result = self.inspect(fixture([8, dtype]))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('unsupported native MLX GGUF tensor type', result.stderr)

    def test_metadata_scalar_string_and_arrays(self):
        metadata = string('general.name') + struct.pack('<I', 8) + string('test')
        metadata += string('alignment') + struct.pack('<II', 4, 32)
        metadata += string('strings') + struct.pack('<IIQ', 9, 8, 2) + string('a') + string('b')
        metadata += string('numbers') + struct.pack('<IIQ2I', 9, 4, 2, 4, 8)
        self.assertEqual(self.inspect(fixture([8], metadata, 4)).returncode, 0)

    def test_invalid_header_truncation_and_overflow(self):
        valid = fixture([8])
        invalid = [b'', b'nope', valid[:23], valid[:35],
                   struct.pack('<IIQQ', 0x46554747, 4, 1, 0),
                   struct.pack('<IIQQ', 0x46554747, 3, 2**64-1, 0),
                   struct.pack('<IIQQQ', 0x46554747, 3, 1, 0, 2**64-1)]
        for data in invalid:
            with self.subTest(length=len(data)):
                self.assertNotEqual(self.inspect(data).returncode, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
