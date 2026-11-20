"""File provenance hashing must fit App/libdispatch worker thread stacks."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path(os.environ.get('TURBOCIDER_TEST_NATIVE_DIR', ROOT / 'build/native')).resolve()


class WorkerStackTests(unittest.TestCase):
    @unittest.skipUnless((NATIVE / 'libturbocider.dylib').is_file(), 'build native first')
    def test_sha256_on_small_worker_stack(self):
        with tempfile.TemporaryDirectory(prefix='tc-hash-test-') as folder:
            directory = Path(folder)
            fixture = directory / 'weights'
            payload = bytes(range(256)) * 16385
            fixture.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            binary = directory / 'hash-worker'
            subprocess.run([
                'clang++', '-std=c++20', '-I', str(ROOT / 'native'),
                '-I', str(ROOT / 'native/core'),
                str(ROOT / 'tests/native/hash_small_stack_test.cpp'),
                '-L' + str(NATIVE), '-lturbocider',
                '-Wl,-rpath,' + str(NATIVE), '-o', str(binary),
            ], check=True, capture_output=True)
            result = subprocess.run([str(binary), str(fixture), expected],
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('PASS: SHA-256 on a 256 KiB worker stack', result.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
