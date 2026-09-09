"""File provenance hashing must fit App/libdispatch worker thread stacks."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class WorkerStackTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / 'build/native/libturbocider.dylib').is_file(), 'build native first')
    def test_sha256_on_small_worker_stack(self):
        with tempfile.TemporaryDirectory(prefix='tc-hash-test-') as folder:
            binary = Path(folder) / 'hash-worker'
            subprocess.run([
                'clang++', '-std=c++20', '-I', str(ROOT / 'native'),
                '-I', str(ROOT / 'native/core'),
                str(ROOT / 'tests/native/hash_small_stack_test.cpp'),
                '-L' + str(ROOT / 'build/native'), '-lturbocider',
                '-Wl,-rpath,' + str(ROOT / 'build/native'), '-o', str(binary),
            ], check=True, capture_output=True)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('PASS: SHA-256 on a 256 KiB worker stack', result.stdout)


if __name__ == '__main__':
    unittest.main(verbosity=2)
