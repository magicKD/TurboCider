"""Bitwise and bounds checks for the safe Core ML output-copy fallback."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class CoreMLOutputCopyTests(unittest.TestCase):
    def test_strided_copy(self):
        with tempfile.TemporaryDirectory(prefix='tc-coreml-copy-') as folder:
            binary = Path(folder) / 'copy-test'
            flags = ['-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror']
            sanitizer = os.environ.get('TC_COPY_SANITIZE', '')
            if sanitizer:
                self.assertIn(sanitizer, ('1', 'address', 'undefined', 'address,undefined'))
                flags += ['-fsanitize=' + ('address,undefined' if sanitizer == '1' else sanitizer),
                          '-fno-omit-frame-pointer']
            subprocess.run(['clang++', *flags, '-I', str(ROOT / 'native'),
                            str(ROOT / 'tests/native/coreml_output_copy_test.cpp'),
                            '-o', str(binary)], check=True)
            result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('PASS: bit-exact FP16 copies', result.stdout)
            print(result.stdout, end='')


if __name__ == '__main__':
    unittest.main(verbosity=2)
