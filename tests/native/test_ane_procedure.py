"""Pure host metadata normalization; no driver/model qualification."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class AneProcedureTests(unittest.TestCase):
    def test_nonzero_procedure_and_global_symbol_indices(self):
        with tempfile.TemporaryDirectory(prefix='tc-ane-procedure-') as d:
            binary = Path(d) / 'probe'
            subprocess.run(['clang++', '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror',
                            str(ROOT / 'tests/native/ane_procedure_test.cpp'), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
