"""CPU-only function-family request/cache-key admission; no model/driver."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class RuntimeFunctionsTests(unittest.TestCase):
    def test_explicit_base40_domain_and_family_identity(self):
        with tempfile.TemporaryDirectory(prefix='tc-qwen-functions-') as d:
            binary=Path(d)/'probe'
            subprocess.run(['clang++','-std=c++20','-O2','-Wall','-Wextra','-Werror',
                '-I',str(ROOT/'native/core'),
                str(ROOT/'tests/native/qwen21_runtime_functions_test.cpp'),str(ROOT/'native/core/common.cpp'),'-o',str(binary)],check=True,cwd=ROOT)
            subprocess.run([str(binary)],check=True,cwd=ROOT)


if __name__=='__main__':unittest.main()
