"""CPU-only edit first-step routing/recipe/prefetch contract; no inference."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class EditQkvStep0Tests(unittest.TestCase):
    def test_domain_and_every_step_prefetch_boundary(self):
        with tempfile.TemporaryDirectory(prefix='tc-edit-qkv-step0-') as d:
            binary=Path(d)/'probe'
            subprocess.run(['clang++','-std=c++20','-O2','-Wall','-Wextra','-Werror','-I',str(ROOT/'native/core'),
                str(ROOT/'tests/native/qwen21_edit_qkv_step0_test.cpp'),str(ROOT/'native/core/common.cpp'),'-o',str(binary)],check=True,cwd=ROOT)
            subprocess.run([str(binary)],check=True,cwd=ROOT)


if __name__=='__main__':unittest.main()
