"""CPU-only explicit approximate-tier admission and all skipped boundaries."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class RuntimeCacheOptionsTests(unittest.TestCase):
    def test_domain_off_preservation_and_every_real_next_consumer(self):
        with tempfile.TemporaryDirectory(prefix='tc-runtime-cache-options-') as directory:
            binary=Path(directory)/'test'
            subprocess.run(['clang++','-std=c++20','-O2','-Wall','-Wextra','-Werror','-I',str(ROOT/'native/core'),
                str(ROOT/'tests/native/qwen21_runtime_cache_options_test.cpp'),'-o',str(binary)],check=True,cwd=ROOT)
            subprocess.run([str(binary)],check=True,cwd=ROOT)


if __name__=='__main__':unittest.main()
