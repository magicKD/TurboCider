"""CPU admission/sampling policy; not ordinary-adapter image qualification."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class BaseRuntimeLoraOptionsTests(unittest.TestCase):
    def test_separate_base_schedule_domain_and_legacy_default(self):
        with tempfile.TemporaryDirectory(prefix='tc-qwen-base-runtime-lora-') as directory:
            binary=Path(directory)/'probe'
            subprocess.run(['clang++','-std=c++20','-O2','-Wall','-Wextra','-Werror',
                str(ROOT/'tests/native/qwen21_base_runtime_lora_options_test.cpp'),str(ROOT/'native/core/common.cpp'),'-o',str(binary)],check=True,cwd=ROOT)
            subprocess.run([str(binary)],check=True,cwd=ROOT)


if __name__=='__main__':unittest.main()
