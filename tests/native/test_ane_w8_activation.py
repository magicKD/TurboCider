"""CPU-only activation emitter/admission regression; no ANE/device claims."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class ActivationPolicyTests(unittest.TestCase):
    def test_isolated_activation_with_unchanged_legacy_and_strict_admission(self):
        with tempfile.TemporaryDirectory(prefix="tc-w8-activation-") as directory:
            binary=Path(directory)/"activation"
            subprocess.run(["clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror","-I",str(ROOT/"native/core"),
                str(ROOT/"tests/native/ane_w8_activation_test.cpp"),str(ROOT/"native/backends/private/ane_mil.cpp"),"-o",str(binary)],check=True,cwd=ROOT)
            subprocess.run([str(binary)],check=True,cwd=ROOT)


if __name__=="__main__":unittest.main()
