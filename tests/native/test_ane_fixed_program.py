"""CPU-only fixed-program option/authorization/state guards."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class FixedProgramTests(unittest.TestCase):
    def test_option_authorization_and_prefix_health(self):
        with tempfile.TemporaryDirectory(prefix="tc-fixed-program-") as directory:
            binary=Path(directory)/"policy"
            subprocess.run(["clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror","-I",str(ROOT/"native/core"),
                str(ROOT/"tests/native/ane_fixed_program_test.cpp"),"-o",str(binary)],check=True)
            subprocess.run([str(binary)],check=True)


if __name__=="__main__":unittest.main()
