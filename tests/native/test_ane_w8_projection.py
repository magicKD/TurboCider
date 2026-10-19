"""CPU emitter recipe/bounds controls; no model or device execution."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class ProjectionPolicyTests(unittest.TestCase):
    def test_emission_and_authorized_fixed_base_only_policy(self):
        with tempfile.TemporaryDirectory(prefix="tc-w8-projection-") as directory:
            binary=Path(directory)/"projection"
            subprocess.run(["clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror","-I",str(ROOT/"native/core"),
                str(ROOT/"tests/native/ane_w8_projection_test.cpp"),str(ROOT/"native/backends/private/ane_mil.cpp"),"-o",str(binary)],check=True,cwd=ROOT)
            subprocess.run([str(binary)],check=True,cwd=ROOT)


if __name__=="__main__":unittest.main()
