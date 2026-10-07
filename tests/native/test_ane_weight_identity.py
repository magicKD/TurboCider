"""Shared Public/Private physical prefetch identity; no hardware claim."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class WeightIdentityTests(unittest.TestCase):
    def test_physical_and_logical_generations_cannot_alias_prefetched_codes(self):
        with tempfile.TemporaryDirectory(prefix="tc-weight-identity-") as directory:
            probe=Path(directory)/"probe"
            subprocess.run(["clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror",
                "tests/native/ane_weight_identity_test.cpp","-o",str(probe)],cwd=ROOT,check=True,
                capture_output=True,text=True,timeout=60)
            result=subprocess.run([str(probe)],check=True,capture_output=True,text=True,timeout=10)
            self.assertIn("PASS shared physical prefetch identity",result.stdout)


if __name__=="__main__":unittest.main()
