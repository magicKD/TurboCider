"""Converted W8 cache policy and actual shared Metal staging, no model fixture."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class WeightCodeCacheTests(unittest.TestCase):
    def test_capacity_policy(self):
        with tempfile.TemporaryDirectory(prefix="tc-code-cache-policy-") as directory:
            probe=Path(directory)/"probe"
            result=subprocess.run(["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror",
                str(ROOT/"tests/native/ane_weight_code_cache_policy_test.cpp"),"-o",str(probe)],
                cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([str(probe)],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr);print(result.stdout)

    @unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU")=="1","explicit shared Metal opt-in required")
    def test_actual_converted_cache_and_producer_leases(self):
        with tempfile.TemporaryDirectory(prefix="tc-code-cache-gpu-") as directory:
            probe=Path(directory)/"probe"
            result=subprocess.run(["xcrun","clang++","-std=c++20","-O2","-ffp-contract=off","-Wall","-Wextra","-Werror",
                "-Wno-deprecated-declarations","-fobjc-arc","-mmacosx-version-min=26.2",
                str(ROOT/"tests/native/ane_weight_code_cache_test.mm"),str(ROOT/"native/backends/ane_gpu.mm"),
                str(ROOT/"native/core/gguf_decode.cpp"),"-framework","Foundation","-framework","Metal",
                "-framework","IOSurface","-o",str(probe)],cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([str(probe)],cwd=ROOT,capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("PASS weight code cache GPU cases=11",result.stdout);print(result.stdout)


if __name__=="__main__":unittest.main()
