"""FP64 numerical gates, including zero/nonfinite source negatives."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]

class TensorMetricsTests(unittest.TestCase):
    def test_fixed_host_metrics(self):
        with tempfile.TemporaryDirectory(prefix="tc-tensor-metrics-") as raw:
            binary=Path(raw)/"probe"
            subprocess.run(["xcrun","clang++","-std=c++20","-O2","-ffp-contract=off","-Wall","-Wextra","-Werror",
                "-I",str(ROOT/"native/core"),str(ROOT/"tests/native/tensor_metrics_test.cpp"),"-o",str(binary)],check=True)
            subprocess.run([str(binary)],check=True)

if __name__=="__main__":unittest.main()
