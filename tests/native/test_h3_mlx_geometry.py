import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class H3MLXGeometryTests(unittest.TestCase):
    def test_geometry_and_scheduler(self):
        with tempfile.TemporaryDirectory(prefix="turbocider-h3-mlx-") as directory:
            binary = Path(directory) / "geometry-test"
            subprocess.run([
                "clang++", "-std=c++20", "-O2",
                str(ROOT / "tests/native/h3_mlx_geometry_test.cpp"),
                str(ROOT / "native/models/h3_mlx/geometry.cpp"),
                str(ROOT / "native/models/h3_mlx/vsa.cpp"),
                str(ROOT / "native/models/h3_mlx/conditioner_math.cpp"),
                str(ROOT / "native/core/common.cpp"),
                "-o", str(binary),
            ], check=True, cwd=ROOT)
            subprocess.run([str(binary)], check=True, cwd=ROOT)


if __name__ == "__main__":
    unittest.main()
