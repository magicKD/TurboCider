"""Verify generic StageExecutor INT8 materialization against resident MLX packing."""
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest

from z_image_convrot_fixture import write_convrot_fixture

ROOT = Path(__file__).resolve().parents[2]

@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU") == "1", "requires Metal GPU")
class ExactInt8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory(prefix="tc-z-int8-exact-")
        cls.root = Path(cls.folder.name)
        cls.binary = cls.root / "probe"
        mlx = Path(sysconfig.get_paths()["purelib"]) / "mlx"
        subprocess.run([
            "clang++", "-std=c++20", "-I", str(ROOT / "native"),
            "-I", str(ROOT / "native/core"), "-isystem", str(mlx / "include"),
            str(ROOT / "tests/native/z_image_int8_exact_test.cpp"),
            "-L" + str(ROOT / "build/native"), "-lturbocider",
            "-L" + str(mlx / "lib"), "-lmlx",
            "-Wl,-rpath," + str(ROOT / "build/native"), "-Wl,-rpath," + str(mlx / "lib"),
            "-o", str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_generic_exact_conversion_execution_and_lifetime(self):
        for slots in (1, 2, 3):
            for suffix in (0, 1):
                with self.subTest(slots=slots, suffix=suffix):
                    path = write_convrot_fixture(self.root)
                    result = subprocess.run([str(self.binary), str(path), str(slots), str(suffix)],
                                            capture_output=True, text=True, timeout=90)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("PASS generic INT8", result.stdout)

    def test_concurrent_slot_fills(self):
        result = subprocess.run([str(self.binary), str(write_convrot_fixture(self.root)), "3", "1", "2"],
            capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fp32_scale_identity_and_values(self):
        result = subprocess.run([str(self.binary), str(write_convrot_fixture(self.root)), "3", "1"],
            env=dict(os.environ, TURBOCIDER_Z_CONVROT_FP32_SCALES="1"),
            capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_scale_geometry_rejected_before_execution(self):
        path = write_convrot_fixture(self.root, lambda h: [v.update(shape=[1, v["shape"][0]])
            for k, v in h.items() if k.endswith(".weight_scale")])
        result = subprocess.run([str(self.binary), str(path), "3", "0"],
                                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("scale geometry", result.stderr)

if __name__ == "__main__":
    unittest.main(verbosity=2)
