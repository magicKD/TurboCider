"""Verify generic StageExecutor INT8 materialization against resident MLX packing."""
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest
import json
import struct

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

    def convrot_fixture(self, change=None):
        import numpy as np
        header, payload = {}, bytearray()
        def add(name, values, dtype):
            data = values.tobytes()
            header[name] = {"dtype": dtype, "shape": list(values.shape),
                           "data_offsets": [len(payload), len(payload) + len(data)]}
            payload.extend(data)
        add("x_embedder.weight", np.array([[1.00390625, 1.01171875], [-0.01, 0.03]], np.float32), "F32")
        def projection(prefix, rows, cols, block):
            q = ((np.arange(rows * cols, dtype=np.int32) + block) % 256 - 128).astype(np.int8).reshape(rows, cols)
            add(prefix + ".weight", q, "I8")
            # Includes BF16 halfway cases to verify CPU packing matches MLX.
            add(prefix + ".weight_scale", ((np.arange(rows, dtype=np.float32) % 13 + 1.00390625) / 128).reshape(rows, 1), "F32")
            add(prefix + ".comfy_quant", np.frombuffer(b'{"format":"int8","rotate":true}', dtype=np.uint8), "U8")
        for kind in ("noise_refiner", "context_refiner", "layers"):
            for block in range(30 if kind == "layers" else 2):
                prefix = f"{kind}.{block}"
                for w, rows, cols in ((1, 512, 256), (2, 256, 512), (3, 512, 256)):
                    projection(f"{prefix}.feed_forward.w{w}", rows, cols, block)
                if kind == "layers":
                    for name, rows in (("attention.qkv", 24), ("attention.out", 8), ("adaLN_modulation.0", 32)):
                        projection(f"{prefix}.{name}", rows, 256, block)
                    for j in range(7):
                        add(f"{prefix}.norm{j}.weight", np.array([block + 1.00390625, -0.01], np.float32), "F32")
        if change:
            change(header)
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 8)
        path = self.root / "convrot.safetensors"
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
        return path


    def test_generic_exact_conversion_execution_and_lifetime(self):
        for slots in (1, 2, 3):
            for suffix in (0, 1):
                with self.subTest(slots=slots, suffix=suffix):
                    path = self.convrot_fixture()
                    result = subprocess.run([str(self.binary), str(path), str(slots), str(suffix)],
                                            capture_output=True, text=True, timeout=90)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("PASS generic INT8", result.stdout)

    def test_concurrent_slot_fills(self):
        result = subprocess.run([str(self.binary), str(self.convrot_fixture()), "3", "1", "2"],
            capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fp32_scale_identity_and_values(self):
        result = subprocess.run([str(self.binary), str(self.convrot_fixture()), "3", "1"],
            env=dict(os.environ, TURBOCIDER_Z_CONVROT_FP32_SCALES="1"),
            capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_scale_geometry_rejected_before_execution(self):
        path = self.convrot_fixture(lambda h: [v.update(shape=[1, v["shape"][0]])
            for k, v in h.items() if k.endswith(".weight_scale")])
        result = subprocess.run([str(self.binary), str(path), "3", "0"],
                                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("scale geometry", result.stderr)

if __name__ == "__main__":
    unittest.main(verbosity=2)
