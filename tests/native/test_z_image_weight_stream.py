"""Exercise real native buffer reuse and reject malformed streaming checkpoints."""
import json
import ctypes as C
import os
import struct
import subprocess
import sysconfig
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU") == "1", "explicit Metal GPU test opt-in required")
class WeightStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory(prefix="tc-z-stream-")
        cls.root = Path(cls.folder.name)
        cls.binary = cls.root / "stream-test"
        mlx = Path(sysconfig.get_paths()["purelib"]) / "mlx"
        subprocess.run([
            "clang++", "-std=c++20", "-I", str(ROOT / "native"),
            "-I", str(ROOT / "native/core"), "-isystem", str(mlx / "include"),
            str(ROOT / "tests/native/z_image_weight_stream_test.cpp"),
            "-L" + str(ROOT / "build/native"), "-lturbocider",
            "-L" + str(mlx / "lib"), "-lmlx",
            "-Wl,-rpath," + str(ROOT / "build/native"), "-Wl,-rpath," + str(mlx / "lib"),
            "-o", str(cls.binary),
        ], check=True, capture_output=True)
        lib = C.CDLL(str(ROOT / "build/native/libturbocider.dylib"))
        lib.tc_system_json.restype = C.c_void_p
        lib.tc_string_free.argtypes = [C.c_void_p]
        result = lib.tc_system_json()
        try:
            cls.measured_device = json.loads(C.string_at(result))["optimization_profile"].get("z_image_int8_streaming", False)
        finally:
            lib.tc_string_free(result)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def fixture(self, change=None):
        header, payload = {}, bytearray()
        keys = ["x_embedder.weight"] + [f"layers.{i}.tensor{j:02d}.weight" for i in range(30) for j in range(13)]
        for key in keys:
            value = int(key.split(".")[1]) + 1 if key.startswith("layers.") else 1
            data = struct.pack("<f", float(value))[2:] * 4
            header[key] = {"dtype": "BF16", "shape": [2, 2], "data_offsets": [len(payload), len(payload) + len(data)]}
            payload.extend(data)
        if change:
            change(header)
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 8)
        path = self.root / "model.safetensors"
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
        return path

    def suffix_fixture(self, change=None):
        header, payload = {}, bytearray()
        tensors = [("x_embedder.weight", [2, 2], 1)]
        for prefix in ("noise_refiner", "context_refiner"):
            for block in range(2):
                for w in (1, 2, 3):
                    tensors.append((f"{prefix}.{block}.feed_forward.w{w}.weight",
                                    [4, 6] if w == 2 else [6, 4], block + 1))
        for block in range(30):
            tensors.extend((f"layers.{block}.tensor{j:02d}.weight", [2, 2], block + 1)
                           for j in range(10))
            for w in (1, 2, 3):
                tensors.append((f"layers.{block}.feed_forward.w{w}.weight",
                                [4, 6] if w == 2 else [6, 4], block + 1))
        for key, shape, base in tensors:
            data = b"".join(struct.pack("<f", float(base + i))[2:]
                            for i in range(shape[0] * shape[1]))
            header[key] = {"dtype": "BF16", "shape": shape,
                           "data_offsets": [len(payload), len(payload) + len(data)]}
            payload.extend(data)
        if change:
            change(header)
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 8)
        path = self.root / "suffix.safetensors"
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
        return path

    def run_probe(self, path, *args):
        return subprocess.run([str(self.binary), str(path), *args], capture_output=True, text=True, timeout=60)

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

    def test_convrot_gpu_values_prefetch_depths_and_full_residency(self):
        if not self.measured_device:
            self.skipTest("INT8 streaming requires the measured M5 Pro 24 GiB profile")
        path = self.convrot_fixture()
        for depth in (1, 2, 4, 8):
            for suffix in (False, True):
                with self.subTest(depth=depth, suffix=suffix):
                    result = self.run_probe(path, f"--convrot-{depth}-{'suffix' if suffix else 'full'}")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("PASS: ConvRot", result.stdout)

    def test_convrot_rejects_incompatible_scale_layout(self):
        if not self.measured_device:
            self.skipTest("INT8 streaming requires the measured M5 Pro 24 GiB profile")
        path = self.convrot_fixture(lambda h: [v.update(shape=[1, v["shape"][0]]) for k, v in h.items()
                                             if k.endswith(".weight_scale")])
        result = self.run_probe(path, "--convrot-1-full")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("scale geometry", result.stderr)

    def test_convrot_unmeasured_device_is_rejected(self):
        if self.measured_device:
            self.skipTest("negative hardware path runs on unmeasured devices; policy matrix covers it on this host")
        result = self.run_probe(self.convrot_fixture(), "--convrot-1-full")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("only enabled for Apple M5 Pro 24 GiB", result.stderr)

    def test_compact_suffix_values_bytes_reuse_and_cancel(self):
        if not self.measured_device:
            self.skipTest("suffix streaming requires the measured device profile")
        result = self.run_probe(self.suffix_fixture(), "--suffix")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS: compact suffix", result.stdout)

    def test_cancel_during_suffix_preparation_then_retry(self):
        if not self.measured_device:
            self.skipTest("suffix streaming requires the measured device profile")
        result = self.run_probe(self.suffix_fixture(), "--cancel-pack")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS: compact suffix", result.stdout)

    def test_suffix_rejects_incompatible_geometry(self):
        if not self.measured_device:
            self.skipTest("suffix streaming requires the measured device profile")
        path = self.suffix_fixture(lambda h: h["noise_refiner.0.feed_forward.w2.weight"].update(shape=[3, 8]))
        result = self.run_probe(path, "--suffix")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("suffix geometry", result.stderr)

    def test_suffix_requires_all_hybrid_weights(self):
        if not self.measured_device:
            self.skipTest("suffix streaming requires the measured device profile")
        result = self.run_probe(self.fixture(), "--suffix")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing Z-Image hybrid MLP", result.stderr)

    def test_repeated_gpu_execution_and_cancel(self):
        result = self.run_probe(self.fixture())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS:", result.stdout)

    def test_rejects_bad_metadata(self):
        cases = [
            (lambda h: h["layers.1.tensor00.weight"].update(dtype="F32"), "BF16"),
            (lambda h: h["layers.1.tensor00.weight"].update(shape=[-1, 2]), "size"),
            (lambda h: h["layers.1.tensor00.weight"].update(data_offsets=[0, 8]), "overlapping"),
            (lambda h: h["layers.1.tensor00.weight"].update(shape=[1, 4]), "matching"),
            (lambda h: h.pop("layers.1.tensor00.weight"), "noncontiguous"),
        ]
        for change, message in cases:
            with self.subTest(message=message):
                result = self.run_probe(self.fixture(change))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_rejects_truncated_payload(self):
        path = self.fixture()
        path.write_bytes(path.read_bytes()[:-1])
        result = self.run_probe(path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("truncated", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
