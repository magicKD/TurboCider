"""Exercise real native buffer reuse and reject malformed streaming checkpoints."""
import json
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

    def test_compact_suffix_values_bytes_reuse_and_cancel(self):
        result = self.run_probe(self.suffix_fixture(), "--suffix")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS: compact suffix", result.stdout)

    def test_cancel_during_suffix_preparation_then_retry(self):
        result = self.run_probe(self.suffix_fixture(), "--cancel-pack")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS: compact suffix", result.stdout)

    def test_suffix_rejects_incompatible_geometry(self):
        path = self.suffix_fixture(lambda h: h["noise_refiner.0.feed_forward.w2.weight"].update(shape=[3, 8]))
        result = self.run_probe(path, "--suffix")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("suffix geometry", result.stderr)

    def test_suffix_requires_all_hybrid_weights(self):
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
