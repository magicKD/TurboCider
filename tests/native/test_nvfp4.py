"""NVFP4 format guards and optional GPU roundtrip with tiny synthetic weights.

GPU test: TURBOCIDER_TEST_GPU=1 Python/bin/python3 tests/native/test_nvfp4.py
Requires build/native/nvfp4-probe (tools/native/nvfp4_probe.cpp).
"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class NVFP4Tests(unittest.TestCase):
    def test_explicit_precision_and_dense_graph_guard(self):
        source = (ROOT / "native/models/z_image/z_image.cpp").read_text()
        self.assertIn('nvfp4_weights_bf16_activations', source)
        self.assertIn('!nvfp4_transformer_', source)
        self.assertIn('!w.nvfp4(prefix + ".feed_forward.w3")', source)
        self.assertIn('NVFP4 currently supports GPU without LoRA/ANE only', source)
        self.assertIn('nvfp4_transformer_ = std::filesystem::is_regular_file(comfy_transformer)', source)

    @unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU") == "1", "explicit Metal GPU test opt-in required")
    def test_layout_and_invalid_storage(self):
        import numpy as np
        from safetensors.numpy import save_file, load_file
        probe = ROOT / "build/native/nvfp4-probe"
        self.assertTrue(probe.is_file(), "Build tools/native/nvfp4_probe.cpp first")
        prefix = "layers.0.attention.qkv"
        packed = np.random.default_rng(42).integers(0, 256, (128, 1920), dtype=np.uint8)
        scales = (np.arange(128 * 240, dtype=np.uint32) % 8 + 48).astype(np.uint8).reshape(128, 240)
        weights = {prefix + ".weight": packed, prefix + ".weight_scale": scales,
                   prefix + ".weight_scale_2": np.array(.125, dtype=np.float32)}
        with tempfile.TemporaryDirectory(prefix="tc-nvfp4-test-") as directory:
            model, output = Path(directory) / "model.safetensors", Path(directory) / "decoded.safetensors"
            save_file(weights, model)
            result = subprocess.run([str(probe), str(model), str(output)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            row, col = np.arange(128)[:, None], np.arange(240)[None, :]
            address = (((col // 4 * 32 + row % 32) * 4 + row // 32) * 4 + col % 4)
            reordered = scales.reshape(-1)[address]
            # Codes 48..55 are E4M3 values .5, .5625, ..., .9375.
            real_scales = (.5 + (reordered.astype(np.float32) - 48) / 16) * .125
            digits = np.stack((packed >> 4, packed & 15), axis=-1).reshape(128, 3840)
            lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
            expected = lut[digits] * np.repeat(real_scales, 16, axis=1)
            np.testing.assert_array_equal(load_file(output)["weight"], expected)
            for bad in (np.array(-1, dtype=np.float32), np.array(float("nan"), dtype=np.float32)):
                weights[prefix + ".weight_scale_2"] = bad
                save_file(weights, model)
                result = subprocess.run([str(probe), str(model), str(output)], text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("invalid Comfy NVFP4 global scale", result.stderr)
            weights[prefix + ".weight_scale_2"] = np.array(1, dtype=np.float32)
            weights[prefix + ".weight_scale"] = scales[:, :-1].copy()
            save_file(weights, model)
            result = subprocess.run([str(probe), str(model), str(output)], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invalid tiled Comfy NVFP4 scales", result.stderr)


if __name__ == "__main__":
    unittest.main()
