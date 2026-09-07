import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "turbocider_coreml_lora", ROOT / "tools/coreml/lora.py"
)
LORA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LORA)

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


def write_safetensors(path, tensors):
    header = {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class CoreMLLoRATests(unittest.TestCase):
    def test_z_image_bf16_delta_matches_native_semantics(self):
        if np is None:
            self.skipTest("numpy unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "adapter.safetensors"
            write_safetensors(path, {
                "diffusion_model.layers.0.feed_forward.w1.lora_A.default.weight": (
                    "BF16", (1, 2), struct.pack("<HH", 0x3F80, 0x4000)
                ),
                "diffusion_model.layers.0.feed_forward.w1.lora_B.default.weight": (
                    "BF16", (2, 1), struct.pack("<HH", 0x4040, 0x4080)
                ),
                "diffusion_model.layers.0.feed_forward.w1.alpha": (
                    "F32", (1,), struct.pack("<f", 2.0)
                ),
            })
            bundles = LORA.load([path], [0.5], ["transformer"], np)
            merged, applied = LORA.apply(
                "layers.0.feed_forward.w1.weight",
                np.zeros((2, 2), dtype=np.float16), bundles, np,
            )
            np.testing.assert_array_equal(
                merged, np.array([[3.0, 6.0], [4.0, 8.0]], dtype=np.float16)
            )
            self.assertEqual(applied, 1)
            self.assertEqual(bundles[0]["applied"], 1)

    def test_flux_fused_projection_aliases_match_native_loader(self):
        self.assertEqual(
            LORA.targets("single_blocks.7.linear1"),
            ["single_transformer_blocks.7.attn.to_qkv_mlp_proj"],
        )
        self.assertEqual(
            LORA.targets("single_blocks.7.linear2"),
            ["single_transformer_blocks.7.attn.to_out"],
        )

    def test_provenance_binds_content_role_and_strength(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "adapter.safetensors"
            path.write_bytes(b"adapter")
            record = LORA.provenance([path], [0.75], ["transformer"])[0]
            self.assertEqual(record["path"], str(path.resolve()))
            self.assertEqual(record["bytes"], 7)
            self.assertEqual(record["role"], "transformer")
            self.assertEqual(record["strength"], 0.75)
            self.assertEqual(len(record["sha256"]), 64)

    def test_exporters_and_runtime_share_lora_identity_contract(self):
        flux = (ROOT / "tools/coreml/export_flux2.py").read_text()
        z_image = (ROOT / "tools/coreml/export_z_image.py").read_text()
        coreml = (ROOT / "native/backends/coreml.mm").read_text()
        app = (ROOT / "apps/macos/StudioState.swift").read_text()
        discovery = (ROOT / "apps/macos/AccelerationDiscovery.swift").read_text()
        for source in (flux, z_image):
            self.assertIn("--lora-strength", source)
            self.assertIn("lora_records", source)
            self.assertIn("loras", source)
            self.assertRegex(source, r"[\"']schema_version[\"']\s*:\s*2")
        self.assertIn("Core ML artifact does not match the active LoRA set", coreml)
        self.assertIn("Core ML LoRA SHA-256 mismatch", coreml)
        self.assertIn("manifestBinds", app)
        self.assertIn('source["loras"]', discovery)
        self.assertIn("loraRequiresBaseGPU", app)


if __name__ == "__main__":
    unittest.main(verbosity=2)
