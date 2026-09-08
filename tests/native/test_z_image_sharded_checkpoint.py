import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "turbocider_export_z_image", ROOT / "tools/coreml/export_z_image.py"
)
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


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


class ShardedCheckpointTests(unittest.TestCase):
    def make_indexed_model(self, root):
        transformer = root / "transformer"
        transformer.mkdir(parents=True)
        first = transformer / "diffusion_pytorch_model-00001-of-00002.safetensors"
        second = transformer / "diffusion_pytorch_model-00002-of-00002.safetensors"
        write_safetensors(first, {
            "layers.0.feed_forward.w1.weight": (
                "BF16", (2,), struct.pack("<HH", 0x3F80, 0x4000)
            )
        })
        write_safetensors(second, {
            "layers.0.feed_forward.w2.weight": (
                "F16", (2,), struct.pack("<HH", 0x4200, 0x4400)
            )
        })
        index = transformer / "diffusion_pytorch_model.safetensors.index.json"
        index.write_text(json.dumps({
            "metadata": {"total_size": first.stat().st_size + second.stat().st_size},
            "weight_map": {
                "layers.0.feed_forward.w1.weight": first.name,
                "layers.0.feed_forward.w2.weight": second.name,
            },
        }))
        (transformer / "config.json").write_text("{}\n")
        return transformer, index, first, second

    def test_index_routes_tensors_to_their_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transformer, index, first, second = self.make_indexed_model(root)
            source = EXPORTER.SafetensorsSource.from_model(root)
            self.assertEqual(source.checkpoint, index.resolve())
            self.assertEqual(source.shards, [first.resolve(), second.resolve()])
            with source as reader:
                self.assertEqual(
                    reader.tensor_path("layers.0.feed_forward.w1.weight"), first.resolve()
                )
                self.assertEqual(
                    reader.tensor_path("layers.0.feed_forward.w2.weight"), second.resolve()
                )
                self.assertIn("layers.0.feed_forward.w1.weight", reader._headers[first.resolve()])
                self.assertIn("layers.0.feed_forward.w2.weight", reader._headers[second.resolve()])
                if np is not None:
                    np.testing.assert_array_equal(
                        reader.tensor("layers.0.feed_forward.w1.weight", (2,), np),
                        np.array([1.0, 2.0], dtype=np.float16),
                    )
                    np.testing.assert_array_equal(
                        reader.tensor("layers.0.feed_forward.w2.weight", (2,), np),
                        np.array([3.0, 4.0], dtype=np.float16),
                    )

    def test_provenance_covers_index_and_all_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _, second = self.make_indexed_model(root)
            source = EXPORTER.SafetensorsSource.from_model(root)
            provenance = source.provenance()
            self.assertEqual(provenance["checkpoint"], str(source.checkpoint))
            self.assertEqual(provenance["checkpoint_bytes"], source.checkpoint.stat().st_size)
            self.assertEqual(set(provenance["checkpoint_shards"]), {
                "diffusion_pytorch_model-00001-of-00002.safetensors",
                "diffusion_pytorch_model-00002-of-00002.safetensors",
            })
            source.validate_unchanged(provenance)
            second.write_bytes(second.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint shard changed during export"):
                source.validate_unchanged(provenance)

    def test_index_rejects_escape_and_missing_shards(self):
        for shard_name, message in [
            ("../outside.safetensors", "sibling filename"),
            ("missing.safetensors", "missing or invalid"),
        ]:
            with self.subTest(shard_name=shard_name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                transformer = root / "transformer"
                transformer.mkdir()
                (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
                    json.dumps({"weight_map": {"tensor": shard_name}})
                )
                with self.assertRaisesRegex(ValueError, message):
                    EXPORTER.SafetensorsSource.from_model(root)

    def test_single_file_layout_remains_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "split_files/diffusion_models/z_image_turbo_bf16.safetensors"
            checkpoint.parent.mkdir(parents=True)
            write_safetensors(checkpoint, {
                "tensor": ("F16", (1,), struct.pack("<H", 0x3C00))
            })
            source = EXPORTER.SafetensorsSource.from_model(root)
            self.assertEqual(source.checkpoint, checkpoint.resolve())
            with source as reader:
                self.assertEqual(reader.tensor_path("tensor"), checkpoint.resolve())
                if np is not None:
                    self.assertEqual(float(reader.tensor("tensor", (1,), np)[0]), 1.0)

    def test_official_diffusers_f32_shards_are_exportable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transformer = root / "transformer"
            transformer.mkdir()
            shard = transformer / "diffusion_pytorch_model-00001-of-00001.safetensors"
            write_safetensors(shard, {
                "layers.0.feed_forward.w1.weight": (
                    "F32", (2,), struct.pack("<ff", 1.5, -2.25)
                )
            })
            (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
                json.dumps({
                    "weight_map": {
                        "layers.0.feed_forward.w1.weight": shard.name,
                    }
                })
            )
            source = EXPORTER.SafetensorsSource.from_model(root)
            with source as reader:
                if np is not None:
                    np.testing.assert_array_equal(
                        reader.tensor("layers.0.feed_forward.w1.weight", (2,), np),
                        np.array([1.5, -2.25], dtype=np.float16),
                    )

    def test_convrot_int8_can_stay_rotated_or_derotate_offline(self):
        if np is None:
            self.skipTest("numpy unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "z_image_turbo_int8_convrot.safetensors"
            quantized = np.array([
                [-4, -2, 0, 2],
                [1, 3, 5, 7],
            ], dtype=np.int8)
            scales = np.array([[0.5], [0.25]], dtype=np.float32)
            marker = json.dumps({
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": 4,
            }).encode()
            name = "layers.0.feed_forward.w1"
            write_safetensors(checkpoint, {
                name + ".weight": ("I8", quantized.shape, quantized.tobytes()),
                name + ".weight_scale": ("F32", scales.shape, scales.tobytes()),
                name + ".comfy_quant": ("U8", (len(marker),), marker),
            })
            expected_rotated = (quantized.astype(np.float32) * scales).astype(np.float16)

            native = EXPORTER.SafetensorsSource.from_model(checkpoint, convrot_mode="native")
            self.assertTrue(native.has_convrot)
            with native as reader:
                self.assertEqual(reader.convrot_group(name + ".weight", np), 4)
                np.testing.assert_array_equal(
                    reader.tensor(name + ".weight", quantized.shape, np),
                    expected_rotated,
                )

            derotated = EXPORTER.SafetensorsSource.from_model(
                checkpoint, convrot_mode="derotate"
            )
            with derotated as reader:
                expected = EXPORTER.rotate_convrot_matrix(
                    expected_rotated.astype(np.float32), 4, np
                ).astype(np.float16)
                np.testing.assert_array_equal(
                    reader.tensor(name + ".weight", quantized.shape, np), expected
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
