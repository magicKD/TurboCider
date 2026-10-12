import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/coreml"))
from flux_checkpoint import Checkpoint, geometry


def safetensors(path, name, values):
    words = (values.astype(np.float32).view(np.uint32) >> 16).astype("<u2")
    header = json.dumps({name: dict(dtype="BF16", shape=list(values.shape), data_offsets=[0, words.nbytes])}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + words.tobytes())


class FluxCoremlCheckpointTests(unittest.TestCase):
    def test_only_original4b_and9b_geometry(self):
        four = geometry(dict(num_single_layers=20, num_attention_heads=24, attention_head_dim=128))
        nine = geometry(dict(num_single_layers=24, num_attention_heads=32, attention_head_dim=128, mlp_ratio=3.0))
        self.assertEqual((four["hidden"], four["mlp_width"]), (3072, 9216))
        self.assertEqual((nine["hidden"], nine["mlp_width"], nine["single_layers"]), (4096, 12288, 24))
        with self.assertRaises(ValueError):
            geometry(dict(num_single_layers=20, num_attention_heads=32, attention_head_dim=128))

    def test_complete_index_identity_and_bounded_tensor_slices_without_flat_duplicate(self):
        with tempfile.TemporaryDirectory(prefix="tc-flux-shard-reader-") as raw:
            model = Path(raw)
            transformer = model / "transformer"
            transformer.mkdir()
            values = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
            safetensors(transformer / "part1.safetensors", "first.weight", values)
            safetensors(transformer / "part2.safetensors", "second.weight", values + 20)
            (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(
                dict(weight_map={"first.weight": "part1.safetensors", "second.weight": "part2.safetensors"})))
            checkpoint = Checkpoint(model)
            identity = checkpoint.identity()
            self.assertEqual(set(identity["checkpoint_shards"]), {"part1.safetensors", "part2.safetensors"})
            with checkpoint as reader:
                np.testing.assert_array_equal(reader.tensor("first.weight", (4, 4), np, rows=(1, 3)), values[1:3].astype(np.float16))
                np.testing.assert_array_equal(reader.tensor("second.weight", (4, 4), np, columns=(1, 3)), (values + 20)[:,1:3].astype(np.float16))
            self.assertFalse((transformer / "diffusion_pytorch_model.safetensors").exists())

    def test_missing_wrong_or_path_traversal_shard_binding_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="tc-flux-index-gate-") as raw:
            model = Path(raw)
            root = model / "transformer"
            root.mkdir()
            safetensors(root / "part.safetensors", "present.weight", np.ones((2, 2), np.float32))
            index = root / "diffusion_pytorch_model.safetensors.index.json"
            index.write_text(json.dumps(dict(weight_map={"missing.weight": "part.safetensors"})))
            with self.assertRaises(ValueError):
                with Checkpoint(model):
                    pass
            index.write_text(json.dumps(dict(weight_map={"present.weight": "../part.safetensors"})))
            with self.assertRaises(ValueError):
                Checkpoint(model)

    def test_single_file_and_source_mutation_are_not_hidden(self):
        with tempfile.TemporaryDirectory(prefix="tc-flux-single-reader-") as raw:
            model = Path(raw)
            root = model / "transformer"
            root.mkdir()
            file = root / "diffusion_pytorch_model.safetensors"
            safetensors(file, "weight", np.ones((2, 2), np.float32))
            reader = Checkpoint(model)
            self.assertNotIn("checkpoint_shards", reader.identity())
            safetensors(file, "weight", np.full((2, 2), 2, np.float32))
            with self.assertRaises(ValueError):
                reader.check()


if __name__ == "__main__":
    unittest.main()
