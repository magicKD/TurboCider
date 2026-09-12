import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/h3/prepare_fasth3_mlx.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_fasth3_mlx", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class H3MLXPrepareTests(unittest.TestCase):
    def test_validate_transformer_rejects_incomplete_snapshot(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"num_layers": 50}))
            with self.assertRaises(FileNotFoundError):
                module.validate_transformer(root)

    def test_modelscope_shards_get_a_deterministic_local_index(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({"num_layers": 50}))
            shard = root / "diffusion_pytorch_model-00001-of-00001.safetensors"
            header = json.dumps({
                "tensor.weight": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4],
                }
            }, separators=(",", ":")).encode()
            shard.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 4)
            module.validate_transformer(root)
            index = json.loads((root / module.TRANSFORMER_INDEX).read_text())
            self.assertEqual(index["weight_map"], {"tensor.weight": shard.name})
            self.assertEqual(index["metadata"]["total_size"], 4)

    def test_prepare_is_local_modelscope_only(self):
        module = load_module()
        self.assertFalse(hasattr(module, "find_hf_download"))
        self.assertFalse(hasattr(module, "download_snapshot"))
        self.assertEqual(
            module.DEFAULT_REPO,
            "FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree",
        )

    def test_output_validator_enforces_int6_g64_and_50_plus_2(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "transformer"
            source.mkdir()
            (source / module.TRANSFORMER_INDEX).write_text(json.dumps({"weight_map": {"x": "a"}}))
            (source / "config.json").write_text(json.dumps({"num_layers": 50}))
            (source / "a").write_bytes(b"x")
            (root / "checkpoint_content.json").write_text(json.dumps({
                "aggregate_sha256": "a" * 64,
            }))
            output = root / "out" / "int6"
            output.mkdir(parents=True)
            (output / "mlx_h3_dit.safetensors").write_bytes(b"x")
            (output / "mlx_h3_dit.json").write_text(json.dumps({
                "num_blocks": 50,
                "num_refiner_blocks": 2,
                "quantization": {"mode": "affine", "bits": 6, "group_size": 64},
            }))
            module.validate_output(root / "out", ["int6"], source, "repo", "rev")
            manifest = json.loads((output / "mlx_h3_dit.json").read_text())
            self.assertEqual(manifest["source"]["revision"], "rev")
            self.assertTrue(manifest["source"]["transformer_index_sha256"])
            self.assertEqual(manifest["source"]["checkpoint_sha256"], "a" * 64)
            self.assertEqual(module.DEFAULT_REPO,
                             "FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree")


if __name__ == "__main__":
    unittest.main()
