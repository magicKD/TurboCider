import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/h3/prepare_vdn_h3_mlx.py"
BASE = ROOT / "models/MiniMax-H3-ModelScope/FL2VA/transformer"
ADAPTER_SPEC = (
    ROOT / "models/VDN-H3-ModelScope/stage-dmd-step-250/adapters/turbo/adapter_spec.json"
)


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_vdn_h3_mlx", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PrepareVdnH3MlxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_all_535_base_keys_map_exactly_without_collisions(self):
        config = self.module.normalized_h3_config(
            json.loads((BASE / "config.json").read_text())
        )
        weight_map = json.loads(
            (BASE / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.assertEqual(len(weight_map), 535)
        mapped = self.module.validate_source_mapping(weight_map, config)
        self.assertEqual(len(mapped), 638)

    def test_all_363_adapter_targets_are_reachable_once(self):
        config = self.module.normalized_h3_config(
            json.loads((BASE / "config.json").read_text())
        )
        weight_map = json.loads(
            (BASE / "model.safetensors.index.json").read_text()
        )["weight_map"]
        mapped = self.module.validate_source_mapping(weight_map, config)
        targets = json.loads(ADAPTER_SPEC.read_text())["config"]["targets"]
        self.assertEqual(len(targets), 363)
        self.module.validate_adapter_target_coverage(mapped, targets)

    def test_per_head_qkv_split_preserves_q_k_v_head_order(self):
        source = np.arange(12 * 3, dtype=np.float32).reshape(12, 3)
        entries = self.module.mapped_h3_tensor_entries(
            "blocks.0.attn.qkv_proj.weight",
            source,
            num_heads=2,
            head_dim=2,
            concatenate=np.concatenate,
        )
        self.assertEqual(
            [name for name, _ in entries],
            [
                "transformer_blocks.0.attn.to_q.weight",
                "transformer_blocks.0.attn.to_k.weight",
                "transformer_blocks.0.attn.to_v.weight",
            ],
        )
        np.testing.assert_array_equal(entries[0][1], source[[0, 1, 6, 7]])
        np.testing.assert_array_equal(entries[1][1], source[[2, 3, 8, 9]])
        np.testing.assert_array_equal(entries[2][1], source[[4, 5, 10, 11]])

    def test_ffn_gate_value_halves_are_swapped_to_value_gate(self):
        source = np.arange(8, dtype=np.float32).reshape(8, 1)
        entries = self.module.mapped_h3_tensor_entries(
            "blocks.0.mlp.fc1.weight",
            source,
            num_heads=2,
            head_dim=2,
            concatenate=np.concatenate,
        )
        self.assertEqual(entries[0][0], "transformer_blocks.0.ff.net.0.proj.weight")
        np.testing.assert_array_equal(
            entries[0][1].reshape(-1), np.array([4, 5, 6, 7, 0, 1, 2, 3])
        )

    def test_converter_uses_the_python_mlx_032_load_api(self):
        source = SCRIPT.read_text()
        self.assertIn("adapter_arrays = mx.load(str(adapter_path))", source)
        self.assertNotIn("mx.load_safetensors", source)


if __name__ == "__main__":
    unittest.main()
