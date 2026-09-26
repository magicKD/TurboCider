"""Offline fused checkpoint mapping; no Core ML or model assets required."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("qwen_export", ROOT / "tools/coreml/export_qwen3.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class Source:
    def __init__(self, values):
        self.values = values

    def tensor(self, key, shape, numpy, mlx):
        value = self.values[key]
        if value.shape != shape:
            raise ValueError("shape mismatch")
        return value


class Qwen21ExportTests(unittest.TestCase):
    def test_fused_gate_up_and_down_axis(self):
        gate = np.arange(24, dtype=np.float32).reshape(6, 4)
        up = gate + 100
        down = np.arange(24, dtype=np.float32).reshape(4, 6) + 200
        source = Source({"transformer_blocks.3.img_mlp.gate_up.weight": np.concatenate([gate, up]),
                         "transformer_blocks.3.img_mlp.out.weight": down})
        actual = export.mlp_weights(source, 3, "qwen21", "ignored", 4, 6, np, None)
        for a, b in zip(actual, (gate, up, down)):
            np.testing.assert_array_equal(a, b)
        with self.assertRaises(ValueError):
            export.mlp_weights(source, 3, "qwen21", "ignored", 4, 8, np, None)

    def test_existing_qwen3_mapping_unchanged(self):
        values = [np.ones((6, 4)), np.full((6, 4), 2), np.full((4, 6), 3)]
        source = Source(dict(zip(["model.layers.2.mlp." + name + ".weight"
                                  for name in ("gate_proj", "up_proj", "down_proj")], values)))
        actual = export.mlp_weights(source, 2, "qwen3", "model", 4, 6, np, None)
        for a, b in zip(actual, values):
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
