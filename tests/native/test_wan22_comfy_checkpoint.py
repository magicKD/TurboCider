"""Wan2.2 Diffusers-to-Comfy tensor key and payload conversion."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from safetensors.torch import load_file, save_file
import torch

from tools.validation.wan22_diffusers_to_comfy import convert, map_key


class Wan22CheckpointConversionTests(unittest.TestCase):
    def test_mapping_and_fail_closed_unknown_key(self):
        self.assertEqual(map_key("blocks.0.attn1.to_out.0.weight"), "blocks.0.self_attn.o.weight")
        self.assertEqual(map_key("blocks.29.attn2.norm_k.weight"), "blocks.29.cross_attn.norm_k.weight")
        self.assertEqual(map_key("blocks.4.ffn.net.0.proj.bias"), "blocks.4.ffn.0.bias")
        self.assertEqual(map_key("blocks.4.scale_shift_table"), "blocks.4.modulation")
        self.assertEqual(map_key("scale_shift_table"), "head.modulation")
        with self.assertRaises(ValueError):
            map_key("blocks.0.unrecognized.weight")

    def test_clone_preserves_source_and_values(self):
        with tempfile.TemporaryDirectory(prefix="tc-wan22-convert-") as temporary:
            root = Path(temporary)
            source, output = root / "source.safetensors", root / "output.safetensors"
            values = {
                "condition_embedder.text_embedder.linear_1.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
                "blocks.0.attn1.to_q.bias": torch.arange(4, dtype=torch.float16),
                "scale_shift_table": torch.arange(8, dtype=torch.float32).reshape(1, 2, 4),
            }
            save_file(values, str(source))
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            report = convert(source, output, before)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
            actual = load_file(str(output))
            for key, value in values.items():
                self.assertTrue(torch.equal(value, actual[map_key(key)]))
            self.assertEqual(report["tensor_count"], len(values))
            with self.assertRaises(FileExistsError):
                convert(source, output, before)
            with self.assertRaises(ValueError):
                convert(source, root / "wrong.safetensors", "0" * 64)


if __name__ == "__main__":
    unittest.main()
