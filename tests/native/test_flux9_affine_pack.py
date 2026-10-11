from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from flux9_affine_pack import mirrored_header, native_key


class FluxMatchedPackTests(unittest.TestCase):
    def test_quantized_planes_follow_exact_same_projection_alias(self):
        inverse = {"attention.to_out.weight": "transformer_blocks.0.attn.to_out.0.weight"}
        self.assertEqual(native_key("attention.to_out.scales", inverse), "transformer_blocks.0.attn.to_out.0.scales")
        self.assertEqual(native_key("attention.to_out.biases", inverse), "transformer_blocks.0.attn.to_out.0.biases")
        with self.assertRaises(ValueError):
            native_key("random.weight", inverse)

    def test_native_header_does_not_change_any_dtype_shape_or_payload_range(self):
        original = {"attention.to_out.weight": dict(dtype="U32", shape=[2, 8], data_offsets=[0, 64]),
                    "attention.to_out.scales": dict(dtype="BF16", shape=[2, 2], data_offsets=[64, 72])}
        inverse = {"attention.to_out.weight": "transformer_blocks.0.attn.to_out.0.weight"}
        actual = mirrored_header(original, inverse)
        self.assertEqual(actual["transformer_blocks.0.attn.to_out.0.weight"], original["attention.to_out.weight"])
        self.assertEqual(actual["transformer_blocks.0.attn.to_out.0.scales"], original["attention.to_out.scales"])


if __name__ == "__main__":
    unittest.main()
