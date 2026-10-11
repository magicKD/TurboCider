from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from flux9_affine_pack import mirrored_header, native_key, tensor_storage_plan


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

    def test_storage_plan_counts_all_affine_planes_and_dense_embeddings(self):
        tensor = dict(dtype="BF16",shape=[8,64],data_offsets=[0,1024])
        q4 = tensor_storage_plan("attn.weight",tensor,4,"transformer")
        self.assertEqual(q4["payload_bytes"],256+64)
        self.assertTrue(q4["quantized"])
        self.assertEqual(tensor_storage_plan("attn.weight",tensor,8,"transformer")["payload_bytes"],512+64)
        embedding = tensor_storage_plan("model.embed_tokens.weight",tensor,4,"text_encoder")
        self.assertEqual(embedding["payload_bytes"],1024)
        self.assertFalse(embedding["quantized"])
        self.assertTrue(tensor_storage_plan("lm_head.weight",tensor,4,"text_encoder")["excluded"])

    def test_storage_preflight_rejects_wrong_dtype_extent_or_group_geometry(self):
        for tensor in (dict(dtype="F32",shape=[8,64],data_offsets=[0,2048]),
                dict(dtype="BF16",shape=[8,64],data_offsets=[0,1000]),
                dict(dtype="BF16",shape=[8,40],data_offsets=[0,640])):
            with self.assertRaises(ValueError):
                tensor_storage_plan("attn.weight",tensor,4,"transformer")


if __name__ == "__main__":
    unittest.main()
