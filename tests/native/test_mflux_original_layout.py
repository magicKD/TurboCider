import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from mflux_original_layout import mapped_header, qwen_vae_key, read_header, rewrite, z_vae_key


class ReferenceLayoutTests(unittest.TestCase):
    def test_qwen_language_prefix_does_not_change_vision_or_dtype(self):
        header = {"model.layers.0.mlp.up_proj.weight": dict(dtype="BF16", shape=[2, 3], data_offsets=[0, 12]),
                  "visual.weight": dict(dtype="F32", shape=[1], data_offsets=[12, 16])}
        mapped, bindings = mapped_header(header, "qwen-text")
        self.assertEqual(mapped["model.language_model.layers.0.mlp.up_proj.weight"], header["model.layers.0.mlp.up_proj.weight"])
        self.assertEqual(mapped["visual.weight"], header["visual.weight"])
        self.assertEqual(len(bindings), 2)

    def test_z_qkv_is_q_k_v_contiguous_ranges_not_requantized(self):
        mapped, _ = mapped_header({"layers.0.attention.qkv.weight": dict(dtype="BF16", shape=[6, 2], data_offsets=[0, 24])}, "z-dit")
        for index, part in enumerate(("q", "k", "v")):
            self.assertEqual(mapped[f"layers.0.attention.to_{part}.weight"],
                             dict(dtype="BF16", shape=[2, 2], data_offsets=[index * 8, (index + 1) * 8]))

    def test_singleton_reshape_keeps_all_values_and_rejects_temporal_discard(self):
        header = {"decoder.conv1.weight": dict(dtype="BF16", shape=[2, 3, 1, 2, 2], data_offsets=[0, 48])}
        mapped, _ = mapped_header(header, "qwen-vae")
        self.assertEqual(mapped["decoder.conv_in.weight"]["shape"], [2, 3, 2, 2])
        header["decoder.conv1.weight"].update(shape=[2, 3, 2, 2, 2], data_offsets=[0, 96])
        with self.assertRaises(ValueError):
            mapped_header(header, "qwen-vae")

    def test_reverse_vae_mappings_keep_exact_block_order(self):
        self.assertEqual(z_vae_key("decoder.up.3.block.1.nin_shortcut.weight"), "decoder.up_blocks.0.resnets.1.conv_shortcut.weight")
        self.assertEqual(z_vae_key("encoder.down.2.downsample.conv.weight"), "encoder.down_blocks.2.downsamplers.0.conv.weight")
        self.assertEqual(qwen_vae_key("decoder.upsamples.2.upsamples.1.residual.6.weight"), "decoder.up_blocks.2.resnets.1.conv2.weight")

    def test_full_written_payload_original_identity_and_exclusive_target(self):
        with tempfile.TemporaryDirectory(prefix="tc-reference-layout-") as raw:
            source, target = Path(raw) / "original.safetensors", Path(raw) / "derived.safetensors"
            header = {"model.norm.weight": dict(dtype="BF16", shape=[3], data_offsets=[0, 6])}
            encoded = json.dumps(header).encode()
            source.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\x00\x00\x80\x3f\x00\x80")
            original, before = source.read_bytes(), source.stat()
            record = rewrite(source, target, "qwen-text")
            self.assertIs(record["full_payload_equal"], True)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(source.stat().st_ctime_ns, before.st_ctime_ns)
            mapped, prefix = read_header(target)
            self.assertIn("model.language_model.norm.weight", mapped)
            self.assertEqual(target.read_bytes()[len(prefix):], b"\x00\x00\x80\x3f\x00\x80")
            with self.assertRaises(FileExistsError):
                rewrite(source, target, "qwen-text")

    def test_inconsistent_extent_cannot_be_promoted_as_lossless(self):
        for value in (dict(dtype="BF16", shape=[2], data_offsets=[0, 3]),
                      dict(dtype="BF16", shape=[2], data_offsets=[1, 5])):
            with self.subTest(value=value), self.assertRaises(ValueError):
                mapped_header({"weight": value}, "identity")


if __name__ == "__main__":
    unittest.main()
