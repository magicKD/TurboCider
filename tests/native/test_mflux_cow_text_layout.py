import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from mflux_cow_text_layout import compact_header, rewrite_cow
from mflux_original_layout import read_header


class CowReferenceLayoutTests(unittest.TestCase):
    def test_only_unconsumed_names_are_shortened_and_no_tensor_is_removed(self):
        original = {"model.norm.weight": dict(dtype="BF16", shape=[2], data_offsets=[0, 4]),
                    "model.visual.blocks.0.attn.projection.weight": dict(dtype="BF16", shape=[2], data_offsets=[4, 8])}
        mapped, bindings = compact_header(original)
        self.assertEqual(len(mapped) - 1, 2)
        self.assertEqual(mapped["model.language_model.norm.weight"], original["model.norm.weight"])
        self.assertEqual(mapped["__tc_unused_1"], original["model.visual.blocks.0.attn.projection.weight"])
        self.assertEqual([item["reference_consumed"] for item in bindings], [True, False])

    def test_real_private_clone_header_change_keeps_original_and_full_payload(self):
        with tempfile.TemporaryDirectory(prefix="tc-cow-reference-") as raw:
            source, destination = Path(raw) / "source.safetensors", Path(raw) / "derived.safetensors"
            header = {"model.norm.weight": dict(dtype="BF16", shape=[3], data_offsets=[0, 6])}
            encoded = json.dumps(header).encode().ljust(1024, b" ")
            source.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\x00\x00\x80\x3f\x00\x80")
            original, before = source.read_bytes(), source.stat()
            result = rewrite_cow(source, destination)
            self.assertIs(result["full_payload_equal"], True)
            self.assertNotEqual(source.stat().st_ino, destination.stat().st_ino)
            self.assertEqual(source.stat().st_ctime_ns, before.st_ctime_ns)
            self.assertEqual(source.read_bytes(), original)
            mapped, prefix = read_header(destination)
            self.assertEqual(len(prefix), 1032)
            self.assertEqual(destination.read_bytes()[len(prefix):], original[len(prefix):])
            self.assertIn("model.language_model.norm.weight", mapped)
            with self.assertRaises(ValueError):
                rewrite_cow(source, destination)

    def test_header_cannot_move_payload_to_fit_more_names(self):
        with tempfile.TemporaryDirectory(prefix="tc-cow-header-") as raw:
            source, destination = Path(raw) / "source.safetensors", Path(raw) / "derived.safetensors"
            header = {"model.norm.weight": dict(dtype="BF16", shape=[3], data_offsets=[0, 6])}
            encoded = json.dumps(header, separators=(",", ":")).encode()
            source.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"abcdef")
            with self.assertRaises(ValueError):
                rewrite_cow(source, destination)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
