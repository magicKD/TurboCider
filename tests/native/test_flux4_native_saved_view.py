import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/validation"))
from flux4_native_saved_view import affine_header, rewrite_affine
from mflux_original_layout import read_header


class Flux4NativeSavedViewTests(unittest.TestCase):
    def header(self):
        return {"p.weight": dict(dtype="U32", shape=[2, 16], data_offsets=[0, 128]),
                "p.scales": dict(dtype="BF16", shape=[2, 2], data_offsets=[128, 136]),
                "p.biases": dict(dtype="BF16", shape=[2, 2], data_offsets=[136, 144])}

    def test_native_rename_preserves_all_affine_planes_and_declares_actual_g64(self):
        header = self.header()
        mapped = affine_header(header, {"p.weight": "native.p.weight"}, "transformer")
        self.assertEqual(mapped["__metadata__"]["tc_affine_group_size"], "64")
        self.assertEqual(mapped["__metadata__"]["tc_affine_bits"], "4")
        for key, value in header.items():
            self.assertEqual(mapped["native." + key], value)

    def test_g32_shape_missing_planes_and_unknown_mapping_are_rejected(self):
        header = self.header()
        for change in (lambda h: h["p.weight"].update(shape=[2, 8]),
                       lambda h: h.pop("p.scales")):
            malformed = copy.deepcopy(header)
            change(malformed)
            with self.assertRaises((ValueError, KeyError)):
                affine_header(malformed, {"p.weight": "native.p.weight"}, "transformer")
        with self.assertRaises(ValueError):
            affine_header(header, {}, "transformer")

    def test_stored_f32_rope_is_retained_without_inventing_a_learned_mapping(self):
        header = {"rotary_emb.inv_freq": dict(dtype="F32", shape=[64], data_offsets=[0, 256])}
        self.assertEqual(affine_header(header, {}, "text_encoder")["rotary_emb.inv_freq"], header["rotary_emb.inv_freq"])
        with self.assertRaises(ValueError):
            affine_header(header, {}, "vae")

    def test_affine_planes_may_span_actual_upstream_shards(self):
        header = self.header()
        inverse = {"p.weight": "native.p.weight"}
        for key in header:
            mapped = affine_header({key: header[key]}, inverse, "transformer", header)
            self.assertEqual(mapped["native." + key], header[key])

    def test_complete_stream_rewrite_changes_only_header_not_payload(self):
        header = self.header()
        with tempfile.TemporaryDirectory(prefix="tc-flux4-native-view-") as raw:
            source, destination = Path(raw) / "source", Path(raw) / "native"
            encoded = json.dumps(header).encode()
            payload = bytes(range(144))
            source.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
            original = source.read_bytes()
            record = rewrite_affine(source, destination, affine_header(header, {"p.weight": "native.p.weight"}, "transformer"))
            self.assertTrue(record["full_payload_equal"])
            self.assertEqual(record["payload_sha256"], hashlib.sha256(payload).hexdigest())
            _, prefix = read_header(destination)
            self.assertEqual(destination.read_bytes()[len(prefix):], payload)
            self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
