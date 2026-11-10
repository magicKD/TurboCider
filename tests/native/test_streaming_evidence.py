#!/usr/bin/env python3
"""File and serialization contracts shared by streaming evidence tools."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))

import build_streaming_catalog as builder
import prepare_streaming_release_policies as preparer
import run_streaming_campaign as runner
import verify_streaming_campaign as verifier

SERIALIZERS = (builder, preparer, runner)
FILE_HASHERS = (*SERIALIZERS, verifier)


class StreamingEvidenceTests(unittest.TestCase):
    def test_canonical_json_sorts_nested_keys_and_preserves_utf8(self):
        value = {"z": "雪", "a": [3, {"x": False}]}
        expected = '{"a":[3,{"x":false}],"z":"雪"}'.encode("utf-8")
        for owner in SERIALIZERS:
            with self.subTest(owner=owner.__name__):
                self.assertEqual(owner.canonical_json(value), expected)

    def test_canonical_json_preserves_existing_nonfinite_encoding(self):
        value = {"c": -float("inf"), "a": float("nan"), "b": float("inf")}
        for owner in SERIALIZERS:
            with self.subTest(owner=owner.__name__):
                self.assertEqual(
                    owner.canonical_json(value),
                    b'{"a":NaN,"b":Infinity,"c":-Infinity}',
                )

    def test_canonical_json_rejects_unsupported_values_and_mixed_keys(self):
        for owner in SERIALIZERS:
            for value in ({"value": object()}, {"text": 1, 2: "number"}):
                with self.subTest(owner=owner.__name__, value=value):
                    with self.assertRaises(TypeError):
                        owner.canonical_json(value)

    def test_byte_hash_matches_known_vectors(self):
        vectors = (
            (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
            (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        )
        for owner in SERIALIZERS:
            for value, expected in vectors:
                with self.subTest(owner=owner.__name__, value=value):
                    self.assertEqual(owner.sha256_bytes(value), expected)

    def test_file_hash_handles_empty_and_chunk_boundary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.bin"
            for size in (0, (1 << 20) - 1, 1 << 20, (1 << 20) + 1):
                value = (bytes(range(256)) * ((size + 255) // 256))[:size]
                path.write_bytes(value)
                for owner in FILE_HASHERS:
                    with self.subTest(owner=owner.__name__, size=size):
                        self.assertEqual(owner.sha256_file(path), hashlib.sha256(value).hexdigest())

    def test_file_hash_follows_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "source.bin"
            target.write_bytes(b"abc")
            link = Path(directory) / "source-link.bin"
            link.symlink_to(target)
            for owner in FILE_HASHERS:
                with self.subTest(owner=owner.__name__):
                    self.assertEqual(
                        owner.sha256_file(link),
                        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                    )

    def test_file_hash_preserves_missing_file_and_directory_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for owner in FILE_HASHERS:
                with self.subTest(owner=owner.__name__):
                    with self.assertRaises(FileNotFoundError):
                        owner.sha256_file(root / "missing.bin")
                    with self.assertRaises(IsADirectoryError):
                        owner.sha256_file(root)


if __name__ == "__main__":
    unittest.main()
