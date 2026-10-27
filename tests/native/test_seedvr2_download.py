#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("download_seedvr2", ROOT / "tools/models/download_seedvr2.py")
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class DownloadTests(unittest.TestCase):
    def test_publishes_only_verified_partial(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / "model.safetensors.part"
            payload = b"verified-model"
            partial.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            with mock.patch.object(downloader, "FILES", (("model.safetensors", len(payload), digest),)):
                result = downloader.download(root, verify_only=True)
            self.assertEqual((root / "model.safetensors").read_bytes(), payload)
            self.assertFalse(partial.exists())
            self.assertEqual(result["files"][0]["sha256"], digest)
            self.assertTrue((root / "source.json").is_file())

    def test_rejects_bad_size_or_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors.part"
            path.write_bytes(b"wrong")
            for size in (1, 5):
                with self.subTest(size=size), self.assertRaises(ValueError):
                    downloader.verify(path, size, "0" * 64)
            self.assertTrue(path.exists())
            self.assertFalse(path.with_suffix("").exists())

    def test_existing_final_is_not_blindly_trusted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = root / "model.safetensors"
            final.write_bytes(b"invalid")
            with mock.patch.object(downloader, "FILES", ((final.name, 7, "0" * 64),)):
                with self.assertRaises(ValueError):
                    downloader.download(root, verify_only=True)
            self.assertEqual(final.read_bytes(), b"invalid")
            self.assertFalse((root / "source.json").exists())


if __name__ == "__main__":
    unittest.main()
