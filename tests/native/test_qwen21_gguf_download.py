import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("qwen21_download", ROOT / "tools/validation/download_qwen21_gguf.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Response:
    def __init__(self, payload, status=200, headers=None):
        self.payload = payload
        self.status = status
        self.headers = headers or {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, size):
        chunk, self.payload = self.payload[:size], self.payload[size:]
        return chunk


class Qwen21GgufDownloadTests(unittest.TestCase):
    def asset(self, payload):
        return dict(repository="fixture/model", revision="pinned", filename="fixture.gguf",
                    relative_path="diffusion_models/fixture.gguf", bytes=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest())

    def test_checked_publication_and_existing_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = self.asset(b"GGUF-test-original-payload")
            with patch.object(MODULE.urllib.request, "urlopen", return_value=Response(b"GGUF-test-original-payload")):
                result = MODULE.download_asset(asset, root, ["https://fixture"])
            self.assertEqual(result["status"], "downloaded_verified")
            self.assertEqual(MODULE.download_asset(asset, root)["status"], "verified_existing")
            (root / asset["relative_path"]).write_bytes(b"bad")
            with self.assertRaises(ValueError):
                MODULE.download_asset(asset, root)

    def test_exact_range_resume_and_no_duplicate_prefix(self):
        payload = b"GGUF-resumable-original"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = self.asset(payload)
            partial = root / (asset["relative_path"] + ".part")
            partial.parent.mkdir()
            partial.write_bytes(payload[:5])
            headers = {"Content-Range": f"bytes 5-{len(payload)-1}/{len(payload)}", "Content-Length": str(len(payload)-5)}
            with patch.object(MODULE.urllib.request, "urlopen", return_value=Response(payload[5:], 206, headers)) as opened:
                MODULE.download_asset(asset, root, ["https://fixture"])
            self.assertEqual(opened.call_args.args[0].get_header("Range"), "bytes=5-")
            self.assertEqual((root / asset["relative_path"]).read_bytes(), payload)

    def test_bad_range_encoding_digest_and_alias_are_not_published(self):
        for response in (Response(b"wrong", 200), Response(b"wrong", 206, {"Content-Range": "bytes 1-5/6"}),
                         Response(b"wrong", 206, {"Content-Range": "bytes 1-5/6", "Content-Encoding": "gzip"})):
            with self.subTest(response=response), self.assertRaises(ValueError):
                MODULE.check_response(response, 1, 8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset = self.asset(b"GGUF")
            with patch.object(MODULE.urllib.request, "urlopen", return_value=Response(b"FAIL")), self.assertRaises(ValueError):
                MODULE.download_asset(asset, root, ["https://fixture"])
            self.assertFalse((root / asset["relative_path"]).exists())
            alias = root / "alias"
            alias.symlink_to(root / (asset["relative_path"] + ".part"))
            with self.assertRaises(ValueError):
                MODULE.local_size(alias)


if __name__ == "__main__":
    unittest.main()
