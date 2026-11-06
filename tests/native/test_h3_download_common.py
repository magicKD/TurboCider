"""Characterize the two H3 download entry points without network or weights."""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import urllib.error


ROOT = Path(__file__).resolve().parents[2]


class Response(io.BytesIO):
    def __init__(self, payload, status=200, content_length=None):
        super().__init__(payload)
        self.status = status
        self.headers = {
            "Content-Length": str(len(payload) if content_length is None else content_length)
        }

    def getcode(self):
        return self.status


class DownloadBehavior:
    def setUp(self):
        self.script = ROOT / "tools/h3" / self.filename
        spec = importlib.util.spec_from_file_location(self.script.stem, self.script)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.target = self.root / "component" / "weights.bin"
        self.target.parent.mkdir()
        self.part = self.target.with_name(self.target.name + ".part")
        self.url = "https://example.invalid/weights.bin"

    def download(self, responses, retries=2):
        # HTTP is the controlled boundary; assertions below cover actual files.
        with mock.patch("urllib.request.urlopen", side_effect=responses) as request:
            with contextlib.redirect_stdout(io.StringIO()):
                self.module.download_file(self.url, self.target, retries=retries)
        return request

    def test_complete_response_publishes_file_and_removes_partial(self):
        request = self.download([Response(b"abcdef")])
        self.assertEqual(self.target.read_bytes(), b"abcdef")
        self.assertFalse(self.part.exists())
        self.assertIsNone(request.call_args.args[0].get_header("Range"))

    def test_partial_download_resumes_without_repeating_prefix(self):
        self.part.write_bytes(b"abc")
        request = self.download([Response(b"def", status=206)])
        self.assertEqual(self.target.read_bytes(), b"abcdef")
        self.assertFalse(self.part.exists())
        self.assertEqual(request.call_args.args[0].get_header("Range"), "bytes=3-")

    def test_server_ignoring_range_restarts_partial(self):
        self.part.write_bytes(b"stale prefix")
        self.download([Response(b"abcdef", status=200)])
        self.assertEqual(self.target.read_bytes(), b"abcdef")
        self.assertFalse(self.part.exists())

    def test_short_response_retries_from_received_bytes(self):
        request = self.download([
            Response(b"ab", content_length=6),
            Response(b"cdef", status=206),
        ])
        self.assertEqual(self.target.read_bytes(), b"abcdef")
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args.args[0].get_header("Range"), "bytes=2-")

    def test_exhausted_retries_preserve_final_and_partial_files(self):
        self.target.write_bytes(b"previous complete file")
        self.part.write_bytes(b"partial replacement")
        with self.assertRaises(urllib.error.URLError):
            self.download([urllib.error.URLError("offline") for _ in range(2)])
        self.assertEqual(self.target.read_bytes(), b"previous complete file")
        self.assertEqual(self.part.read_bytes(), b"partial replacement")

    def test_sha256_matches_known_digest(self):
        self.target.write_bytes(b"abc")
        self.assertEqual(
            self.module.sha256_file(self.target),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )

    def test_cli_help_works_outside_repository(self):
        result = subprocess.run(
            [sys.executable, "-B", str(self.script), "--help"],
            cwd=self.root, capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--verify-only", result.stdout)


class H3VaeDownloadTests(DownloadBehavior, unittest.TestCase):
    filename = "download_h3_vae_modelscope.py"


class H3TransformerDownloadTests(DownloadBehavior, unittest.TestCase):
    filename = "download_h3_fl2va_modelscope.py"


if __name__ == "__main__":
    unittest.main()
