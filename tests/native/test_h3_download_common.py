"""Characterize H3 download entry points without network or weights."""
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
    default_retries = 5
    timeout = 180
    skips_existing = False
    progress_at_boundary = True
    hash_chunk_bytes = 16 * 1024 * 1024

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
        if not self.skips_existing:
            self.target.write_bytes(b"previous complete file")
        self.part.write_bytes(b"partial replacement")
        with self.assertRaises(urllib.error.URLError):
            self.download([urllib.error.URLError("offline") for _ in range(2)])
        if self.skips_existing:
            self.assertFalse(self.target.exists())
        else:
            self.assertEqual(self.target.read_bytes(), b"previous complete file")
        self.assertEqual(self.part.read_bytes(), b"partial replacement")

    def test_default_timeout_and_retry_budget_are_preserved(self):
        self.part.write_bytes(b"abc")
        with mock.patch("urllib.request.urlopen", side_effect=TimeoutError("offline")) as request:
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(TimeoutError):
                    self.module.download_file(self.url, self.target)
        self.assertEqual(request.call_count, self.default_retries)
        for call in request.call_args_list:
            self.assertEqual(call.kwargs["timeout"], self.timeout)
            self.assertEqual(call.args[0].get_header("Range"), "bytes=3-")
        self.assertEqual(self.part.read_bytes(), b"abc")
        self.assertFalse(self.target.exists())

    def test_existing_target_policy_is_preserved(self):
        self.target.write_bytes(b"previous complete file")

        def open_response(*args, **kwargs):
            self.assertEqual(self.target.read_bytes(), b"previous complete file")
            return Response(b"replacement")

        request = self.download(open_response)
        if self.skips_existing:
            request.assert_not_called()
            self.assertEqual(self.target.read_bytes(), b"previous complete file")
        else:
            self.assertEqual(self.target.read_bytes(), b"replacement")
        self.assertFalse(self.part.exists())

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
        self.assertIn("--source" if self.skips_existing else "--verify-only", result.stdout)


    def test_progress_boundary_is_preserved(self):
        for elapsed in (10.0, 10.1):
            self.target.unlink(missing_ok=True)
            output = io.StringIO()
            with mock.patch("urllib.request.urlopen", return_value=Response(b"abc")), \
                    mock.patch("time.monotonic", side_effect=[0.0, elapsed]), \
                    contextlib.redirect_stdout(output):
                self.module.download_file(self.url, self.target)
            self.assertEqual(f"  {self.target.name}:" in output.getvalue(),
                             elapsed > 10 or self.progress_at_boundary)
            self.assertEqual(self.target.read_bytes(), b"abc")

    def test_completion_and_skip_messages_are_preserved(self):
        output = io.StringIO()
        with mock.patch("urllib.request.urlopen", return_value=Response(b"abc")), \
                mock.patch("time.monotonic", return_value=0), contextlib.redirect_stdout(output):
            self.module.download_file(self.url, self.target)
        self.assertEqual(output.getvalue(), f"[done] {self.target}\n" if self.skips_existing else "")
        if self.skips_existing:
            output = io.StringIO()
            with mock.patch("urllib.request.urlopen") as request, contextlib.redirect_stdout(output):
                self.module.download_file(self.url, self.target)
            request.assert_not_called()
            self.assertEqual(output.getvalue(), f"[skip] {self.target}\n")

    def test_zero_retry_budget_does_not_publish_or_report_completion(self):
        output = io.StringIO()
        with mock.patch("urllib.request.urlopen") as request, contextlib.redirect_stdout(output):
            self.module.download_file(self.url, self.target, retries=0)
        request.assert_not_called()
        self.assertEqual(output.getvalue(), "")
        self.assertFalse(self.target.exists())

    def test_hash_read_granularity_and_missing_file_behavior(self):
        class Reader(io.BytesIO):
            def read(stream, size=-1):
                sizes.append(size)
                return super().read(size)

        sizes = []
        path = mock.Mock(spec=Path)
        path.open.return_value = Reader(b"abc")
        self.assertEqual(self.module.sha256_file(path),
                         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        self.assertEqual(sizes, [self.hash_chunk_bytes, self.hash_chunk_bytes])
        with self.assertRaises(FileNotFoundError):
            self.module.sha256_file(self.target)


class H3VaeDownloadTests(DownloadBehavior, unittest.TestCase):
    filename = "download_h3_vae_modelscope.py"


class H3TransformerDownloadTests(DownloadBehavior, unittest.TestCase):
    filename = "download_h3_fl2va_modelscope.py"


class VdnDownloadTests(DownloadBehavior, unittest.TestCase):
    filename = "download_vdn_modelscope.py"
    default_retries = 4
    timeout = 120


class FastH3DownloadTests(DownloadBehavior, unittest.TestCase):
    filename = "prepare_fasth3_mlx.py"
    default_retries = 3
    timeout = 90
    skips_existing = True
    progress_at_boundary = False
    hash_chunk_bytes = 8 * 1024 * 1024


class HashOnlyEntryPointTests(unittest.TestCase):
    def test_fast_downloader_and_vdn_preparer_keep_digest_and_read_granularity(self):
        for name in ("download_fasth3_modelscope.py", "prepare_vdn_h3_mlx.py"):
            with self.subTest(script=name):
                spec = importlib.util.spec_from_file_location(Path(name).stem, ROOT / "tools/h3" / name)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sizes = []

                class Reader(io.BytesIO):
                    def read(stream, size=-1):
                        sizes.append(size)
                        return super().read(size)

                path = mock.Mock(spec=Path)
                path.open.return_value = Reader(b"abc")
                self.assertEqual(module.sha256_file(path),
                                 "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
                self.assertEqual(sizes, [16 * 1024 * 1024, 16 * 1024 * 1024])


if __name__ == "__main__":
    unittest.main()
