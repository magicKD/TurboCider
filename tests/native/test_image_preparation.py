"""CPU-only C ABI/CLI reference preparation; no model or GPU work.

Run against the freshly built candidate with TURBOCIDER_TEST_NATIVE_DIR.
Fixtures use only the standard library, and every writable path is owned by
this test. Tests intentionally fail if the selected library lacks the new ABI.
"""
import concurrent.futures
import ctypes
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
import zlib


ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native")).resolve()
PRESETS = ("original", "automatic", "fit512", "portrait512", "landscape512")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def png_chunk(kind, payload):
    return (struct.pack(">I", len(payload)) + kind + payload +
            struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))


def rgba_png(width, height):
    """8-bit RGBA, filter zero, with transparent and opaque source pixels."""
    pair = b"\x23\x65\xc9\x00\xc8\x43\x21\xff"
    row = (pair * ((width + 1) // 2))[:width * 4]
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (PNG_SIGNATURE + png_chunk(b"IHDR", header) +
            png_chunk(b"IDAT", zlib.compress((b"\0" + row) * height)) +
            png_chunk(b"IEND", b""))


def png_dimensions(data):
    # Only inspect the standard IHDR; ImageIO/Swift owns pixel/EXIF coverage.
    if (len(data) < 33 or data[:8] != PNG_SIGNATURE or
            data[8:16] != b"\0\0\0\rIHDR"):
        raise AssertionError("preparation did not return a PNG IHDR")
    return struct.unpack(">II", data[16:24])


@unittest.skipUnless(sys.platform == "darwin", "requires the macOS native image preparation library")
class ImagePreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = ctypes.CDLL(str(NATIVE / "libturbocider.dylib"))
        pointer = ctypes.POINTER(ctypes.c_void_p)
        cls.library.tc_reference_image_render.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, pointer, pointer,
            ctypes.POINTER(ctypes.c_uint64), pointer]
        cls.library.tc_reference_image_render.restype = ctypes.c_int
        cls.library.tc_image_prepare_json.argtypes = [ctypes.c_char_p, pointer, pointer]
        cls.library.tc_image_prepare_json.restype = ctypes.c_int
        cls.library.tc_string_free.argtypes = [ctypes.c_void_p]
        cls.library.tc_string_free.restype = None
        cls.library.tc_buffer_free.argtypes = [ctypes.c_void_p]
        cls.library.tc_buffer_free.restype = None
        cls.scratch = tempfile.TemporaryDirectory(prefix="tc-image-preparation-")
        cls.addClassCleanup(cls.scratch.cleanup)
        cls.root = Path(cls.scratch.name).resolve()
        cls.sources, cls.original_bytes = {}, {}
        for name, dimensions in (("wide", (2048, 1024)), ("tall", (1024, 2048)),
                                 ("small", (64, 32))):
            path = cls.root / (name + ".png")
            data = rgba_png(*dimensions)
            path.write_bytes(data)
            cls.sources[name], cls.original_bytes[name] = path, data

    def setUp(self):
        scratch = tempfile.TemporaryDirectory(prefix="case-", dir=self.root)
        self.addCleanup(scratch.cleanup)
        self.directory = Path(scratch.name)

    def tearDown(self):
        for name, path in self.sources.items():
            self.assertEqual(path.read_bytes(), self.original_bytes[name], "source PNG was modified")

    @staticmethod
    def wire(value):
        return value if isinstance(value, bytes) or value is None else json.dumps(value, ensure_ascii=False).encode()

    def request(self, source="wide", preset="fit512", **fields):
        return {"source_path": str(self.sources[source]), "preset": preset, **fields}

    def entries(self):
        return sorted(str(path.relative_to(self.directory)) for path in self.directory.rglob("*"))

    def raw_prepare(self, value):
        output, error = ctypes.c_void_p(), ctypes.c_void_p()
        status = self.library.tc_image_prepare_json(self.wire(value), ctypes.byref(output), ctypes.byref(error))
        try:
            result = ctypes.string_at(output.value).decode("utf-8") if output.value else ""
            message = ctypes.string_at(error.value).decode("utf-8") if error.value else ""
            if status != 0 and output.value:
                raise AssertionError("failed image preparation returned a result buffer")
            return status, json.loads(result) if result else None, message
        finally:
            self.library.tc_string_free(output.value)
            self.library.tc_string_free(error.value)

    def prepare(self, value):
        status, result, message = self.raw_prepare(value)
        self.assertEqual(status, 0, message)
        self.assertFalse(message)
        self.assertIsInstance(result, dict)
        return result

    def reject(self, value):
        before = self.entries()
        status, result, message = self.raw_prepare(value)
        self.assertEqual(status, 1, result)
        self.assertIsNone(result)
        self.assertTrue(message)
        self.assertEqual(self.entries(), before, "failed preparation created or left a temporary file")

    def render(self, source, preset):
        metadata, png, error = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_void_p()
        count = ctypes.c_uint64()
        status = self.library.tc_reference_image_render(
            os.fsencode(source), preset.encode(), ctypes.byref(metadata), ctypes.byref(png),
            ctypes.byref(count), ctypes.byref(error))
        try:
            description = ctypes.string_at(metadata.value).decode() if metadata.value else ""
            message = ctypes.string_at(error.value).decode() if error.value else ""
            data = ctypes.string_at(png.value, count.value) if png.value else b""
            if status != 0 and (metadata.value or png.value or count.value):
                raise AssertionError("failed render returned metadata or PNG storage")
            return status, json.loads(description) if description else None, data, message, bool(png.value), count.value
        finally:
            self.library.tc_string_free(metadata.value)
            self.library.tc_string_free(error.value)
            self.library.tc_buffer_free(png.value)

    def assert_metadata(self, metadata, original, dimensions, changed):
        expected = {"original_width": original[0], "original_height": original[1],
                    "width": dimensions[0], "height": dimensions[1], "changed": changed}
        self.assertEqual({key: metadata[key] for key in expected}, expected)
        self.assertIs(type(metadata["changed"]), bool)
        for key in ("original_width", "original_height", "width", "height"):
            self.assertIs(type(metadata[key]), int)

    def test_resize_bounds_and_render_file_byte_equivalence(self):
        cases = (("wide", "automatic", (2048, 1024), (1024, 512)),
                 ("wide", "fit512", (2048, 1024), (512, 256)),
                 ("tall", "portrait512", (1024, 2048), (384, 768)),
                 ("wide", "landscape512", (2048, 1024), (768, 384)))
        for index, (source, preset, original, dimensions) in enumerate(cases):
            with self.subTest(preset=preset):
                status, metadata, png, error, has_png, count = self.render(self.sources[source], preset)
                self.assertEqual(status, 0, error)
                self.assertFalse(error)
                self.assertTrue(has_png)
                self.assertEqual(count, len(png))
                self.assertEqual(png_dimensions(png), dimensions)
                self.assert_metadata(metadata, original, dimensions, True)
                output = self.directory / f"prepared-{index}.png"
                result = self.prepare(self.request(source, preset, schema_version=1, output_path=str(output)))
                self.assert_metadata(result, original, dimensions, True)
                self.assertEqual(result["schema_version"], 1)
                self.assertEqual(result["source_path"], str(self.sources[source]))
                self.assertEqual(result["preset"], preset)
                self.assertEqual(result["image_path"], str(output))
                self.assertIs(result["output_created"], True)
                self.assertEqual(output.read_bytes(), png)
                self.assertEqual(self.entries(), [f"prepared-{i}.png" for i in range(index + 1)])

    def test_original_and_small_images_never_upscale_or_write_output(self):
        for source, presets, dimensions in (("small", PRESETS, (64, 32)),
                                             ("wide", ("original",), (2048, 1024))):
            for preset in presets:
                with self.subTest(source=source, preset=preset):
                    status, metadata, png, error, has_png, count = self.render(self.sources[source], preset)
                    self.assertEqual(status, 0, error)
                    self.assertFalse(error)
                    self.assertFalse(has_png)
                    self.assertEqual((png, count), (b"", 0))
                    self.assert_metadata(metadata, dimensions, dimensions, False)
                    output = self.directory / f"unused-{source}-{preset}.png"
                    result = self.prepare(self.request(source, preset, output_path=str(output)))
                    self.assert_metadata(result, dimensions, dimensions, False)
                    self.assertIs(result["output_created"], False)
                    self.assertEqual(result["image_path"], str(self.sources[source]))
                    self.assertFalse(output.exists())
        self.assertEqual(self.entries(), [])

    def test_noop_preserves_existing_file_directory_and_symlink_destinations(self):
        file = self.directory / "existing.png"
        file.write_bytes(b"do not replace me")
        directory = self.directory / "directory.png"
        directory.mkdir()
        (directory / "keep.txt").write_text("keep")
        dangling = self.directory / "dangling.png"
        dangling.symlink_to(self.directory / "absent.png")
        missing_parent = self.directory / "missing-parent" / "unused.png"
        before = self.entries()
        for output in (file, directory, dangling, missing_parent):
            identity = output.lstat().st_ino if output.exists() or output.is_symlink() else None
            result = self.prepare(self.request("small", "fit512", output_path=str(output)))
            self.assertIs(result["output_created"], False)
            self.assertEqual(result["image_path"], str(self.sources["small"]))
            if identity is not None:
                self.assertEqual(output.lstat().st_ino, identity)
        self.assertEqual(file.read_bytes(), b"do not replace me")
        self.assertEqual((directory / "keep.txt").read_text(), "keep")
        self.assertTrue(dangling.is_symlink())
        self.assertEqual(self.entries(), before)

    def test_invalid_json_fields_types_duplicates_nul_and_paths_fail_closed(self):
        base = self.request(output_path=str(self.directory / "never.png"))
        cases = [None, b"null", b"", b"not json", b"{", b" " * 1048577, [], [base], "not an object",
                 {**base, "unknown": True},
                 {key: value for key, value in base.items() if key != "source_path"},
                 {key: value for key, value in base.items() if key != "preset"},
                 self.request(),
                 {**base, "source_path": ""},
                 {**base, "source_path": "relative.png"},
                 {**base, "source_path": "/has\0nul.png"},
                 {**base, "source_path": str(self.directory / "missing.png")},
                 {**base, "preset": "fit256"}, {**base, "preset": ""},
                 {**base, "preset": "fit512\0original"},
                 {**base, "output_path": "relative.png"},
                 {**base, "output_path": ""},
                 {**base, "output_path": str(self.directory / "wrong.jpg")},
                 {**base, "output_path": str(self.directory / "nul\0.png")}]
        for key in ("source_path", "preset", "output_path"):
            cases.extend({**base, key: value} for value in (None, True, False, 1, 1.5, [], {}))
        cases.extend({**base, "schema_version": value} for value in (None, True, False, 0, 2, 1.5, "1", [], {}))
        serialized = json.dumps(base).encode()
        cases.extend((serialized[:-1] + b',"preset":"original"}',
                      serialized[:-1] + b',"\\u0070reset":"original"}',
                      serialized.replace(b"fit512", b"fit\x00512"),
                      serialized + b" trailing"))
        for index, value in enumerate(cases):
            with self.subTest(index=index, value_type=type(value).__name__):
                self.reject(value)

    def test_malformed_png_rejected_even_when_original_needs_no_resize(self):
        for index, data in enumerate((b"not a PNG", rgba_png(64, 32)[:33])):
            source = self.directory / f"broken-{index}.png"
            source.write_bytes(data)
            for preset in ("original", "fit512"):
                with self.subTest(index=index, preset=preset):
                    self.reject({"source_path": str(source), "preset": preset,
                                 "output_path": str(self.directory / "never.png")})
                    status, metadata, png, error, has_png, count = self.render(source, preset)
                    self.assertEqual(status, 1)
                    self.assertIsNone(metadata)
                    self.assertTrue(error)
                    self.assertEqual((png, has_png, count), (b"", False, 0))
            self.assertEqual(source.read_bytes(), data)

    def test_missing_parent_and_existing_destinations_are_preserved_on_failure(self):
        self.reject(self.request(output_path=str(self.directory / "missing-parent" / "output.png")))
        destinations = []
        file = self.directory / "existing.png"
        file.write_bytes(b"keep file bytes")
        destinations.append(file)
        directory = self.directory / "existing-dir.png"
        directory.mkdir()
        (directory / "keep.txt").write_text("keep child")
        destinations.append(directory)
        for name, target in (("dangling.png", self.directory / "absent.png"),
                             ("source-link.png", self.sources["wide"])):
            link = self.directory / name
            link.symlink_to(target)
            destinations.append(link)
        for output in destinations:
            with self.subTest(output=output.name):
                before = output.lstat()
                self.reject(self.request(output_path=str(output)))
                after = output.lstat()
                self.assertEqual((after.st_dev, after.st_ino), (before.st_dev, before.st_ino))
        self.assertEqual(file.read_bytes(), b"keep file bytes")
        self.assertEqual((directory / "keep.txt").read_text(), "keep child")
        self.assertEqual(os.readlink(destinations[2]), str(self.directory / "absent.png"))
        self.assertEqual(os.readlink(destinations[3]), str(self.sources["wide"]))

    def test_concurrent_same_output_publishes_once_without_temporary_residue(self):
        output = self.directory / "winner.png"
        barrier = threading.Barrier(2)
        request = self.request(output_path=str(output))

        def worker():
            barrier.wait(timeout=5)
            return self.raw_prepare(request)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as workers:
            futures = [workers.submit(worker) for _ in range(2)]
            replies = [future.result(timeout=20) for future in futures]
        self.assertEqual(sorted(reply[0] for reply in replies), [0, 1])
        for status, result, message in replies:
            if status == 0:
                self.assertIs(result["output_created"], True)
                self.assertEqual(result["image_path"], str(output))
                self.assertFalse(message)
            else:
                self.assertIsNone(result)
                self.assertTrue(message)
        self.assertEqual(png_dimensions(output.read_bytes()), (512, 256))
        status, _, expected, error, _, _ = self.render(self.sources["wide"], "fit512")
        self.assertEqual(status, 0, error)
        self.assertEqual(output.read_bytes(), expected)
        self.assertEqual(self.entries(), ["winner.png"])

    def test_cli_matches_c_api_for_resize_and_noop_and_rejects_embedded_nul(self):
        cli = NATIVE / "turbocider"
        for preset in ("fit512", "original"):
            with self.subTest(preset=preset):
                c_output = self.directory / f"c-{preset}.png"
                cli_output = self.directory / f"cli-{preset}.png"
                c_result = self.prepare(self.request(preset=preset, output_path=str(c_output)))
                cli_request = self.request(preset=preset, output_path=str(cli_output))
                envelope = self.directory / f"{preset}.json"
                envelope.write_bytes(self.wire(cli_request))
                command = [str(cli), "prepare-image", str(envelope)]
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                cli_result = json.loads(result.stdout)
                expected = {**c_result, "image_path": str(cli_output) if preset == "fit512" else c_result["image_path"]}
                self.assertEqual(cli_result, expected)
                if preset == "fit512":
                    self.assertEqual(cli_output.read_bytes(), c_output.read_bytes())
                else:
                    self.assertFalse(cli_output.exists())
                    self.assertFalse(c_output.exists())
        invalid = self.directory / "embedded-nul.json"
        invalid.write_bytes(self.wire(self.request("small", "original")) + b"\0ignored suffix")
        before = self.entries()
        result = subprocess.run([str(cli), "prepare-image", str(invalid)], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(result.stdout.strip())
        self.assertIn("NUL", result.stderr)
        self.assertEqual(self.entries(), before)

    def test_null_c_inputs_outputs_are_initialized_and_reject_without_writes(self):
        source, preset = os.fsencode(self.sources["wide"]), b"fit512"
        for missing in ("source", "preset", "metadata", "png", "count"):
            metadata, png, error = ctypes.c_void_p(1), ctypes.c_void_p(1), ctypes.c_void_p()
            count = ctypes.c_uint64(99)
            status = self.library.tc_reference_image_render(
                None if missing == "source" else source, None if missing == "preset" else preset,
                None if missing == "metadata" else ctypes.byref(metadata),
                None if missing == "png" else ctypes.byref(png),
                None if missing == "count" else ctypes.byref(count), ctypes.byref(error))
            try:
                self.assertEqual(status, 1, missing)
                self.assertTrue(error.value, missing)
                if missing != "metadata":
                    self.assertFalse(metadata.value, missing)
                if missing != "png":
                    self.assertFalse(png.value, missing)
                if missing != "count":
                    self.assertEqual(count.value, 0, missing)
            finally:
                if missing != "metadata" and metadata.value not in (None, 1):
                    self.library.tc_string_free(metadata.value)
                if missing != "png" and png.value not in (None, 1):
                    self.library.tc_buffer_free(png.value)
                self.library.tc_string_free(error.value)
        error = ctypes.c_void_p()
        self.assertEqual(self.library.tc_image_prepare_json(self.wire(self.request()), None, ctypes.byref(error)), 1)
        self.assertTrue(error.value)
        self.library.tc_string_free(error.value)
        output = ctypes.c_void_p(1)
        error = ctypes.c_void_p()
        self.assertEqual(self.library.tc_image_prepare_json(None, ctypes.byref(output), ctypes.byref(error)), 1)
        self.assertFalse(output.value)
        self.assertTrue(error.value)
        self.library.tc_string_free(error.value)
        self.assertEqual(self.library.tc_image_prepare_json(None, None, None), 1)
        self.assertEqual(self.library.tc_reference_image_render(source, preset, None, None, None, None), 1)
        self.library.tc_string_free(None)
        self.library.tc_buffer_free(None)
        self.assertEqual(self.entries(), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
