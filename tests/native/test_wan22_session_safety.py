"""Opt-in real Wan2.2 session cancellation and output-publication checks."""
import ctypes as C
import json
import os
from pathlib import Path
import tempfile
import unittest


LIBRARY = Path(os.environ.get("TURBOCIDER_NATIVE_LIBRARY", "/nonexistent")).resolve()
MODEL = Path(os.environ.get("TURBOCIDER_WAN22_MODEL", "/nonexistent")).resolve()
IMAGE = Path(os.environ.get("TURBOCIDER_WAN22_IMAGE", "/nonexistent")).resolve()
CALLBACK = C.CFUNCTYPE(None, C.c_char_p, C.c_void_p)


@unittest.skipUnless(LIBRARY.is_file() and MODEL.is_dir() and IMAGE.is_file(),
                     "native library, real Wan2.2 model and image required")
class Wan22SessionSafetyTests(unittest.TestCase):
    def setUp(self):
        self.lib = C.CDLL(str(LIBRARY))
        self.lib.tc_engine_create_model.argtypes = [C.c_char_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        self.lib.tc_engine_generate.argtypes = [C.c_void_p, C.c_char_p, CALLBACK, C.c_void_p,
                                               C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        self.lib.tc_engine_cancel.argtypes = [C.c_void_p]
        self.lib.tc_engine_free.argtypes = [C.c_void_p]
        self.lib.tc_string_free.argtypes = [C.c_void_p]
        self.engine, error = C.c_void_p(), C.c_void_p()
        code = self.lib.tc_engine_create_model(b"wan2.2-ti2v-5b-fastwan", str(MODEL).encode(),
                                               C.byref(self.engine), C.byref(error))
        message = self.consume(error)
        self.assertEqual(code, 0, message)
        self.addCleanup(self.lib.tc_engine_free, self.engine)
        self.temporary = tempfile.TemporaryDirectory(prefix="tc-wan22-safety-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def consume(self, pointer):
        if not pointer.value:
            return ""
        result = C.string_at(pointer).decode()
        self.lib.tc_string_free(pointer)
        return result

    def request(self, path):
        return dict(model="wan2.2-ti2v-5b-fastwan", operation="video.image", prompt="A fox walks slowly.",
                    output=str(path), width=128, height=128, frames=9, fps=24, steps=3, seed=42,
                    audio=False, execution="gpu", residency="resident",
                    inputs=[dict(kind="image", role="first_frame", path=str(IMAGE), strength=1)])

    def generate(self, request, callback):
        output, error = C.c_void_p(), C.c_void_p()
        code = self.lib.tc_engine_generate(self.engine, json.dumps(request).encode(), callback, None,
                                           C.byref(output), C.byref(error))
        return code, self.consume(output), self.consume(error)

    def test_existing_output_is_not_overwritten(self):
        destination = self.root / "existing.mp4"
        original = b"existing user data"
        destination.write_bytes(original)
        callback = CALLBACK(lambda _event, _context: None)
        code, _, error = self.generate(self.request(destination), callback)
        self.assertNotEqual(code, 0)
        self.assertIn("destination already exists", error)
        self.assertEqual(destination.read_bytes(), original)
        self.assertEqual(list(self.root.glob("existing.mp4.tmp.*")), [])

    def test_cancel_during_text_encoding_does_not_publish_video(self):
        destination = self.root / "cancelled.mp4"
        seen = []

        @CALLBACK
        def callback(raw, _context):
            event = json.loads(raw.decode())
            if event.get("phase") == "umt5_text_encode" and not seen:
                seen.append(event["phase"])
                self.lib.tc_engine_cancel(self.engine)

        code, _, error = self.generate(self.request(destination), callback)
        self.assertTrue(seen, "text-encode callback never fired")
        self.assertEqual(code, 2, error)
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob("cancelled.mp4.tmp.*")), [])


if __name__ == "__main__":
    unittest.main()
