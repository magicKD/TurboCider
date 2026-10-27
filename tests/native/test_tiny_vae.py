"""CPU request contracts; real Metal parity/acceptance are explicit targets."""
import ctypes as C
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "build/native/libturbocider.dylib"


@unittest.skipUnless(LIBRARY.is_file(), "native build required")
class TinyVAEContracts(unittest.TestCase):
    def setUp(self):
        self.lib = C.CDLL(str(LIBRARY))
        self.lib.tc_plan_json.argtypes = [C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        self.lib.tc_string_free.argtypes = [C.c_void_p]
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "tiny.safetensors"
        self.path.write_bytes(b"fixture")  # plan must not load model/GPU weights

    def plan(self, **changes):
        request = dict(model="z-image-turbo", operation="image.generate", prompt="test", width=512,
                       height=512, steps=4, frames=1, output=str(self.path.parent / "out.png"),
                       tiny_vae=dict(path=str(self.path), preview=True, interval=2))
        request.update(changes)
        return self.call(request)

    def call(self, request):
        output, error = C.c_void_p(), C.c_void_p()
        status = self.lib.tc_plan_json(json.dumps(request).encode(), C.byref(output), C.byref(error))
        def consume(ptr):
            if not ptr.value:
                return None
            text = C.string_at(ptr).decode()
            self.lib.tc_string_free(ptr)
            return text
        return status, consume(output), consume(error)

    def test_preview_is_independent_of_approximation_permission(self):
        code, result, error = self.plan()
        self.assertEqual(code, 0, error)
        self.assertNotIn("tiny_vae_final_decoding", json.loads(result)["algorithm_approximations"])

    def test_tiny_final_requires_permission_and_is_reported(self):
        tiny = dict(path=str(self.path), decoder="tiny")
        self.assertNotEqual(self.plan(tiny_vae=tiny)[0], 0)
        code, result, error = self.plan(tiny_vae=tiny, allow_approximation=True)
        self.assertEqual(code, 0, error)
        self.assertIn("tiny_vae_final_decoding", json.loads(result)["algorithm_approximations"])

    def test_strict_types_ranges_fields_and_model(self):
        for config in (dict(interval=0), dict(interval=51), dict(interval=1.5), dict(interval=True),
                       dict(preview="true"), dict(encoder="auto"), dict(decoder="typo"),
                       dict(unrecognized=1), dict(path="relative.safetensors"), dict(path="/missing/tiny.safetensors")):
            with self.subTest(config=config):
                value = dict(path=str(self.path), preview=True)
                value.update(config)
                self.assertNotEqual(self.plan(tiny_vae=value)[0], 0)
        self.assertNotEqual(self.plan(model="flux2-klein-4b")[0], 0)

    def test_schema2_keeps_tiny_vae_in_parameters(self):
        request = dict(schema_version=2, model="z-image-turbo", operation="image.generate",
                       inputs=[dict(kind="text", role="prompt", text="test")],
                       outputs=[dict(kind="image", path=str(self.path.parent / "v2.png"), width=512, height=512,
                                     frames=1, fps=24, audio=False)], sampling=dict(steps=4, seed=42),
                       execution=dict(policy="gpu"),
                       parameters=dict(tiny_vae=dict(path=str(self.path), preview=True, interval=3)))
        code, result, error = self.call(request)
        self.assertEqual(code, 0, error)
        self.assertEqual(json.loads(result)["tiny_vae"]["interval"], 3)
        request["parameters"]["tiny_vae"]["decoder"] = "tiny"
        self.assertNotEqual(self.call(request)[0], 0)
        request["execution"]["allow_approximation"] = True
        self.assertEqual(self.call(request)[0], 0)


if __name__ == "__main__":
    unittest.main()
