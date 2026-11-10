"""CPU request contracts; real Metal parity/acceptance are explicit targets."""
import ctypes as C
import json
import os
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = Path(os.environ.get("TURBOCIDER_NATIVE_LIBRARY", ROOT / "build/native/libturbocider.dylib"))


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

    def test_wan22_super_is_final_decode_only_and_requires_permission(self):
        request = dict(model="wan2.2-ti2v-5b-fastwan", operation="video.image", prompt="test",
                       width=512, height=512, steps=3, frames=97, fps=24, audio=False,
                       output=str(self.path.parent / "out.mp4"), execution="gpu", residency="resident",
                       inputs=[dict(kind="image", role="first_frame", path=str(self.path), strength=1)],
                       tiny_vae=dict(path=str(self.path), encoder="original", decoder="tiny", preview=False))
        self.assertNotEqual(self.call(request)[0], 0)
        request["allow_approximation"] = True
        code, result, error = self.call(request)
        self.assertEqual(code, 0, error)
        self.assertIn("tiny_vae_final_decoding", json.loads(result)["algorithm_approximations"])
        request["tiny_vae"]["preview"] = True
        self.assertNotEqual(self.call(request)[0], 0)
        request["tiny_vae"]["preview"] = False
        request["tiny_vae"]["encoder"] = "tiny"
        self.assertNotEqual(self.call(request)[0], 0)

    def test_wan22_even_resolution_is_padded_and_has_integer_guards(self):
        request = dict(model="wan2.2-ti2v-5b-fastwan", operation="video.image", prompt="test",
                       width=768, height=768, steps=3, frames=97, fps=24, audio=False,
                       output=str(self.path.parent / "out.mp4"), execution="gpu", residency="resident",
                       inputs=[dict(kind="image", role="first_frame", path=str(self.path), strength=1)])
        for model, steps in (("wan2.2-ti2v-5b-fastwan", 3), ("wan2.2-ti2v-5b-quanhaol", 4)):
            request.update(model=model, steps=steps)
            for width, height in ((800, 768), (896, 896), (1024, 1024), (1280, 720),
                                  (720, 1280), (1536, 1024), (2560, 1440), (68, 66)):
                with self.subTest(model=model, width=width, height=height):
                    request["width"], request["height"] = width, height
                    code, result, error = self.call(request)
                    self.assertEqual(code, 0, error)  # Includes full VAE, not a runtime guarantee.
                    plan = json.loads(result)
                    self.assertEqual(plan["requested_shape"], [width, height, 97])
                    self.assertEqual(plan["decoded_shape"], [(width + 31) // 32 * 32, (height + 31) // 32 * 32, 97])
                    self.assertEqual(plan["output_shape"], [width, height, 97])
        request["tiny_vae"] = dict(path=str(self.path), encoder="original", decoder="tiny", preview=False)
        request["allow_approximation"] = True
        request.update(width=2560, height=1440)
        self.assertEqual(self.call(request)[0], 0)
        request.update(width=1280, height=720)
        _, padded_result, _ = self.call(request)
        request["height"] = 736
        _, aligned_result, _ = self.call(request)
        self.assertEqual(json.loads(padded_result)["memory_estimate_bytes"],
                         json.loads(aligned_result)["memory_estimate_bytes"])
        for width, height in ((0, 512), (511, 512), (512, 513), (2_147_483_618, 512),
                              (1_000_000, 1_000_000), (2_147_483_648, 512)):
            with self.subTest(width=width, height=height):
                request["width"], request["height"] = width, height
                self.assertNotEqual(self.call(request)[0], 0)
        request.update(width=512, height=512, frames=96)
        self.assertNotEqual(self.call(request)[0], 0)

    def test_quanhaol_has_a_separate_four_step_contract(self):
        request = dict(model="wan2.2-ti2v-5b-quanhaol", operation="video.image", prompt="test",
                       width=512, height=512, steps=4, frames=97, fps=24, audio=False,
                       output=str(self.path.parent / "out.mp4"), execution="gpu", residency="resident",
                       inputs=[dict(kind="image", role="first_frame", path=str(self.path), strength=1)],
                       tiny_vae=dict(path=str(self.path), encoder="original", decoder="tiny", preview=False),
                       allow_approximation=True)
        code, result, error = self.call(request)
        self.assertEqual(code, 0, error)
        plan = json.loads(result)
        stages = plan["stages"]
        self.assertEqual(next(stage["iterations"] for stage in stages if stage["id"] == "denoise"), 4)
        self.lib.tc_system_json.restype = C.c_void_p
        pointer = C.c_void_p(self.lib.tc_system_json())
        try:
            memory = json.loads(C.string_at(pointer).decode())["physical_memory_bytes"]
        finally:
            self.lib.tc_string_free(pointer)
        estimate = (38 << 30) + 512 * 512 * 97 * 384
        if memory >= (64 << 30):
            estimate += 384 << 20
        self.assertEqual(plan["memory_estimate_bytes"], estimate)
        request["steps"] = 3
        self.assertNotEqual(self.call(request)[0], 0)


if __name__ == "__main__":
    unittest.main()
