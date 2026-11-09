"""CPU-visible native catalog and request contract for LTX-Video 0.9.8."""
import ctypes as C
import json
import os
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
LIBRARY = Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native")).resolve() / "libturbocider.dylib"


@unittest.skipUnless(LIBRARY.is_file(), "native build required")
class LTX098ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = C.CDLL(str(LIBRARY))
        cls.lib.tc_models_json.restype = C.c_void_p
        cls.lib.tc_string_free.argtypes = [C.c_void_p]
        cls.lib.tc_plan_json.argtypes = [C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]

    def consume(self, pointer):
        if not pointer.value:
            return None
        result = C.string_at(pointer).decode()
        self.lib.tc_string_free(pointer)
        return result

    def plan(self, request):
        output, error = C.c_void_p(), C.c_void_p()
        status = self.lib.tc_plan_json(json.dumps(request).encode(), C.byref(output), C.byref(error))
        result, reason = self.consume(output), self.consume(error)
        return status, json.loads(result) if result else None, reason

    def test_catalog_and_exact_gpu_scope(self):
        catalog = json.loads(self.consume(C.c_void_p(self.lib.tc_models_json())))
        model = next(value for value in catalog["models"] if value["id"] == "ltx-video-0.9.8-distilled")
        self.assertTrue(model["executor"])
        self.assertEqual(model["executor_operations"], ["video.image"])
        self.assertEqual((model["default_width"], model["default_height"], model["default_frames"],
                          model["default_fps"], model["default_steps"]), (512, 512, 121, 24, 10))
        self.assertFalse(model["default_audio"])
        base = dict(model=model["id"], operation="video.image", prompt="A lighthouse by the sea",
                    width=512, height=512, frames=121, fps=24, steps=10, audio=False,
                    execution="gpu", residency="resident",
                    inputs=[dict(kind="image", role="first_frame", path="/tmp/ltx098-source.png", strength=.75)])
        code, result, error = self.plan(base)
        self.assertEqual(code, 0, error)
        self.assertEqual(result["execution"], "gpu")
        self.assertEqual(result["encoder_gpu_graph"], "t5xxl_gpu_only")
        self.assertEqual(result["encoder_precision"], "bf16_weights_fp32_activations")
        self.assertEqual(result["encoder_weight_validation"], "native T5-XXL checkpoint required at execution")
        self.assertEqual(result["decoded_shape"], [640, 640, 121])
        self.assertEqual(result["output_shape"], [640, 640, 121])
        self.assertEqual(self.plan({**base, "steps": 8})[0], 0)
        resized_code, resized, resized_error = self.plan({**base, "ltx098_resize_to_base": True})
        self.assertEqual(resized_code, 0, resized_error)
        self.assertEqual(resized["decoded_shape"], [640, 640, 121])
        self.assertEqual(resized["output_shape"], [512, 512, 121])
        self.assertNotEqual(self.plan({**base, "steps": 8, "ltx098_resize_to_base": True})[0], 0)
        for change in (dict(frames=120), dict(frames=129), dict(width=513), dict(steps=7), dict(fps=25),
                       dict(steps=9),
                       dict(audio=True), dict(residency="streamed"), dict(execution="auto"),
                       dict(inputs=[]), dict(inputs=[dict(kind="image", role="first_frame", path="/tmp/x", strength=1.1)]),
                       dict(ltx_sol_stage2=True, allow_approximation=True)):
            with self.subTest(change=change):
                self.assertNotEqual(self.plan({**base, **change})[0], 0)


if __name__ == "__main__":
    unittest.main()
