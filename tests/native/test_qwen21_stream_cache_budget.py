"""Strict native JSON budget contracts; no checkpoint or GPU allocation required."""
import copy
import ctypes as C
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(sys.platform == "darwin", "requires the native macOS JSON bridge")
class StreamCacheBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", "build/native") / "libturbocider.dylib"
        if not path.is_file():
            raise unittest.SkipTest("build the current native library first")
        cls.lib = C.CDLL(str(path))
        cls.lib.tc_plan_json.argtypes = [C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        cls.lib.tc_string_free.argtypes = [C.c_void_p]

    def request(self, version=1):
        if version == 1:
            return dict(model="qwen-image-2.1", operation="image.generate", prompt="test", width=512,
                        height=512, steps=8, seed=42, output="test.png", execution="gpu", residency="streamed",
                        allow_approximation=True)
        return dict(schema_version=2, model="qwen-image-2.1", operation="image.generate",
                    inputs=[dict(kind="text", role="prompt", text="test")],
                    outputs=[dict(kind="image", path="test.png", width=512, height=512)],
                    sampling=dict(steps=8, seed=42),
                    execution=dict(policy="gpu", residency="streamed", allow_approximation=True))

    def plan(self, request):
        value, error = C.c_void_p(), C.c_void_p()
        status = self.lib.tc_plan_json(json.dumps(request).encode(), C.byref(value), C.byref(error))
        def take(pointer):
            if not pointer.value:
                return None
            text = C.string_at(pointer).decode()
            self.lib.tc_string_free(pointer)
            return text
        result, message = take(value), take(error)
        return status, json.loads(result) if result else None, message

    def test_zero_legacy_default_and_all_positive_sizes_in_both_schemas(self):
        for version in (1, 2):
            for budget in (None, 0, 1 << 30, 3584 << 20, 16 << 30):
                with self.subTest(version=version, budget=budget):
                    request = self.request(version)
                    options = request if version == 1 else request["execution"]
                    if budget is not None:
                        options["qwen21_stream_weight_budget_bytes"] = budget
                    status, result, error = self.plan(request)
                    self.assertEqual(status, 0, error)
                    self.assertEqual(result.get("qwen21_stream_weight_budget_bytes", 0), budget or 0)
                    self.assertEqual(result["residency"], "streamed")

    def test_invalid_types_bounds_and_inactive_routes_are_rejected(self):
        for version in (1, 2):
            for budget in (True, "2147483648", -1, 1.5, (1 << 30) - 1, (16 << 30) + 1, 1e100):
                with self.subTest(version=version, budget=budget):
                    request = self.request(version)
                    options = request if version == 1 else request["execution"]
                    options["qwen21_stream_weight_budget_bytes"] = budget
                    self.assertNotEqual(self.plan(request)[0], 0)
            request = self.request(version)
            options = request if version == 1 else request["execution"]
            options.update(qwen21_stream_weight_budget_bytes=2 << 30, residency="resident")
            status, _, error = self.plan(request)
            self.assertNotEqual(status, 0)
            self.assertIn("Qwen21 streamed residency", error)
            other = copy.deepcopy(request)
            other["model"] = "z-image-turbo"
            self.assertNotEqual(self.plan(other)[0], 0)


if __name__ == "__main__":
    unittest.main()
