#!/usr/bin/env python3
"""Opt-in real-model regression for streamed encoder release and cache provenance."""
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))
from benchmark_z_image_encoder_streaming import PROMPT, api, consume


class EncoderLifecycleTests(unittest.TestCase):
    def test_release_cache_hit_and_switch_back_to_gpu(self):
        paths = [os.environ.get(key) for key in ("TURBOCIDER_TEST_LIBRARY", "TURBOCIDER_Z_MODEL", "TURBOCIDER_ENCODER_MANIFEST")]
        if not all(paths):
            self.skipTest("real library, Z-Image model and encoder manifest must be explicitly provided")
        library, model, manifest = paths
        lib = api(library)
        engine, error = c.c_void_p(), c.c_void_p()
        status = lib.tc_engine_create_model_worker(b"z-image-turbo", model.encode(), c.byref(engine), c.byref(error))
        self.assertEqual(status, 0, consume(lib, error))
        callback_type = c.CFUNCTYPE(None, c.c_char_p, c.c_void_p)
        lib.tc_engine_generate.argtypes = [c.c_void_p, c.c_char_p, callback_type, c.c_void_p,
                                         c.POINTER(c.c_void_p), c.POINTER(c.c_void_p)]
        reports = []
        try:
            with tempfile.TemporaryDirectory() as raw:
                for index in range(3):
                    phases = []
                    @callback_type
                    def callback(value, _):
                        phases.append(json.loads(value)["phase"])
                    image = Path(raw) / f"image-{index}.png"
                    execution = {"policy": "gpu", "residency": "streamed", "memory_budget_bytes": 8 << 30}
                    if index < 2:
                        execution.update(encoder_ane_manifest=manifest, allow_approximation=True)
                    request = {"schema_version": 2, "model": "z-image-turbo", "operation": "image.generate",
                               "inputs": [{"kind": "text", "role": "prompt", "text": PROMPT}],
                               "outputs": [{"kind": "image", "path": str(image), "width": 64, "height": 64}],
                               "sampling": {"seed": 7, "steps": 1}, "execution": execution}
                    result, error = c.c_void_p(), c.c_void_p()
                    status = lib.tc_engine_generate(engine, json.dumps(request).encode(), callback, None,
                                                    c.byref(result), c.byref(error))
                    value, failure = consume(lib, result), consume(lib, error)
                    self.assertEqual(status, 0, failure)
                    value = json.loads(value)
                    reports.append({"result": value, "phases": phases,
                                    "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest()})
                    if index < 2:
                        self.assertEqual(value["encoder_execution"], "gpu_ane_experimental")
                        self.assertTrue(value["encoder_hybrid"]["session_released_after_encoding"])
                        self.assertEqual(value["encoder_hybrid"]["runtime_calls_session_total"], 35)
                        self.assertEqual(value["prompt_cache_hit"], index == 1)
                    else:
                        self.assertEqual(value["encoder_execution"], "gpu")
                        self.assertFalse(value.get("encoder_hybrid"))
                        self.assertFalse(value["prompt_cache_hit"])
                first = reports[0]["phases"]
                self.assertLess(first.index("qwen3_encoder_session_released"), first.index("load_z_image_stream"))
                self.assertNotIn("coreml_load", reports[1]["phases"])
                self.assertNotIn("z_image_text_encode", reports[1]["phases"])
                self.assertEqual(reports[0]["image_sha256"], reports[1]["image_sha256"])
        finally:
            lib.tc_engine_free(engine)
        output = os.environ.get("TURBOCIDER_ENCODER_LIFECYCLE_REPORT")
        if output:
            Path(output).write_text(json.dumps({"scope": "64-square, one-step functional lifecycle test; not performance evidence",
                "library_sha256": hashlib.sha256(Path(library).read_bytes()).hexdigest(), "cases": reports}, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
