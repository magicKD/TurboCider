"""Actual C API preflight rejection; tiny metadata fixture, no model inference."""
import ctypes as C
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


class Flux9HybridGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidate = os.environ.get("TURBOCIDER_FLUX9_GATE_LIBRARY")
        if not candidate:
            raise unittest.SkipTest("explicit experimental library required; no inference")
        cls.lib = C.CDLL(str(Path(candidate).resolve(strict=True)))
        pointer = C.c_void_p
        cls.lib.tc_engine_create_model.argtypes = [C.c_char_p,C.c_char_p,C.POINTER(pointer),C.POINTER(pointer)]
        cls.lib.tc_engine_generate.argtypes = [pointer,C.c_char_p,pointer,pointer,C.POINTER(pointer),C.POINTER(pointer)]
        cls.lib.tc_engine_free.argtypes = [pointer]
        cls.lib.tc_string_free.argtypes = [pointer]

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="tc-flux9-hybrid-gate-")
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        configs = {
            "transformer/config.json": dict(num_attention_heads=32, attention_head_dim=128,
                num_layers=8, num_single_layers=24, in_channels=128,
                guidance_embeds=False, joint_attention_dim=12288),
            "text_encoder/config.json": dict(hidden_size=4096, num_hidden_layers=36,
                num_attention_heads=32, num_key_value_heads=8),
            "vae/config.json": dict(latent_channels=32),
            "tokenizer/tokenizer.json": dict(model=dict(type="BPE",vocab={"g":0},merges=[]),
                pre_tokenizer=dict(pretokenizers=[dict(pattern=dict(Regex="."))]),added_tokens=[]),
        }
        for relative, value in configs.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        self.engine, error = C.c_void_p(), C.c_void_p()
        status = self.lib.tc_engine_create_model(b"flux2-klein-9b", str(root).encode(),
            C.byref(self.engine), C.byref(error))
        self.assertEqual(status,0,self.take(error))
        self.addCleanup(self.lib.tc_engine_free,self.engine)

    def take(self, pointer):
        result = C.string_at(pointer).decode() if pointer.value else ""
        if pointer.value:
            self.lib.tc_string_free(pointer)
        return result

    def rejected(self, changes, opt_in):
        task = dict(model="flux2-klein-9b",operation="image.generate",prompt="g",width=512,height=512,
            steps=4,seed=42,audio=False,execution="gpu_ane",residency="resident",dynamic_text=False,
            allow_approximation=True,ane_manifest="/tmp/not-loaded-flux9-manifest.json",
            output="/tmp/not-generated-flux9-hybrid.png")
        task.update(changes)
        result,error = C.c_void_p(),C.c_void_p()
        with patch.dict(os.environ, {"TURBOCIDER_FLUX9_EXPERIMENTAL_HYBRID":opt_in}):
            status = self.lib.tc_engine_generate(self.engine,json.dumps(task).encode(),None,None,
                C.byref(result),C.byref(error))
        self.take(result)
        message = self.take(error)
        self.assertNotEqual(status,0)
        self.assertIn("FLUX9B hybrid requires explicit source-bound512/4 base experimental opt-in",message)

    def test_exact_environment_opt_in_required_before_loading(self):
        for opt_in in ("", "0", "true", "01"):
            with self.subTest(opt_in=opt_in):
                self.rejected({},opt_in)

    def test_wrong_shape_steps_or_dynamic_text_rejected_with_opt_in(self):
        for changes in (dict(width=1024),dict(height=1024),dict(steps=8),dict(dynamic_text=True)):
            with self.subTest(changes=changes):
                self.rejected(changes,"1")


if __name__ == "__main__":
    unittest.main()
