"""Opt-in original mixed-GGUF prefill lifecycle through the public C API.

Prepare-only, no denoising or export. Only an owned processor copy is changed;
the original model components are read-only symlinks.
"""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_QWEN_PREFILL_SESSION") == "1",
                     "explicit original local mixed-GGUF session opt-in required")
class EncoderPrefillSessionTests(unittest.TestCase):
    def test_cache_profile_switch_processor_change_recovery_and_unload(self):
        library = ROOT / os.environ["TURBOCIDER_NATIVE_LIBRARY_DIR"] / "libturbocider.dylib"
        original = ROOT / "models/unsloth-qwen21-q4-k-m"
        lib = C.CDLL(str(library.resolve()))
        lib.tc_engine_create_model.argtypes = [C.c_char_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        lib.tc_engine_prepare.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_void_p, C.c_void_p,
                                         C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        lib.tc_engine_unload.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
        lib.tc_engine_free.argtypes = [C.c_void_p]
        lib.tc_string_free.argtypes = [C.c_void_p]

        def take(pointer):
            value = C.string_at(pointer).decode() if pointer.value else ""
            if pointer.value:
                lib.tc_string_free(pointer)
            return value

        settings = dict(TURBOCIDER_QWEN21_ENCODER_RETAIN_WEIGHTS="1",
                        TURBOCIDER_QWEN21_ENCODER_COMPILED_GPU="0",
                        TURBOCIDER_QWEN21_ENCODER_PREFILL_GPU="1",
                        TURBOCIDER_QWEN21_GGUF_DECODE_WORKERS="8")
        previous = {key: os.environ.get(key) for key in settings}
        os.environ.update(settings)
        evidence = []
        engine = C.c_void_p()
        try:
            with tempfile.TemporaryDirectory(prefix="tc-qwen-prefill-session-") as raw:
                model = Path(raw)
                for name in ("diffusion_models", "text_encoders", "vae"):
                    (model / name).symlink_to(original / name, target_is_directory=True)
                (model / "processor").mkdir()
                processor = model / "processor/tokenizer.json"
                shutil.copyfile(original / "processor/tokenizer.json", processor)
                source_hash = hashlib.sha256(processor.read_bytes()).hexdigest()
                failure = C.c_void_p()
                status = lib.tc_engine_create_model(b"qwen-image-2.1", str(model).encode(),
                                                    C.byref(engine), C.byref(failure))
                self.assertEqual(status, 0, take(failure))

                def run(name, prompt, enabled, hit, loads, reused=False, processor_reused=True):
                    os.environ["TURBOCIDER_QWEN21_ENCODER_PREFILL_GPU"] = str(int(enabled))
                    request = dict(model="qwen-image-2.1", operation="image.generate", prompt=prompt,
                                   width=512, height=512, steps=4, seed=29, audio=False,
                                   execution="gpu", residency="resident", allow_approximation=True,
                                   output=str(model / (name + ".png")))
                    result, failure = C.c_void_p(), C.c_void_p()
                    status = lib.tc_engine_prepare(engine, json.dumps(request).encode(), 0,
                                                   None, None, C.byref(result), C.byref(failure))
                    message, payload = take(failure), take(result)
                    self.assertEqual(status, 0, message)
                    row = json.loads(payload)
                    self.assertTrue(row["prepared"])
                    self.assertIs(row["prompt_cache_hit"], hit)
                    self.assertFalse(Path(request["output"]).exists())
                    weights = row["encoder_weight_residency"]
                    self.assertTrue(weights["weights_retained"])
                    self.assertIs(weights["weights_reused"], reused)
                    self.assertEqual(weights["loads_session_total"], loads)
                    if enabled:
                        p = row["qwen_encoder_prefill"]
                        self.assertEqual(p["processor_sha256"], source_hash)
                        self.assertIs(p["tokenizer_reused"], processor_reused)
                        self.assertIs(p["encoder_evaluated_this_request"], not hit)
                        self.assertIs(p["native_gqa_attention"], not hit)
                        self.assertIs(p["fused_rms_qk_neox_rope"], not hit)
                        self.assertEqual([p[k] for k in ("qkv_fused_layers", "qk_fused_layers", "gate_up_fused_layers")],
                                         [0, 0, 0] if hit else [18, 18, 36])
                        if hit:
                            self.assertEqual((p["input_rows"], p["retained_rows"], p["pack_seconds_this_request"]), (0, 0, 0))
                        else:
                            self.assertGreater(p["input_rows"], p["retained_rows"])
                            self.assertGreater(p["retained_rows"], 0)
                            if reused:
                                self.assertEqual(p["pack_seconds_this_request"], 0)
                            else:
                                self.assertGreater(p["pack_seconds_this_request"], 0)
                    else:
                        self.assertNotIn("qwen_encoder_prefill", row)
                    evidence.append(dict(name=name, prompt_cache_hit=hit, weights=weights,
                                         prefill=row.get("qwen_encoder_prefill"), prepare_wall_seconds=row["seconds"]))
                    return request

                prompt_a, prompt_b = "A red fox in a snowy forest.", "A ceramic teapot on a wooden table."
                run("cold-prefill", prompt_a, True, False, 1, processor_reused=False)
                run("cached-prefill", prompt_a, True, True, 1)
                run("fresh-prefill", prompt_b, True, False, 1, reused=True)
                run("switch-to-legacy", prompt_b, False, False, 2)
                run("cached-legacy", prompt_b, False, True, 2)
                run("switch-to-prefill", prompt_b, True, False, 3)
                request = run("cached-after-switch", prompt_b, True, True, 3)
                # JSON remains valid, but its verified generation must not be
                # reused even when the same prompt would hit conditioning.
                processor.write_bytes(processor.read_bytes() + b"\n")
                source_hash = hashlib.sha256(processor.read_bytes()).hexdigest()
                result, failure = C.c_void_p(), C.c_void_p()
                status = lib.tc_engine_prepare(engine, json.dumps(request).encode(), 0,
                                               None, None, C.byref(result), C.byref(failure))
                message, payload = take(failure), take(result)
                self.assertNotEqual(status, 0)
                self.assertTrue(message)
                self.assertFalse(payload)
                evidence.append(dict(name="changed-processor-rejected", rejected=True, error=message))
                run("recover-processor", prompt_b, True, False, 4, processor_reused=False)
                result, failure = C.c_void_p(), C.c_void_p()
                status = lib.tc_engine_unload(engine, C.byref(result), C.byref(failure))
                take(result)
                self.assertEqual(status, 0, take(failure))
                run("after-unload", prompt_b, True, False, 1, processor_reused=False)
            print(json.dumps(dict(schema="tc-qwen21-encoder-prefill-session-v1",
                                  library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
                                  requests=evidence, qualification_passed=False,
                                  scope="real prepare/cache/source lifecycle; no denoise/performance qualification")))
        finally:
            if engine.value:
                lib.tc_engine_free(engine)
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
