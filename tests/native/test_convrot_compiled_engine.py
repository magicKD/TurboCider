"""Actual private ConvRot cache restoration and request-state tests."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU")=="1","explicit actual Metal/model opt-in required")
class ConvRotCompiledEngineTests(unittest.TestCase):
    def test_cache_restore_policy_cancel_and_prompt_invalidation(self):
        import mlx.core as mx
        library=ROOT/os.environ.get("TC_QUANTIZED_TEST_LIBRARY","build/quantized-execution/libturbocider.dylib")
        lib=C.CDLL(str(library));lib.tc_engine_create_model.argtypes=[C.c_char_p,C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        lib.tc_engine_generate.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p,C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        lib.tc_engine_free.argtypes=[C.c_void_p];lib.tc_string_free.argtypes=[C.c_void_p];lib.tc_engine_cancel.argtypes=[C.c_void_p]
        def take(pointer):
            if not pointer.value:return None
            text=C.string_at(pointer).decode();lib.tc_string_free(pointer);return text
        caller_hint=8<<30;previous=mx.set_cache_limit(caller_hint)
        env={"TURBOCIDER_Z_IMAGE_TRANSFORMER":str(ROOT/"models/Comfy-Org-z_image_turbo/split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors"),
            "TURBOCIDER_Z_CONVROT_GPU_RECIPE":"compiled_dense","TURBOCIDER_Z_CONVROT_CACHE_BYTES":str(3<<30),
            "TURBOCIDER_Z_CONVROT_CACHE_RETAIN":"1"}
        try:
            with tempfile.TemporaryDirectory(prefix="tc-convrot-engine-") as raw,patch.dict(os.environ,env):
                folder=Path(raw);engine,error=C.c_void_p(),C.c_void_p()
                self.assertEqual(lib.tc_engine_create_model(b"z-image-turbo",str(ROOT/"models/Comfy-Org-z_image_turbo").encode(),C.byref(engine),C.byref(error)),0,take(error))
                request={"schema_version":2,"model":"z-image-turbo","operation":"image.generate",
                    "inputs":[{"kind":"text","role":"prompt","text":"An adult ceramic artist holding a blue cup."}],
                    "sampling":{"seed":42,"steps":1},"execution":{"policy":"gpu","allow_approximation":True},"parameters":{"dynamic_text":True}}
                index=0
                def run(cancel=False):
                    nonlocal index
                    image=folder/f"image-{index}.png";index+=1
                    request["outputs"]=[{"kind":"image","path":str(image),"width":512,"height":512,"audio":False}]
                    callback=None
                    if cancel:
                        @C.CFUNCTYPE(None,C.c_char_p,C.c_void_p)
                        def callback(payload,_):
                            event=json.loads(payload)
                            if event["phase"]=="z_image_denoise_block" and event["completed"]==5:lib.tc_engine_cancel(engine)
                    value,error=C.c_void_p(),C.c_void_p()
                    status=lib.tc_engine_generate(engine,json.dumps(request).encode(),callback,None,C.byref(value),C.byref(error))
                    result,message=take(value),take(error)
                    self.assertEqual(mx.set_cache_limit(caller_hint),caller_hint,"request failed to restore caller cache hint")
                    self.assertLessEqual(mx.get_cache_memory(),3<<30,"retained allocator bins exceeded chosen hint")
                    return status,json.loads(result) if result else None,message,hashlib.sha256(image.read_bytes()).hexdigest() if image.is_file() else None
                try:
                    first=run();self.assertEqual(first[0],0,first[2]);golden=first[3]
                    warm=run();self.assertEqual(warm[0],0,warm[2]);self.assertEqual(warm[3],golden)
                    self.assertFalse(warm[1]["plan"]["executable"])
                    request["execution"]["allow_approximation"]=False
                    bad=run();self.assertNotEqual(bad[0],0);self.assertIsNone(bad[3]);self.assertIn("qe_config_conflict",bad[2])
                    request["execution"]["allow_approximation"]=True
                    cancelled=run(True);self.assertEqual(cancelled[0],2);self.assertIsNone(cancelled[3])
                    retried=run();self.assertEqual(retried[0],0,retried[2]);self.assertEqual(retried[3],golden)
                    request["inputs"][0]["text"]="A clear glass pitcher beside a red apple."
                    other=run();self.assertEqual(other[0],0,other[2]);self.assertFalse(other[1]["prompt_cache_hit"])
                    request["inputs"][0]["text"]="An adult ceramic artist holding a blue cup."
                    returned=run();self.assertEqual(returned[0],0,returned[2]);self.assertEqual(returned[3],golden)
                    print("PASS compiled ConvRot engine: caller hint restoration, bounded bins, authorization, cancel/retry and prompt A/B/A exact")
                finally:lib.tc_engine_free(engine)
        finally:mx.clear_cache();mx.set_cache_limit(previous)


if __name__=="__main__":unittest.main()
