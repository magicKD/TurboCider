"""Actual Z engine: metadata generations, cache invalidation and failed publish."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU")=="1","explicit Metal/local model opt-in required")
class MetadataEngineTests(unittest.TestCase):
    def test_warm_policy_rebind_cancel_and_inflight_source_change(self):
        library=ROOT/os.environ.get("TC_QUANTIZED_TEST_LIBRARY","build/quantized-execution/libturbocider.dylib")
        lib=C.CDLL(str(library));lib.tc_engine_create_model.argtypes=[C.c_char_p,C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        lib.tc_engine_generate.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p,C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        lib.tc_string_free.argtypes=[C.c_void_p];lib.tc_engine_free.argtypes=[C.c_void_p];lib.tc_engine_cancel.argtypes=[C.c_void_p]
        def take(pointer):
            if not pointer.value:return None
            value=C.string_at(pointer).decode();lib.tc_string_free(pointer);return value
        with tempfile.TemporaryDirectory(prefix="tc-qwen3-engine-") as raw:
            folder=Path(raw);config=folder/"config.json"
            original=(ROOT/"models/Tongyi-MAI-Z-Image-Turbo/text_encoder/config.json").read_bytes()
            config.write_bytes(original)
            env={"TURBOCIDER_Z_QWEN3_GGUF":str(ROOT/"models/Qwen3-4B-GGUF/Qwen3-4B-Q8_0.gguf"),
                "TURBOCIDER_Z_QWEN3_GGUF_CONFIG":str(config),"TURBOCIDER_QWEN3_GGUF_PREFETCH":"2",
                "TURBOCIDER_QWEN3_GGUF_WEIGHT_LIMIT_BYTES":str(1<<30),
                "TURBOCIDER_QWEN3_GGUF_SOURCE_RESIDENCY":"packed_streamed",
                "TURBOCIDER_QWEN3_GGUF_METADATA_CACHE":"1"}
            with patch.dict(os.environ,env):
                engine,error=C.c_void_p(),C.c_void_p()
                self.assertEqual(lib.tc_engine_create_model(b"z-image-turbo-gguf",str(ROOT/"models/z-image-runtime-gguf-q8").encode(),C.byref(engine),C.byref(error)),0,take(error))
                request={"schema_version":2,"model":"z-image-turbo-gguf","operation":"image.generate",
                    "inputs":[{"kind":"text","role":"prompt","text":"An adult ceramic artist holding a blue cup."}],
                    "sampling":{"seed":42,"steps":1},"parameters":{"dynamic_text":True,"compile_gpu":True},
                    "execution":{"policy":"gpu","allow_approximation":True,"quantized_execution":{"schema_version":1,"enabled":True,
                        "prefetch_layers":1,"source_residency":"packed_streamed","precision_profile":"z-raw-gpu-dependency-refresident-f16-v1"}}}
                index=0
                def run(action=None):
                    nonlocal index
                    image=folder/f"image-{index}.png";index+=1
                    request["outputs"]=[{"kind":"image","path":str(image),"width":512,"height":512,"audio":False}]
                    value,error=C.c_void_p(),C.c_void_p()
                    callback=None
                    if action:
                        @C.CFUNCTYPE(None,C.c_char_p,C.c_void_p)
                        def callback(payload,_):action(json.loads(payload))
                    status=lib.tc_engine_generate(engine,json.dumps(request).encode(),callback,None,C.byref(value),C.byref(error))
                    raw_result,message=take(value),take(error)
                    return status,json.loads(raw_result) if raw_result else None,message,hashlib.sha256(image.read_bytes()).hexdigest() if image.is_file() else None
                try:
                    first=run();self.assertEqual(first[0],0,first[2]);golden=first[3]
                    m=first[1]["encoder_quantized_execution"];generation=m["conditioning_producer_generation"]
                    warm=run();self.assertEqual(warm[0],0,warm[2]);self.assertEqual(warm[3],golden)
                    m=warm[1]["encoder_quantized_execution"];self.assertTrue(m["source_metadata_reused"])
                    self.assertEqual(m["source_metadata_preparations"],1);self.assertEqual(m["conditioning_producer_generation"],generation)
                    os.environ["TURBOCIDER_QWEN3_EVAL_INTERVAL"]="4"
                    rejected=run();self.assertNotEqual(rejected[0],0);self.assertIsNone(rejected[3]);self.assertIn("each-layer eval",rejected[2])
                    os.environ.pop("TURBOCIDER_QWEN3_EVAL_INTERVAL")
                    config.write_bytes(original+b" ")
                    changed=run();self.assertEqual(changed[0],0,changed[2]);self.assertEqual(changed[3],golden)
                    m=changed[1]["encoder_quantized_execution"];self.assertFalse(m["source_metadata_reused"])
                    self.assertEqual(m["source_metadata_preparations"],2);self.assertNotEqual(m["conditioning_producer_generation"],generation)
                    os.environ["TURBOCIDER_QWEN3_GGUF_PREFETCH"]="1"
                    rebound=run();self.assertEqual(rebound[0],0,rebound[2]);self.assertEqual(rebound[3],golden)
                    m=rebound[1]["encoder_quantized_execution"];self.assertTrue(m["source_metadata_reused"])
                    self.assertFalse(rebound[1]["prompt_cache_hit"]);self.assertEqual(m["slot_count"],2)
                    self.assertEqual(m["source_metadata_preparations"],2)
                    request["inputs"][0]["text"]="A clear glass pitcher beside a red apple."
                    def cancel(progress):
                        if progress["phase"]=="z_image_text_encode" and progress["completed"]==5:lib.tc_engine_cancel(engine)
                    cancelled=run(cancel);self.assertEqual(cancelled[0],2);self.assertIsNone(cancelled[3])
                    retried=run();self.assertEqual(retried[0],0,retried[2]);self.assertFalse(retried[1]["prompt_cache_hit"])
                    self.assertTrue(retried[1]["encoder_quantized_execution"]["source_metadata_reused"])
                    request["inputs"][0]["text"]="An adult ceramic artist holding a blue cup."
                    returned=run();self.assertEqual(returned[0],0,returned[2]);self.assertEqual(returned[3],golden)
                    mutated=[False]
                    def mutate(progress):
                        if not mutated[0] and progress["phase"]=="z_image_denoise_block" and progress["completed"]==2:
                            config.write_bytes(original+b"  ");mutated[0]=True
                    inflight=run(mutate);self.assertTrue(mutated[0]);self.assertNotEqual(inflight[0],0)
                    self.assertIsNone(inflight[3]);self.assertIn("source",inflight[2])
                    recovered=run();self.assertEqual(recovered[0],0,recovered[2]);self.assertEqual(recovered[3],golden)
                    self.assertFalse(recovered[1]["prompt_cache_hit"])
                    print("PASS metadata engine: warm policy rejection, config/prefetch rebind, cancel/retry, A/B/A and in-flight source mutation without PNG")
                finally:lib.tc_engine_free(engine)


if __name__=="__main__":unittest.main()
