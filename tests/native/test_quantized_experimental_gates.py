"""Actual initialization/generate gates; explicit ordinary binary/fixtures required."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PROBE = r'''
import ctypes as C, json, sys
lib=C.CDLL(sys.argv[1]);lib.tc_engine_create_model.argtypes=[C.c_char_p,C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
lib.tc_string_free.argtypes=[C.c_void_p];lib.tc_engine_free.argtypes=[C.c_void_p]
engine,error=C.c_void_p(),C.c_void_p()
status=lib.tc_engine_create_model(sys.argv[2].encode(),sys.argv[3].encode(),C.byref(engine),C.byref(error))
stage="create"
# The GGUF wrapper deliberately defers its inner Z session until generate.
if not status:
    stage="generate"
    lib.tc_engine_generate.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p,C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
    result=C.c_void_p()
    request=dict(schema_version=2,model=sys.argv[2],operation="image.generate",
        inputs=[dict(kind="text",role="prompt",text="experimental rejection fixture")],
        outputs=[dict(kind="image",path=sys.argv[4],width=512,height=512,audio=False)],
        sampling=dict(seed=42,steps=1),execution=dict(policy="gpu"),parameters=dict(dynamic_text=True))
    status=lib.tc_engine_generate(engine,json.dumps(request).encode(),None,None,C.byref(result),C.byref(error))
    if result.value:lib.tc_string_free(result)
message=C.string_at(error).decode() if error.value else None
if error.value:lib.tc_string_free(error)
if engine.value:lib.tc_engine_free(engine)
print(json.dumps(dict(status=status,error=message,stage=stage)))
'''


@unittest.skipUnless(os.environ.get("TC_ORDINARY_TEST_LIBRARY"), "explicit ordinary build and local fixture opt-in required")
class ExperimentalGateTests(unittest.TestCase):
    def test_ordinary_build_rejects_all_new_explicit_routes(self):
        library = str(Path(os.environ["TC_ORDINARY_TEST_LIBRARY"]).resolve())
        base_env = {k:v for k,v in os.environ.items() if not k.startswith("TURBOCIDER_")}
        bf16 = ROOT/"models/Comfy-Org-z_image_turbo"
        q8 = ROOT/"models/z-image-runtime-gguf-q8"
        if not (bf16.is_dir() and q8.is_dir()): self.fail("required local model fixtures missing")
        cases = [("z-image-turbo-gguf",q8,{"TURBOCIDER_Z_GGUF_IMPORT":"cpu_direct"}),
                 ("z-image-turbo-gguf",q8,{"TURBOCIDER_Z_GGUF_IMPORT":"cpu_direct","TURBOCIDER_Z_GGUF_COMPILE_PACKED":"1"}),
                 ("z-image-turbo-gguf",q8,{"TURBOCIDER_Z_GGUF_IMPORT":"cpu_direct","TURBOCIDER_Z_GGUF_RETAIN_PACKED":"1"}),
                 ("z-image-turbo",bf16,{"TURBOCIDER_Z_RUNTIME_CONVROT":"1"})]
        for model_id, root, controls in cases:
            with self.subTest(controls=controls), tempfile.TemporaryDirectory(prefix="tc-z-gate-") as raw:
                image=Path(raw)/"rejected.png"
                result=subprocess.run([sys.executable,"-c",PROBE,library,model_id,str(root),str(image)],
                                      env={**base_env,**controls},capture_output=True,text=True,timeout=20)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                row=json.loads(result.stdout)
                self.assertNotEqual(row["status"],0)
                self.assertIn("qe_capability_unqualified",row["error"])
                self.assertFalse(image.exists())


if __name__=="__main__": unittest.main()
