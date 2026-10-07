"""Opt-in real local Qwen encoder/cache switches through the public C API.

Prepare-only: no DiT denoising, generated media or activation dumps. Model and
template paths are explicit fixtures; nothing is downloaded or changed.
"""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_QWEN_ENCODER_SESSION")=="1",
                     "explicit local model/session integration opt-in required")
class EncoderSessionTests(unittest.TestCase):
    def test_generation_edit_crossover_runtime_switch_and_cache_hit(self):
        library=(ROOT/os.environ["TURBOCIDER_NATIVE_LIBRARY_DIR"]/"libturbocider.dylib").resolve()
        model=ROOT/"models/Comfy-Org-Qwen-Image-2.1"
        manifests=[Path(os.environ[key]).resolve() for key in
            ("TURBOCIDER_TEST_ENCODER_MANIFEST_A","TURBOCIDER_TEST_ENCODER_MANIFEST_B")]
        self.assertNotEqual(hashlib.sha256(manifests[0].read_bytes()).hexdigest(),
                            hashlib.sha256(manifests[1].read_bytes()).hexdigest())
        lib=C.CDLL(str(library));engine=C.c_void_p();error=C.c_void_p()
        lib.tc_engine_create_model.argtypes=[C.c_char_p,C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        lib.tc_engine_prepare.argtypes=[C.c_void_p,C.c_char_p,C.c_int,C.c_void_p,C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        lib.tc_engine_free.argtypes=[C.c_void_p];lib.tc_string_free.argtypes=[C.c_void_p]
        def message(pointer):return C.string_at(pointer).decode() if pointer.value else ""
        status=lib.tc_engine_create_model(b"qwen-image-2.1",str(model).encode(),C.byref(engine),C.byref(error))
        text=message(error)
        if error.value:lib.tc_string_free(error)
        self.assertEqual(status,0,text)
        evidence=[]
        try:
            with tempfile.TemporaryDirectory(prefix="tc-qwen-encoder-session-") as directory:
                def run(name,manifest,edit,expected_hit):
                    request=dict(model="qwen-image-2.1",operation="image.edit" if edit else "image.generate",
                        prompt="Place the beige ceramic teapot from <image1> on a wooden table." if edit else "A ceramic teapot.",
                        width=512,height=512,steps=40,seed=29,audio=False,execution="gpu",residency="resident",
                        allow_approximation=True,qwen21_reference_size=512 if edit else 1024,
                        output=str(Path(directory)/(name+".png")),inputs=[dict(kind="image",role="reference",
                            path=str(ROOT/"results/qwen21/teapot-native-512.png"))] if edit else [])
                    if manifest:request["encoder_ane_manifest"]=str(manifest)
                    result=C.c_void_p();failure=C.c_void_p()
                    status=lib.tc_engine_prepare(engine,json.dumps(request).encode(),0,None,None,C.byref(result),C.byref(failure))
                    text=message(failure);payload=message(result)
                    if failure.value:lib.tc_string_free(failure)
                    if result.value:lib.tc_string_free(result)
                    self.assertEqual(status,0,text)
                    data=json.loads(payload)
                    self.assertTrue(data["prepared"])
                    self.assertEqual(data["prompt_cache_hit"],expected_hit)
                    self.assertFalse(Path(request["output"]).exists())
                    metrics=data.get("encoder_hybrid",{});reuse=data.get("encoder_runtime_reuse")
                    calls=reuse["actual_calls_this_request"] if reuse else metrics.get("runtime_calls_session_total",0)
                    retained=os.environ.get("TURBOCIDER_QWEN21_ENCODER_RETAIN_RUNTIME")=="1" and bool(manifest)
                    if expected_hit or not manifest:
                        self.assertEqual(calls,0);self.assertEqual(data["encoder_execution"],"gpu")
                    else:
                        self.assertGreater(calls,0);self.assertFalse(metrics["runtime_failed"])
                        self.assertEqual(metrics["session_released_after_encoding"],not retained)
                    if reuse:
                        self.assertEqual(reuse["enabled"],retained)
                        self.assertEqual(reuse["executor_retained"],retained)
                        if retained and not expected_hit:
                            self.assertEqual(reuse["executor_reused"],name=="ane-edit-after-generation")
                            self.assertGreater(reuse["retained_estimated_bytes"],0)
                            self.assertLessEqual(reuse["retained_estimated_bytes"],1<<30)
                    evidence.append(dict(name=name,cache_hit=data["prompt_cache_hit"],actual_encoder_calls=calls,
                        execution=data["encoder_execution"],backend=data["encoder_runtime_backend"],reuse=reuse,
                        cumulative_calls=metrics.get("runtime_calls_session_total",0)))
                    return request
                run("gpu-edit",None,True,False)
                run("ane-generation",manifests[0],False,False)
                run("ane-edit-after-generation",manifests[0],True,False)
                run("ane-edit-cache-hit",manifests[0],True,True)
                run("gpu-edit-after-ane",None,True,False)
                run("gpu-edit-cache-hit",None,True,True)
                run("changed-manifest",manifests[1],True,False)
                last=run("restored-manifest",manifests[0],True,False)
                for name,contents in (("malformed","{not-json"),
                    ("wrong-geometry",json.dumps({**json.loads(manifests[0].read_text()),"hidden":128}))):
                    invalid=Path(directory)/(name+".json");invalid.write_text(contents)
                    request={**last,"encoder_ane_manifest":str(invalid)}
                    result=C.c_void_p();failure=C.c_void_p()
                    status=lib.tc_engine_prepare(engine,json.dumps(request).encode(),0,None,None,C.byref(result),C.byref(failure))
                    text=message(failure)
                    if failure.value:lib.tc_string_free(failure)
                    if result.value:lib.tc_string_free(result)
                    self.assertNotEqual(status,0);self.assertTrue(text)
                    self.assertFalse(Path(request["output"]).exists())
                    evidence.append(dict(name=name,rejected=True,error=text))
                run("recover-after-invalid-template",manifests[0],True,False)
                if os.environ.get("TURBOCIDER_QWEN21_ENCODER_RETAIN_RUNTIME")=="1":
                    os.environ["TURBOCIDER_QWEN21_ENCODER_RETAIN_RUNTIME"]="0"
                    try:run("retention-off",manifests[0],True,False)
                    finally:os.environ["TURBOCIDER_QWEN21_ENCODER_RETAIN_RUNTIME"]="1"
            print(json.dumps(dict(schema="tc-qwen-encoder-session-switch-v1",
                library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),requests=evidence,
                scope="actual prepare-only session/cache contract; not denoise/performance/visual qualification")))
        finally:lib.tc_engine_free(engine)


if __name__=="__main__":unittest.main()
