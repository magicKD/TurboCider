"""Real BF16 prepare-only lifecycle; mutate ONLY an owned processor copy."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_QWEN_TOKENIZER_SESSION")=="1",
                     "explicit original BF16 processor/session opt-in required")
class TokenizerReuseSessionTests(unittest.TestCase):
    def test_fresh_prompts_switch_tamper_failure_recovery_and_unload(self):
        native=Path(os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR",ROOT/"build/native")).resolve()
        original=ROOT/"models/Comfy-Org-Qwen-Image-2.1"
        lib=C.CDLL(str(native/"libturbocider.dylib"))
        lib.tc_engine_create_model.argtypes=[C.c_char_p,C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        lib.tc_engine_prepare.argtypes=[C.c_void_p,C.c_char_p,C.c_int,C.c_void_p,C.c_void_p,
                                       C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        lib.tc_engine_unload.argtypes=[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        lib.tc_engine_cancel.argtypes=[C.c_void_p]
        lib.tc_engine_generate.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p,C.c_void_p,
                                        C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
        lib.tc_engine_free.argtypes=[C.c_void_p];lib.tc_string_free.argtypes=[C.c_void_p]
        def consume(pointer):
            value=C.string_at(pointer).decode() if pointer.value else ""
            if pointer.value:lib.tc_string_free(pointer)
            return value
        settings={key:value for key,value in os.environ.items() if key.startswith("TURBOCIDER_QWEN21_")}
        for key in settings:os.environ.pop(key)
        flag="TURBOCIDER_QWEN21_TOKENIZER_REUSE"
        engine=C.c_void_p()
        try:
            with tempfile.TemporaryDirectory(prefix="tc-qwen-tokenizer-session-") as raw:
                model=Path(raw)
                for name in ("diffusion_models","text_encoders","vae"):
                    (model/name).symlink_to(original/name,target_is_directory=True)
                (model/"processor").mkdir();processor=model/"processor/tokenizer.json"
                shutil.copyfile(original/"processor/tokenizer.json",processor)
                # Equal-size valid JSON tail mutations let the test preserve
                # mtime without ever changing a user-owned processor.
                processor.write_bytes(processor.read_bytes()+b" ")
                error=C.c_void_p()
                self.assertEqual(lib.tc_engine_create_model(b"qwen-image-2.1",str(model).encode(),C.byref(engine),C.byref(error)),0,
                                 consume(error))
                def call(name,prompt,reuse,hit,reused=False,constructed=False,expect_error=False,cancel=False):
                    os.environ[flag]=str(int(reuse))
                    request=dict(model="qwen-image-2.1",operation="image.generate",prompt=prompt,
                        width=512,height=512,steps=40,frames=1,audio=False,execution="gpu",residency="resident",
                        allow_approximation=False,output=str(model/(name+".png")),dump_tensors=str(model/name))
                    result,error=C.c_void_p(),C.c_void_p()
                    cancelled=[]
                    @C.CFUNCTYPE(None,C.c_char_p,C.c_void_p)
                    def callback(payload,_):
                        event=json.loads(payload)
                        if event["phase"]=="qwen21_text_encode" and event["completed"]==1:
                            cancelled.append(True);lib.tc_engine_cancel(engine)
                    status=lib.tc_engine_prepare(engine,json.dumps(request).encode(),0,
                        C.cast(callback,C.c_void_p) if cancel else None,None,C.byref(result),C.byref(error))
                    payload,message=consume(result),consume(error)
                    if expect_error:
                        self.assertNotEqual(status,0);self.assertFalse(payload);self.assertTrue(message)
                        if cancel:self.assertTrue(cancelled);self.assertEqual(status,2)
                        return None
                    self.assertEqual(status,0,message);row=json.loads(payload)
                    self.assertTrue(row["prepared"]);self.assertIs(row["prompt_cache_hit"],hit)
                    self.assertFalse(Path(request["output"]).exists())
                    self.assertFalse(row["encoder_weight_residency"]["enabled"])
                    self.assertFalse(row["encoder_weight_residency"]["weights_retained"])
                    self.assertNotIn("qwen_encoder_prefill",row)
                    self.assertEqual(row["encoder_gpu_graph"],"qwen3_vl_deepstack_gpu_only")
                    if reuse:
                        m=row["qwen_tokenizer_reuse"]
                        self.assertTrue(m["enabled"]);self.assertTrue(m["retained"])
                        self.assertIs(m["reused"],reused);self.assertIs(m["constructed_this_request"],constructed)
                        self.assertIs(m["tokenized_this_request"],not hit)
                        self.assertEqual(m["processor_sha256"],hashlib.sha256(processor.read_bytes()).hexdigest())
                        self.assertEqual(m["processor_source_bytes"],processor.stat().st_size)
                        self.assertGreater(m["validation_seconds_this_request"],0)
                        self.assertEqual(m["construction_seconds_this_request"]==0,not constructed)
                        self.assertEqual(m["tokenization_seconds_this_request"]==0,hit)
                    else:self.assertNotIn("qwen_tokenizer_reuse",row)
                    return row
                a,b="A blue ceramic teapot on a wooden table.","一只红狐狸站在雪地里。"
                call("ordinary-a",a,False,False)
                call("reuse-a",a,True,False,constructed=True)
                # Prepare-only deliberately suppresses tensor exports. Raw
                # conditioning parity belongs to separate generate captures.
                call("hit-a",a,True,True,reused=True)
                call("fresh-b",b,True,False,reused=True)
                call("ordinary-b",b,False,False)
                call("ordinary-b-hit",b,False,True)
                call("reuse-b",b,True,False,constructed=True)
                previous=processor.stat();data=processor.read_bytes()
                processor.write_bytes(data[:-1]+b"\n")
                os.utime(processor,ns=(previous.st_atime_ns,previous.st_mtime_ns))
                self.assertEqual(processor.stat().st_size,previous.st_size)
                call("changed-rejected",b,True,True,expect_error=True)
                call("recovered",b,True,False,constructed=True)
                result,error=C.c_void_p(),C.c_void_p()
                self.assertEqual(lib.tc_engine_unload(engine,C.byref(result),C.byref(error)),0,consume(error));consume(result)
                call("after-unload",b,True,False,constructed=True)
                call("cancelled-a",a,True,False,expect_error=True,cancel=True)
                call("after-cancel",a,True,False,constructed=True)
                # Late source changes must fail BEFORE PNG publication, not
                # merely report an error after a successful-looking output.
                output=model/"late-source-change.png";changed=[]
                @C.CFUNCTYPE(None,C.c_char_p,C.c_void_p)
                def change_after_step(payload,_):
                    event=json.loads(payload)
                    if event["phase"]=="denoise" and event["completed"]==1:
                        stamp=processor.stat()
                        with processor.open("r+b") as stream:stream.seek(-1,2);stream.write(b" ")
                        os.utime(processor,ns=(stamp.st_atime_ns,stamp.st_mtime_ns));changed.append(True)
                request=dict(model="qwen-image-2.1",operation="image.generate",prompt=a,width=512,height=512,
                    steps=2,frames=1,audio=False,execution="gpu",residency="resident",allow_approximation=False,output=str(output))
                result,error=C.c_void_p(),C.c_void_p()
                status=lib.tc_engine_generate(engine,json.dumps(request).encode(),C.cast(change_after_step,C.c_void_p),
                                              None,C.byref(result),C.byref(error))
                payload,message=consume(result),consume(error)
                self.assertTrue(changed);self.assertNotEqual(status,0);self.assertFalse(payload);self.assertTrue(message)
                self.assertFalse(output.exists())
                call("after-late-source-change",b,True,False,constructed=True)
                # An owner from fused prefill has a different source-size
                # policy. Adoption into the bounded standalone route must
                # reject it too, even though no new parse would be needed.
                result,error=C.c_void_p(),C.c_void_p()
                self.assertEqual(lib.tc_engine_unload(engine,C.byref(result),C.byref(error)),0,consume(error));consume(result)
                small=processor.read_bytes();processor.write_bytes(small+b" "*((16<<20)+1-len(small)))
                os.environ[flag]="0";os.environ["TURBOCIDER_QWEN21_ENCODER_PREFILL_GPU"]="1"
                oversized=dict(request,steps=40,allow_approximation=True,prompt=b)
                result,error=C.c_void_p(),C.c_void_p()
                status=lib.tc_engine_prepare(engine,json.dumps(oversized).encode(),0,None,None,C.byref(result),C.byref(error))
                payload,message=consume(result),consume(error)
                self.assertEqual(status,0,message);self.assertTrue(json.loads(payload)["qwen_encoder_prefill"]["processor_sha256"])
                os.environ["TURBOCIDER_QWEN21_ENCODER_PREFILL_GPU"]="0"
                call("oversized-owner-rejected",b,True,False,expect_error=True)
                processor.write_bytes(small)
                call("bounded-after-adoption-rejection",b,True,False,constructed=True)
        finally:
            if engine.value:lib.tc_engine_free(engine)
            for key in list(os.environ):
                if key.startswith("TURBOCIDER_QWEN21_"):os.environ.pop(key)
            os.environ.update(settings)


if __name__=="__main__":unittest.main()
