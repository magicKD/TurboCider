import os
import sys
import json
import copy
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
LIB=(ROOT/os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR","build/native")).resolve()
sys.path.insert(0,str(ROOT/"tools/validation"))
from runtime_ane_common import validate_z_matched_rows
from z_runtime_bucket_switch import validate as validate_switch


class BucketPolicyTests(unittest.TestCase):
    def test_generic_gguf_matched_rows_need_actual_model_calls(self):
        def row(calls=248):return dict(model="z-image-turbo-gguf",width=512,height=512,text_tokens=37,
            acceleration_selection="experimental request-matched Private FFN rows=1152",hybrid=dict(bucket=1152,
                runtime_calls_session_total=calls,runtime_weight=dict(executor_backend="private_ane",partition_axis="intermediate_channels",
                    overflow_retries_session_total=0)))
        validate_z_matched_rows([row(),row(496),row(744)],8,(2,))
        for bad in ([row(480)],[row(),row()],[{}]):
            with self.assertRaises(ValueError):validate_z_matched_rows(bad,8,(2,))
        kept=row();kept["text_tokens"]=31;kept["hybrid"]["bucket"]=1056
        kept["acceleration_selection"]="experimental request-matched Private FFN rows=1056"
        validate_z_matched_rows([kept],8,(2,),1056)
    def test_actual_policy_and_caption_bounds(self):
        with tempfile.TemporaryDirectory(prefix="tc-bucket-policy-") as folder:
            probe=Path(folder)/"probe"
            subprocess.run(["clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror",str(ROOT/"tests/native/z_runtime_bucket_config_test.cpp"),
                str(ROOT/"native/core/common.cpp"),"-o",str(probe)],cwd=ROOT,check=True,capture_output=True,text=True,timeout=60)
            result=subprocess.run([str(probe)],capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertIn("PASS request-matched Z bucket",result.stdout)


class BucketSwitchTests(unittest.TestCase):
    @staticmethod
    def fixture():
        rows=[]
        for index in range(4):
            short=index in (0,3);bucket=1056 if short else 1152;calls=496 if index==2 else 248
            rows.append(dict(model="z-image-turbo",width=512,height=512,text_tokens=31 if short else 37,
                seed=42 if short else 17,actual_denoise_steps=8,lora_applied_projections=238,lora_strategy="inference_time",
                acceleration_selection=f"experimental request-matched Private FFN rows={bucket}",
                hybrid=dict(bucket=bucket,runtime_calls_session_total=calls,runtime_failed=False,
                    runtime_weight=dict(executor_backend="private_ane",partition_axis="intermediate_channels",
                        channel_blocks_session_total=calls,fallback_blocks_session_total=0,overflow_retries_session_total=0))))
        return rows,["a"*64,"b"*64,"b"*64,"a"*64]

    def test_original_source_switch_and_same_bucket_reuse(self):
        validate_switch(*self.fixture())

    def test_fake_bucket_reset_progress_and_failed_work_rejected(self):
        rows,hashes=self.fixture()
        for index,key,value in ((0,"bucket",1152),(1,"runtime_calls_session_total",480),
                                (2,"runtime_calls_session_total",248),(3,"runtime_calls_session_total",496),
                                (0,"runtime_failed",True)):
            bad=copy.deepcopy(rows);bad[index]["hybrid"][key]=value
            with self.assertRaises(ValueError):validate_switch(bad,hashes)
        for key,value in (("executor_backend","public"),("partition_axis","rows"),
                          ("channel_blocks_session_total",0),("overflow_retries_session_total",1),
                          ("fallback_blocks_session_total",1)):
            bad=copy.deepcopy(rows);bad[1]["hybrid"]["runtime_weight"][key]=value
            with self.assertRaises(ValueError):validate_switch(bad,hashes)

    def test_wrong_model_condition_geometry_and_adapter_rejected(self):
        rows,hashes=self.fixture()
        for key,value in (("model","z-image-turbo-gguf"),("width",1024),("text_tokens",45),("seed",42),
                          ("actual_denoise_steps",9),("lora_applied_projections",0),("lora_strategy","merged"),
                          ("acceleration_selection","")):
            bad=copy.deepcopy(rows);bad[1][key]=value
            with self.assertRaises(ValueError):validate_switch(bad,hashes)

    def test_unrestored_outputs_and_incomplete_evidence_rejected(self):
        rows,hashes=self.fixture()
        for bad_rows,bad_hashes in ((rows[:-1],hashes),(rows,hashes[:-1]),(rows,["a"*64]*3+["c"*64]),
                                   (rows,["a"*64,"b"*64,"c"*64,"a"*64]),(rows,[""]*4)):
            with self.assertRaises(ValueError):validate_switch(bad_rows,bad_hashes)


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU")=="1","actual built CLI plan opt-in required")
class BucketPlanTests(unittest.TestCase):
    def test_both_models_scope_and_gpu_control(self):
        env={k:v for k,v in os.environ.items() if not k.startswith("TURBOCIDER_")}
        env.update(TURBOCIDER_Z_RUNTIME_MATCH_ROWS="1",TURBOCIDER_ANE_BACKEND="private",TURBOCIDER_ALLOW_PRIVATE_ANE="1",
            TURBOCIDER_PRIVATE_ANE_CHANNELS="4096",TURBOCIDER_PRIVATE_ANE_DATA_PATH="w8a8",
            TURBOCIDER_RUNTIME_ANE_CHUNKS="1",TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC="1")
        with tempfile.TemporaryDirectory(prefix="tc-bucket-plan-") as folder:
            request=Path(folder)/"request.json"
            base=dict(model="z-image-turbo",operation="image.generate",prompt="A fox",width=512,height=512,steps=8,audio=False,
                execution="gpu_ane",hybrid_mlp_mode="runtime",ane_manifest="checked-at-load.json",residency="resident",
                allow_approximation=True,output=str(Path(folder)/"out.png"))
            def plan(change=None):
                request.write_text(json.dumps({**base,**(change or {})}))
                return subprocess.run([str(LIB/"turbocider"),"plan",str(request)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
            for model in ("z-image-turbo","z-image-turbo-gguf"):
                result=plan(dict(model=model));self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            for change in (dict(width=1024),dict(allow_approximation=False),dict(memory_budget_bytes=1<<30),
                dict(residency="component_staged"),dict(encoder_ane_manifest="unused.json")):
                self.assertNotEqual(plan(change).returncode,0)
            env["TURBOCIDER_ANE_BACKEND"]="public";self.assertNotEqual(plan().returncode,0)
            env["TURBOCIDER_ANE_BACKEND"]="private"
            self.assertEqual(plan(dict(execution="gpu",hybrid_mlp_mode="auto",ane_manifest="")).returncode,0)
            env["TURBOCIDER_Z_RUNTIME_MATCH_ROWS"]="2";self.assertNotEqual(plan().returncode,0)
            self.assertFalse((Path(folder)/"out.png").exists())


if __name__=="__main__":unittest.main()
