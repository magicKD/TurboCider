import json
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
LIB=(ROOT/os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR","build/native")).resolve()


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU")=="1","explicit actual GPU opt-in required")
class FfnPhaseTests(unittest.TestCase):
    def test_compiled_base_lora_and_actual_prefix_routing(self):
        mlx=Path(sysconfig.get_paths()["purelib"])/"mlx"
        with tempfile.TemporaryDirectory(prefix="tc-ffn-phase-") as folder:
            root=Path(folder);probe=root/"probe"
            result=subprocess.run(["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror","-mmacosx-version-min=26.2",
                "-I",str(ROOT/"native/core"),"-isystem",str(mlx/"include"),str(ROOT/"tests/native/qwen_ffn_phase_test.cpp"),
                "-L",str(LIB),"-lturbocider","-L",str(mlx/"lib"),"-lmlx","-Wl,-rpath,"+str(LIB),"-Wl,-rpath,"+str(mlx/"lib"),"-o",str(probe)],
                cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([str(probe),str(root/"adapter.safetensors")],cwd=ROOT,capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertIn("PASS Qwen FFN phase cases=24",result.stdout);print(result.stdout)

    def test_explicit_phase_scope_and_plan_identity(self):
        env={k:v for k,v in os.environ.items() if not k.startswith("TURBOCIDER_")}
        env.update(TURBOCIDER_ANE_BACKEND="private",TURBOCIDER_ALLOW_PRIVATE_ANE="1",TURBOCIDER_PRIVATE_ANE_CHANNELS="5120")
        with tempfile.TemporaryDirectory(prefix="tc-ffn-phase-plan-") as folder:
            root=Path(folder);request=root/"request.json"
            base=dict(model="qwen-image-2.1",operation="image.edit",prompt="A teapot",output=str(root/"out.png"),width=512,height=512,
                steps=40,execution="gpu_ane",hybrid_mlp_mode="runtime",ane_manifest="unused.json",residency="resident",audio=False,
                allow_approximation=True,qwen21_reference_size=512,inputs=[dict(kind="image",role="reference",path="unused.png")])
            def plan(change=None):
                request.write_text(json.dumps({**base,**(change or {})}))
                return subprocess.run([str(LIB/"turbocider"),"plan",str(request)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
            for phase in ("all","prefill","decode"):
                env["TURBOCIDER_QWEN21_RUNTIME_FFN_PHASE"]=phase
                result=plan();self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                marker="qwen21_runtime_ffn_phase_"+phase
                self.assertEqual(marker in json.loads(result.stdout)["algorithm_approximations"],phase!="all")
            for change in (dict(width=1024),dict(allow_approximation=False),dict(residency="component_staged"),
                dict(memory_budget_bytes=1<<30),dict(encoder_ane_manifest="unused.json"),dict(inputs=base["inputs"]*3),
                dict(operation="image.generate",inputs=[])):
                self.assertNotEqual(plan(change).returncode,0)
            env["TURBOCIDER_ANE_BACKEND"]="public";self.assertNotEqual(plan().returncode,0);env["TURBOCIDER_ANE_BACKEND"]="private"
            env["TURBOCIDER_PRIVATE_ANE_CHANNELS"]="auto";self.assertNotEqual(plan().returncode,0);env["TURBOCIDER_PRIVATE_ANE_CHANNELS"]="5120"
            gpu=dict(execution="gpu",hybrid_mlp_mode="auto",ane_manifest="")
            result=plan(gpu);self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertNotIn("qwen21_runtime_ffn_phase_decode",json.loads(result.stdout)["algorithm_approximations"])
            for bad in ("","0","1","gpu","ALL"):
                env["TURBOCIDER_QWEN21_RUNTIME_FFN_PHASE"]=bad;self.assertNotEqual(plan(gpu).returncode,0)
            self.assertFalse((root/"out.png").exists())


if __name__=="__main__":unittest.main()
