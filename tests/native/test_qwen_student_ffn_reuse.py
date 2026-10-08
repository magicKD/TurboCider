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
class StudentReuseTests(unittest.TestCase):
    def test_actual_compiled_full_lora_capture_reuse_and_cleanup(self):
        mlx=Path(sysconfig.get_paths()["purelib"])/"mlx"
        with tempfile.TemporaryDirectory(prefix="tc-student-cache-") as folder:
            root=Path(folder);probe=root/"probe"
            result=subprocess.run(["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror","-mmacosx-version-min=26.2",
                "-I",str(ROOT/"native/core"),"-isystem",str(mlx/"include"),str(ROOT/"tests/native/qwen_student_ffn_reuse_test.cpp"),
                "-L",str(LIB),"-lturbocider","-L",str(mlx/"lib"),"-lmlx","-Wl,-rpath,"+str(LIB),"-Wl,-rpath,"+str(mlx/"lib"),"-o",str(probe)],
                cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([str(probe),str(root/"adapter.safetensors")],cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertIn("PASS student full-LoRA FFN cache",result.stdout);print(result.stdout)

    def test_explicit_request_gates_and_plan_label(self):
        env={key:value for key,value in os.environ.items() if not key.startswith("TURBOCIDER_")}
        env["TURBOCIDER_QWEN21_STUDENT_FINAL_FFN_REUSE"]="1"
        with tempfile.TemporaryDirectory(prefix="tc-student-cache-gates-") as folder:
            root=Path(folder);request=root/"request.json"
            base=dict(model="qwen-image-2.1",operation="image.generate",prompt="A teapot",output=str(root/"output.png"),
                width=512,height=512,steps=6,execution="gpu",residency="resident",audio=False,allow_approximation=True,
                lora_strategy="inference_time",loras=[dict(path="Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors",strength=1.0,role="transformer")])
            def plan(change=None):
                request.write_text(json.dumps({**base,**(change or {})}))
                return subprocess.run([str(LIB/"turbocider"),"plan",str(request)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
            result=plan();self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("qwen21_student_final_ffn_reuse",json.loads(result.stdout)["algorithm_approximations"])
            for value in ("16","32"):
                env["TURBOCIDER_QWEN21_STUDENT_FINAL_FFN_REUSE"]=value;self.assertEqual(plan().returncode,0)
            env["TURBOCIDER_QWEN21_STUDENT_FINAL_FFN_REUSE"]="1"
            for change in (dict(width=1024),dict(steps=5),dict(allow_approximation=False),dict(residency="component_staged"),
                dict(memory_budget_bytes=1<<30),dict(lora_strategy="in_memory_merge")):
                with self.subTest(change=change):self.assertNotEqual(plan(change).returncode,0)
            for key in ("TURBOCIDER_QWEN21_GPU_REUSE_FINAL_FFN","TURBOCIDER_QWEN21_VIGGLE_LORA_FP16","TURBOCIDER_QWEN21_LORA_BF16_OPERANDS_FP32_RANKS"):
                env[key]="1";self.assertNotEqual(plan().returncode,0);env.pop(key)
            env["TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC"]="1"
            reference=dict(kind="image",role="reference",path="unused.png")
            for count in (1,2):self.assertEqual(plan(dict(operation="image.edit",inputs=[reference]*count,qwen21_reference_size=512)).returncode,0)
            self.assertNotEqual(plan(dict(operation="image.edit",inputs=[reference]*3,qwen21_reference_size=512)).returncode,0)
            env.pop("TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC")
            result=plan(dict(loras=[],lora_strategy="auto",steps=40,allow_approximation=False));self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertNotIn("qwen21_student_final_ffn_reuse",json.loads(result.stdout)["algorithm_approximations"])
            env["TURBOCIDER_QWEN21_STUDENT_FINAL_FFN_REUSE"]="2";self.assertNotEqual(plan(dict(loras=[],lora_strategy="auto",steps=40)).returncode,0)
            self.assertFalse((root/"output.png").exists())


if __name__=="__main__":unittest.main()
