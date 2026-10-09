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
class JointAbTests(unittest.TestCase):
    def test_original_source_joint_math_and_alpha_oracle(self):
        mlx=Path(sysconfig.get_paths()["purelib"])/"mlx"
        with tempfile.TemporaryDirectory(prefix="tc-joint-ab-") as folder:
            root=Path(folder);probe=root/"probe"
            result=subprocess.run(["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror","-mmacosx-version-min=26.2",
                "-I",str(ROOT/"native/core"),"-isystem",str(mlx/"include"),str(ROOT/"tests/native/lora_joint_ab_test.cpp"),
                "-L",str(LIB),"-lturbocider","-L",str(mlx/"lib"),"-lmlx","-Wl,-rpath,"+str(LIB),"-Wl,-rpath,"+str(mlx/"lib"),"-o",str(probe)],
                cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([str(probe),str(root/"adapter.safetensors")],cwd=ROOT,capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr);self.assertIn("PASS joint BF16 A/B cases=24",result.stdout);print(result.stdout)

    def test_joint_policy_scope_and_legacy_flags_unchanged(self):
        env={k:v for k,v in os.environ.items() if not k.startswith("TURBOCIDER_")};env["TURBOCIDER_QWEN21_LORA_BF16_AB"]="1"
        with tempfile.TemporaryDirectory(prefix="tc-joint-ab-plan-") as folder:
            root=Path(folder);request=root/"request.json"
            base=dict(model="qwen-image-2.1",operation="image.generate",prompt="A teapot",output=str(root/"out.png"),width=512,height=512,
                steps=6,execution="gpu",residency="resident",audio=False,allow_approximation=True,lora_strategy="inference_time",
                loras=[dict(path="Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors",strength=1.,role="transformer")])
            def plan(change=None):
                request.write_text(json.dumps({**base,**(change or {})}))
                return subprocess.run([str(LIB/"turbocider"),"plan",str(request)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
            result=plan();self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("qwen21_joint_bf16_lora_ab_operands",json.loads(result.stdout)["algorithm_approximations"])
            for key in ("TURBOCIDER_QWEN21_LORA_B_FUSED_EPILOGUE","TURBOCIDER_QWEN21_LORA_BF16_OPERANDS_FP32_RANKS","TURBOCIDER_QWEN21_VIGGLE_LORA_FP16"):
                env[key]="1";self.assertNotEqual(plan().returncode,0);env.pop(key)
            for change in (dict(width=1024),dict(steps=5),dict(allow_approximation=False),dict(memory_budget_bytes=1<<30),dict(residency="component_staged")):
                self.assertNotEqual(plan(change).returncode,0)
            result=plan(dict(loras=[],lora_strategy="auto",steps=40,allow_approximation=False));self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertNotIn("qwen21_joint_bf16_lora_ab_operands",json.loads(result.stdout)["algorithm_approximations"])
            env["TURBOCIDER_QWEN21_LORA_BF16_AB"]="2";self.assertNotEqual(plan(dict(loras=[],lora_strategy="auto",steps=40)).returncode,0)
            env.pop("TURBOCIDER_QWEN21_LORA_BF16_AB");env.update(TURBOCIDER_QWEN21_LORA_B_FUSED_EPILOGUE="1",TURBOCIDER_QWEN21_LORA_BF16_OPERANDS_FP32_RANKS="1")
            self.assertNotEqual(plan().returncode,0)
            self.assertFalse((root/"out.png").exists())


if __name__=="__main__":unittest.main()
