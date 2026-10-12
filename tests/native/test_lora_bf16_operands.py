import json
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
LIB=(ROOT/os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR","build/native")).resolve()


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU")=="1","explicit actual MLX/Metal opt-in required")
class LoraBf16OperandTests(unittest.TestCase):
    def test_source_operands_retaining_fp32_ranks(self):
        mlx=Path(sysconfig.get_paths()["purelib"])/"mlx"
        with tempfile.TemporaryDirectory(prefix="tc-bf16-source-ranks-") as folder:
            root=Path(folder);probe=root/"probe"
            result=subprocess.run(["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror",
                "-mmacosx-version-min=26.2","-I",str(ROOT/"native/core"),"-isystem",str(mlx/"include"),
                str(ROOT/"tests/native/lora_bf16_operand_test.cpp"),"-L",str(LIB),"-lturbocider","-L",str(mlx/"lib"),"-lmlx",
                "-Wl,-rpath,"+str(LIB),"-Wl,-rpath,"+str(mlx/"lib"),"-o",str(probe)],cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([str(probe),str(root/"adapter.safetensors")],cwd=ROOT,capture_output=True,text=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("PASS BF16 operand/F32 rank cases=108",result.stdout);print(result.stdout)

    def test_request_snapshot_and_precision_gates(self):
        env={key:value for key,value in os.environ.items() if not key.startswith("TURBOCIDER_")}
        env["TURBOCIDER_QWEN21_LORA_BF16_OPERANDS_FP32_RANKS"]="1"
        with tempfile.TemporaryDirectory(prefix="tc-rank-operand-gates-") as folder:
            root=Path(folder);request=root/"request.json"
            base=dict(model="qwen-image-2.1",operation="image.generate",prompt="A teapot",output=str(root/"output.png"),
                width=512,height=512,steps=6,execution="gpu",residency="resident",audio=False,allow_approximation=True,
                lora_strategy="inference_time",loras=[dict(path="Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors",strength=1.0,role="transformer")])
            def plan(change=None):
                request.write_text(json.dumps({**base,**(change or {})}))
                return subprocess.run([str(LIB/"turbocider"),"plan",str(request)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
            result=plan();self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            for change in (dict(width=1024),dict(steps=8),dict(allow_approximation=False),dict(residency="component_staged")):
                with self.subTest(change=change):self.assertNotEqual(plan(change).returncode,0)
            env["TURBOCIDER_QWEN21_VIGGLE_LORA_FP16"]="1";self.assertNotEqual(plan().returncode,0);env.pop("TURBOCIDER_QWEN21_VIGGLE_LORA_FP16")
            result=plan(dict(loras=[],lora_strategy="auto",steps=40,allow_approximation=False))
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            env["TURBOCIDER_QWEN21_LORA_BF16_OPERANDS_FP32_RANKS"]="2"
            self.assertNotEqual(plan(dict(loras=[],lora_strategy="auto",steps=40)).returncode,0)
            self.assertFalse((root/"output.png").exists())


if __name__=="__main__":unittest.main()
