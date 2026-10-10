import json
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
LIB=(ROOT/os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR","build/native")).resolve()


@unittest.skipUnless((LIB/"libturbocider.dylib").is_file(),"current native library required")
class ConvRotBf16PartialTests(unittest.TestCase):
    def test_explicit_plan_scope_and_mutually_exclusive_recipes(self):
        env={k:v for k,v in os.environ.items() if not k.startswith("TURBOCIDER_")}
        env.update(TURBOCIDER_Z_CONVROT_BF16_PARTIAL="1",TURBOCIDER_Z_RUNTIME_CONVROT="1",
            TURBOCIDER_ANE_BACKEND="private",TURBOCIDER_ALLOW_PRIVATE_ANE="1",TURBOCIDER_PRIVATE_ANE_CHANNELS="4096",
            TURBOCIDER_PRIVATE_ANE_DATA_PATH="convrot_w8a8",TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN="1")
        with tempfile.TemporaryDirectory(prefix="tc-convrot-bf16-partial-plan-") as folder:
            root=Path(folder);request=root/"request.json"
            base=dict(model="z-image-turbo",operation="image.generate",prompt="A red fox",output=str(root/"out.png"),width=512,height=512,
                steps=4,execution="gpu_ane",residency="resident",audio=False,allow_approximation=True,hybrid_mlp_mode="runtime",ane_manifest="checked-at-load.json")
            def plan(change=None):
                request.write_text(json.dumps({**base,**(change or {})}))
                return subprocess.run([str(LIB/"turbocider"),"plan",str(request)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
            result=plan();self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            for change in (dict(width=1024),dict(allow_approximation=False),dict(residency="streamed"),
                dict(lora_strategy="inference_time",loras=[dict(path="unused.safetensors",strength=1.,role="transformer")])):
                self.assertNotEqual(plan(change).returncode,0)
            for key,value in (("TURBOCIDER_ANE_BACKEND","public"),("TURBOCIDER_PRIVATE_ANE_CHANNELS","0"),
                ("TURBOCIDER_PRIVATE_ANE_CHANNELS","auto"),("TURBOCIDER_PRIVATE_ANE_DATA_PATH","w8a8"),
                ("TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN","0"),("TURBOCIDER_Z_RUNTIME_CONVROT","0"),
                ("TURBOCIDER_Z_CONVROT_FP32_SCALES","1"),("TURBOCIDER_Z_CONVROT_FP32_MPP","1")):
                before=env.get(key);env[key]=value;self.assertNotEqual(plan().returncode,0)
                if before is None:env.pop(key)
                else:env[key]=before
            env["TURBOCIDER_Z_RUNTIME_CONVROT_LORA"]="1"
            result=plan(dict(lora_strategy="inference_time",loras=[dict(path="unused.safetensors",strength=1.,role="transformer")]))
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            gpu=dict(execution="gpu",hybrid_mlp_mode="auto",ane_manifest="")
            self.assertEqual(plan(gpu).returncode,0)
            for bad in ("","2","true"):
                env["TURBOCIDER_Z_CONVROT_BF16_PARTIAL"]=bad;self.assertNotEqual(plan(gpu).returncode,0)
            self.assertFalse((root/"out.png").exists())

    @unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU")=="1","actual MLX/Metal opt-in required")
    def test_original_quantized_boundary_and_complete_adapter_source_contract(self):
        mlx=Path(sysconfig.get_paths()["purelib"])/"mlx"
        with tempfile.TemporaryDirectory(prefix="tc-convrot-bf16-partial-") as folder:
            root=Path(folder);probe=root/"probe"
            result=subprocess.run(["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror","-mmacosx-version-min=26.2",
                "-I",str(ROOT/"native/core"),"-isystem",str(mlx/"include"),str(ROOT/"tests/native/convrot_bf16_partial_test.cpp"),
                "-L",str(LIB),"-lturbocider","-L",str(mlx/"lib"),"-lmlx","-Wl,-rpath,"+str(LIB),"-Wl,-rpath,"+str(mlx/"lib"),"-o",str(probe)],
                cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([str(probe),str(root/"adapter.safetensors")],cwd=ROOT,capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("PASS BF16 partial cases=18",result.stdout);print(result.stdout)


if __name__=="__main__":unittest.main()
