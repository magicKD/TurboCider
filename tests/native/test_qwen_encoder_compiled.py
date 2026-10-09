import json
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
LIB=(ROOT/os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR","build/native")).resolve()


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU")=="1","actual MLX/Metal opt-in required")
class CompiledEncoderTests(unittest.TestCase):
    def test_dynamic_source_and_language_boundaries(self):
        mlx=Path(sysconfig.get_paths()["purelib"])/"mlx"
        with tempfile.TemporaryDirectory(prefix="tc-compiled-encoder-") as folder:
            probe=Path(folder)/"probe"
            result=subprocess.run(["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror","-mmacosx-version-min=26.2",
                "-I",str(ROOT/"native/core"),"-isystem",str(mlx/"include"),str(ROOT/"tests/native/qwen21_encoder_compiled_test.cpp"),
                "-L",str(LIB),"-lturbocider","-L",str(mlx/"lib"),"-lmlx","-Wl,-rpath,"+str(LIB),"-Wl,-rpath,"+str(mlx/"lib"),"-o",str(probe)],
                cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([str(probe)],cwd=ROOT,capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("PASS compiled encoder cases=48",result.stdout);print(result.stdout)

    def test_request_scope_and_profile_snapshot(self):
        env={k:v for k,v in os.environ.items() if not k.startswith("TURBOCIDER_")}
        env["TURBOCIDER_QWEN21_ENCODER_COMPILED_GPU"]="1"
        with tempfile.TemporaryDirectory(prefix="tc-compiled-encoder-plan-") as folder:
            root=Path(folder);request=root/"request.json"
            base=dict(model="qwen-image-2.1",operation="image.generate",prompt="A teapot",output=str(root/"out.png"),
                width=512,height=512,steps=40,execution="gpu",residency="resident",audio=False,allow_approximation=True)
            def plan(change=None):
                request.write_text(json.dumps({**base,**(change or {})}))
                return subprocess.run([str(LIB/"turbocider"),"plan",str(request)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
            self.assertEqual(plan().returncode,0)
            ref=dict(kind="image",role="reference",path="checked-at-load.png")
            for count in (1,2):
                self.assertEqual(plan(dict(operation="image.edit",inputs=[ref]*count,qwen21_reference_size=512)).returncode,0)
            for change in (dict(width=1024),dict(allow_approximation=False),dict(residency="component_staged"),
                dict(memory_budget_bytes=1<<30),dict(prompt_enhance=True),
                dict(operation="image.edit",inputs=[ref]*3,qwen21_reference_size=512)):
                self.assertNotEqual(plan(change).returncode,0)
            for bad in ("","2","true"):
                env["TURBOCIDER_QWEN21_ENCODER_COMPILED_GPU"]=bad;self.assertNotEqual(plan().returncode,0)
            self.assertFalse((root/"out.png").exists())


if __name__=="__main__":unittest.main()
