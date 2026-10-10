import os
import json
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU")=="1","explicit actual MLX/Metal opt-in required")
class SharedRanksTests(unittest.TestCase):
    def test_receipts_and_request_gates(self):
        mlx=Path(sysconfig.get_paths()["purelib"])/"mlx"
        lib=(ROOT/os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR","build/native")).resolve()
        env={key:value for key,value in os.environ.items() if not key.startswith("TURBOCIDER_")}
        with tempfile.TemporaryDirectory(prefix="tc-shared-rank-gates-") as directory:
            root=Path(directory);probe=root/"probe";request=root/"request.json"
            result=subprocess.run(["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror",
                "-mmacosx-version-min=26.2","-fobjc-arc","-I",str(ROOT/"native/core"),"-isystem",str(mlx/"include"),
                str(ROOT/"tests/native/shared_lora_rank_receipt_test.mm"),"-L",str(lib),"-lturbocider",
                "-L",str(mlx/"lib"),"-lmlx","-framework","Foundation",
                "-Wl,-rpath,"+str(lib),"-Wl,-rpath,"+str(mlx/"lib"),"-o",str(probe)],
                cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([str(probe)],cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr);print(result.stdout)
            base=dict(model="qwen-image-2.1",operation="image.generate",prompt="A ceramic teapot",
                output=str(root/"output.png"),width=512,height=512,steps=6,execution="gpu_ane",
                residency="resident",audio=False,allow_approximation=True,hybrid_mlp_mode="runtime",
                ane_manifest="checked-at-load.json",lora_strategy="inference_time",
                loras=[dict(path="Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors",strength=1.0,role="transformer")])
            def plan(change=None):
                request.write_text(json.dumps({**base,**(change or {})}))
                return subprocess.run([str(lib/"turbocider"),"plan",str(request)],cwd=ROOT,env=env,
                    capture_output=True,text=True,timeout=30)
            env.update(TURBOCIDER_ANE_BACKEND="private",TURBOCIDER_ALLOW_PRIVATE_ANE="1",
                TURBOCIDER_PRIVATE_ANE_CHANNELS="5120",TURBOCIDER_QWEN21_RUNTIME_SHARE_LORA_RANKS="1")
            result=plan();self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            for change in (dict(width=1024),dict(height=1024),dict(allow_approximation=False),dict(residency="component_staged")):
                with self.subTest(change=change):self.assertNotEqual(plan(change).returncode,0)
            for backend,channels in (("public","5120"),("private","0"),("private","auto")):
                env.update(TURBOCIDER_ANE_BACKEND=backend,TURBOCIDER_PRIVATE_ANE_CHANNELS=channels)
                with self.subTest(backend=backend,channels=channels):self.assertNotEqual(plan().returncode,0)
            env.update(TURBOCIDER_ANE_BACKEND="public",TURBOCIDER_PRIVATE_ANE_CHANNELS="0")
            # A valid flag on ordinary GPU is intentionally ignored, so the
            # optimized baseline can use the same environment without ANE.
            result=plan(dict(execution="gpu",hybrid_mlp_mode="auto",ane_manifest=""))
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            env["TURBOCIDER_QWEN21_RUNTIME_SHARE_LORA_RANKS"]="2"
            self.assertNotEqual(plan(dict(execution="gpu",hybrid_mlp_mode="auto",ane_manifest="")).returncode,0)
            self.assertFalse((root/"output.png").exists())

    def test_projection_delta_and_compiled_ffn_boundaries(self):
        mlx=Path(sysconfig.get_paths()["purelib"])/"mlx"
        lib=(ROOT/os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR","build/native")).resolve()
        with tempfile.TemporaryDirectory(prefix="tc-shared-ranks-") as directory:
            root=Path(directory);probe=root/"probe";adapter=root/"adapter.safetensors"
            compilation=subprocess.run(["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror",
                "-mmacosx-version-min=26.2","-I",str(ROOT/"native/core"),"-isystem",str(mlx/"include"),
                str(ROOT/"tests/native/lora_shared_ranks_test.cpp"),"-L",str(lib),"-lturbocider",
                "-L",str(mlx/"lib"),"-lmlx","-Wl,-rpath,"+str(lib),"-Wl,-rpath,"+str(mlx/"lib"),
                "-o",str(probe)],cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(compilation.returncode,0,compilation.stdout+compilation.stderr)
            result=subprocess.run([str(probe),str(adapter)],cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(result.stdout.count("PASS shared LoRA ranks"),6);print(result.stdout)


if __name__=="__main__":unittest.main()
