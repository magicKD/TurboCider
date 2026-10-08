import os
import json
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest
import importlib.util

ROOT=Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_GPU")=="1","explicit actual MLX opt-in required")
class DownRanksTests(unittest.TestCase):
    def test_request_gates_without_loading_models(self):
        lib=(ROOT/os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR","build/native")).resolve()
        env={key:value for key,value in os.environ.items() if not key.startswith("TURBOCIDER_")}
        env.update(TURBOCIDER_ANE_BACKEND="private",TURBOCIDER_ALLOW_PRIVATE_ANE="1",
            TURBOCIDER_PRIVATE_ANE_CHANNELS="5120",TURBOCIDER_QWEN21_RUNTIME_SPLIT_DOWN_RANKS="1")
        with tempfile.TemporaryDirectory(prefix="tc-down-rank-gates-") as folder:
            root=Path(folder);request=root/"request.json"
            base=dict(model="qwen-image-2.1",operation="image.generate",prompt="A ceramic teapot",
                output=str(root/"output.png"),width=512,height=512,steps=6,execution="gpu_ane",
                residency="resident",audio=False,allow_approximation=True,hybrid_mlp_mode="runtime",
                ane_manifest="checked-at-load.json",lora_strategy="inference_time",
                loras=[dict(path="Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors",strength=1.0,role="transformer")])
            def plan(change=None):
                request.write_text(json.dumps({**base,**(change or {})}))
                return subprocess.run([str(lib/"turbocider"),"plan",str(request)],cwd=ROOT,env=env,
                    capture_output=True,text=True,timeout=30)
            result=plan();self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            for change in (dict(width=1024),dict(height=1024),dict(allow_approximation=False),dict(residency="component_staged")):
                with self.subTest(change=change):self.assertNotEqual(plan(change).returncode,0)
            for backend,channels in (("public","5120"),("private","0"),("private","auto")):
                env.update(TURBOCIDER_ANE_BACKEND=backend,TURBOCIDER_PRIVATE_ANE_CHANNELS=channels)
                with self.subTest(backend=backend,channels=channels):self.assertNotEqual(plan().returncode,0)
            env.update(TURBOCIDER_ANE_BACKEND="private",TURBOCIDER_PRIVATE_ANE_CHANNELS="5120",
                TURBOCIDER_QWEN21_VIGGLE_LORA_FP16="1")
            self.assertNotEqual(plan().returncode,0)
            env.pop("TURBOCIDER_QWEN21_VIGGLE_LORA_FP16")
            env.update(TURBOCIDER_ANE_BACKEND="public",TURBOCIDER_PRIVATE_ANE_CHANNELS="0")
            # Valid flags on ordinary GPU/base controls remain inert.
            result=plan(dict(execution="gpu",hybrid_mlp_mode="auto",ane_manifest=""))
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=plan(dict(loras=[],lora_strategy="auto",execution="gpu",hybrid_mlp_mode="auto",ane_manifest=""))
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            env["TURBOCIDER_QWEN21_RUNTIME_SPLIT_DOWN_RANKS"]="2"
            self.assertNotEqual(plan(dict(execution="gpu",hybrid_mlp_mode="auto",ane_manifest="")).returncode,0)
            self.assertFalse((root/"output.png").exists())

    @unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_PRIVATE_CHANNEL_MLX")=="1","explicit Private channel opt-in required")
    def test_actual_channel_failures_and_partial_rank_cleanup(self):
        spec=importlib.util.spec_from_file_location("down_rank_export",ROOT/"tools/coreml/export_runtime_ane.py")
        export=importlib.util.module_from_spec(spec);spec.loader.exec_module(export)
        mlx=Path(sysconfig.get_paths()["purelib"])/"mlx"
        lib=(ROOT/os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR","build/native")).resolve()
        with tempfile.TemporaryDirectory(prefix="tc-down-rank-channel-") as folder:
            root=Path(folder).resolve();probe=root/"probe";graph=root/"graph"
            export.export(graph,export.geometry("swiglu",33,128,1024,128,128,lora_inputs=True))
            result=subprocess.run(["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror",
                "-mmacosx-version-min=26.2","-I",str(ROOT/"native/core"),"-isystem",str(mlx/"include"),
                str(ROOT/"tests/native/down_rank_channel_test.cpp"),"-L",str(lib),"-lturbocider",
                "-L",str(mlx/"lib"),"-lmlx","-Wl,-rpath,"+str(lib),"-Wl,-rpath,"+str(mlx/"lib"),"-o",str(probe)],
                cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            env={key:value for key,value in os.environ.items() if not key.startswith("TURBOCIDER_")}
            env.update(TURBOCIDER_ANE_BACKEND="private",TURBOCIDER_ALLOW_PRIVATE_ANE="1",
                TURBOCIDER_PRIVATE_ANE_DATA_PATH="w8a8",TURBOCIDER_PRIVATE_ANE_GPU_IO="1",
                TURBOCIDER_PRIVATE_ANE_CHANNELS="512",TURBOCIDER_RUNTIME_ANE_CHUNKS="1",
                TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC="1",TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN="1",
                TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE="1")
            result=subprocess.run([str(probe),str(graph/"manifest.json"),str(root/"adapter.safetensors")],
                cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("PASS down-rank channel",result.stdout);print(result.stdout)

    def test_split_input_ranks_and_one_output_projection(self):
        mlx=Path(sysconfig.get_paths()["purelib"])/"mlx"
        lib=(ROOT/os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR","build/native")).resolve()
        with tempfile.TemporaryDirectory(prefix="tc-down-ranks-") as folder:
            root=Path(folder);probe=root/"probe"
            result=subprocess.run(["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror",
                "-mmacosx-version-min=26.2","-I",str(ROOT/"native/core"),"-isystem",str(mlx/"include"),
                str(ROOT/"tests/native/lora_down_ranks_test.cpp"),"-L",str(lib),"-lturbocider",
                "-L",str(mlx/"lib"),"-lmlx","-Wl,-rpath,"+str(lib),"-Wl,-rpath,"+str(mlx/"lib"),"-o",str(probe)],
                cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=subprocess.run([str(probe),str(root/"adapter.safetensors")],cwd=ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn("PASS down rank shards cases=18",result.stdout);print(result.stdout)


if __name__=="__main__":unittest.main()
