"""Actual shared FFN Qwen3-VL language encoder; no downloaded checkpoint."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
S=importlib.util.spec_from_file_location("encoder_export",ROOT/"tools/coreml/export_runtime_ane.py")
EXPORT=importlib.util.module_from_spec(S);S.loader.exec_module(EXPORT)
LIB=(ROOT/os.environ.get("TURBOCIDER_NATIVE_LIBRARY_DIR","build/native")).resolve()


def compile_probe(source, output):
    mlx=Path(sysconfig.get_paths()["purelib"])/"mlx"
    command=["xcrun","clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror",
        "-mmacosx-version-min=26.2","-I",str(ROOT/"native"),"-I",str(ROOT/"native/core"),
        "-isystem",str(mlx/"include"),str(ROOT/"tests/native"/source),
        "-L",str(LIB),"-lturbocider","-L",str(mlx/"lib"),"-lmlx","-Wl,-rpath,"+str(LIB),
        "-Wl,-rpath,"+str(mlx/"lib"),"-o",str(output)]
    if source.endswith(".mm"):command.extend(["-fobjc-arc","-framework","Foundation"])
    subprocess.run(command,check=True,cwd=ROOT,capture_output=True,text=True,timeout=60)


@unittest.skipUnless((LIB/"libturbocider.dylib").is_file(),"native library required")
class EncoderContractTests(unittest.TestCase):
    def test_shared_cache_scope_and_actual_receipts(self):
        with tempfile.TemporaryDirectory(prefix="tc-qwen-encoder-contract-") as directory:
            for source in ("qwen21_conditioning_cache_test.cpp","qwen21_encoder_receipt_test.mm"):
                probe=Path(directory)/Path(source).stem
                compile_probe(source,probe)
                result=subprocess.run([str(probe)],cwd=ROOT,capture_output=True,text=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr);print(result.stdout)

    def test_explicit_encoder_request_gates(self):
        cli=LIB/"turbocider"
        self.assertTrue(cli.is_file(),"thin native CLI required for request contract")
        env={k:v for k,v in os.environ.items() if not k.startswith("TURBOCIDER_")}
        with tempfile.TemporaryDirectory(prefix="tc-qwen-encoder-gates-") as directory:
            root=Path(directory);request=root/"request.json"
            base=dict(model="qwen-image-2.1",operation="image.generate",prompt="A teapot",
                output=str(root/"output.png"),width=512,height=512,steps=40,execution="gpu",
                residency="resident",audio=False,allow_approximation=True,encoder_ane_manifest="checked-at-load.json")
            def plan(change):
                request.write_text(json.dumps({**base,**change}))
                return subprocess.run([str(cli),"plan",str(request)],cwd=ROOT,env=env,
                    capture_output=True,text=True,timeout=30)
            reference=dict(kind="image",role="reference",path="checked-at-load.png")
            for count in (0,1,2):
                change={} if not count else dict(operation="image.edit",inputs=[reference]*count)
                result=plan(change)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertEqual(json.loads(result.stdout)["execution"],"gpu")
                self.assertFalse((root/"output.png").exists())
            for change in (dict(allow_approximation=False),dict(residency="component_staged"),
                dict(width=1024),dict(height=1024),dict(prompt_enhance=True),
                dict(operation="image.edit",inputs=[]),dict(operation="image.edit",inputs=[reference]*3)):
                with self.subTest(change=change):self.assertNotEqual(plan(change).returncode,0)
            env["TURBOCIDER_QWEN21_ENCODER_RETAIN_RUNTIME"]="1"
            self.assertEqual(plan({}).returncode,0)
            self.assertNotEqual(plan(dict(memory_budget_bytes=1<<30)).returncode,0)
            env["TURBOCIDER_QWEN21_ENCODER_RETAIN_RUNTIME"]="2"
            self.assertNotEqual(plan({}).returncode,0)


@unittest.skipUnless(os.environ.get("TURBOCIDER_TEST_QWEN_ENCODER_ANE")=="1","explicit actual encoder GPU/Core ML opt-in required")
class EncoderAneTests(unittest.TestCase):
    def test_causal_language_and_visual_deepstack_shared_runtime(self):
        with tempfile.TemporaryDirectory(prefix="tc-qwen-encoder-ane-") as folder:
            root=Path(folder);graph=root/"graph";probe=root/"probe"
            EXPORT.export(graph,EXPORT.geometry("swiglu",33,128,512,128,128))
            compile_probe("qwen21_encoder_ane_test.cpp",probe)
            env={k:v for k,v in os.environ.items() if not k.startswith(("TURBOCIDER_PRIVATE_ANE_","TURBOCIDER_RUNTIME_ANE_"))}
            env.update(TURBOCIDER_ANE_BACKEND="public",TURBOCIDER_RUNTIME_ANE_CHUNKS="1",TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC="1")
            result=subprocess.run([str(probe),str(graph/"manifest.json")],cwd=ROOT,env=env,
                capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertEqual(result.stdout.count("PASS encoder"),16);print(result.stdout)
            self.assertIn("PASS retained encoder source scope",result.stdout)
            self.assertIn("PASS compiled encoder GPU fallback",result.stdout)


if __name__=="__main__":unittest.main()
