"""Source routing guards; numerical parity is a separate opt-in model screen."""
from pathlib import Path
import json
import os
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]

CREATE_PROBE=r'''
import ctypes as C,json,sys
lib=C.CDLL(sys.argv[1])
lib.tc_engine_create_model.argtypes=[C.c_char_p,C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
lib.tc_engine_free.argtypes=[C.c_void_p];lib.tc_string_free.argtypes=[C.c_void_p]
engine,error=C.c_void_p(),C.c_void_p()
status=lib.tc_engine_create_model(b"z-image-turbo",sys.argv[2].encode(),C.byref(engine),C.byref(error))
message=C.string_at(error).decode() if error.value else None
if error.value:lib.tc_string_free(error)
if engine.value:lib.tc_engine_free(engine)
print(json.dumps(dict(status=status,error=message)))
'''


class RuntimeBoundarySourceTests(unittest.TestCase):
    def test_dense_runtime_and_gpu_share_arithmetic_not_frozen_hybrid_bodies(self):
        source=(ROOT/"native/models/z_image/z_image.cpp").read_text()
        whole=source.split("&z_gpu_block_graph()",1)[1].split("Tensor z_compiled_gpu_block",1)[0]
        pre=source.split("ZImageGpuGraph &z_runtime_pre_graph()",1)[1].split("ZImageGpuGraph &z_runtime_post_graph()",1)[0]
        ffn=source.split("ZImageGpuGraph &z_runtime_full_ffn_graph()",1)[1].split("bool z_dense_split_gpu_control()",1)[0]
        self.assertIn("z_dense_gpu_pre_body(args)",whole)
        self.assertIn("z_dense_gpu_pre_body(a)",pre)
        self.assertIn("z_dense_gpu_ffn_body(",whole)
        self.assertIn("z_dense_gpu_ffn_body(",ffn)
        runtime=source.split("Tensor z_runtime_block(",1)[1].split("Tensor z_context_block(",1)[0]
        self.assertNotIn("z_hybrid_pre_graph(",runtime)
        self.assertNotIn("z_hybrid_post_graph(",runtime)
        self.assertIn("z_runtime_pre_graph()",runtime)
        self.assertIn("z_runtime_post_graph()",runtime)
        self.assertIn("gguf_compatibility || w.convrot",runtime)
        self.assertIn("w.has_runtime_loras()",runtime)

    def test_diagnostic_is_gated_gpu_only_and_not_a_runtime_executor(self):
        source=(ROOT/"native/models/z_image/z_image.cpp").read_text()
        option=source.split("bool z_dense_split_gpu_control()",1)[1].split("Tensor z_compiled_split_gpu_block(",1)[0]
        self.assertIn("TURBOCIDER_ENABLE_QUANTIZED_EXECUTION_EXPERIMENTS",option)
        self.assertIn("qe_capability_unqualified",option)
        self.assertIn('std::string_view(raw)=="0" || std::string_view(raw)=="1"',option)
        control=source.split("Tensor z_compiled_split_gpu_block(",1)[1].split("Tensor z_runtime_block(",1)[0]
        for forbidden in ("runtime.run", "runtime.stage", "runtime.plan_block", "Surface", "Executor"):
            self.assertNotIn(forbidden,control)
        self.assertIn("mx::eval(pre[1])",control)
        self.assertIn("mx::eval(feed)",control)
        self.assertIn("dense split GPU control requires resident dense GPU",source)
        self.assertIn('result.backend="mlx_cpp_metal_dense_split_gpu_control"',source)
        results=(ROOT/"native/platform/apple/results.mm").read_text()
        self.assertIn('result.backend=="mlx_cpp_metal_dense_split_gpu_control"',results)
        self.assertIn('@"compiled_split_gpu_ffn_control"',results)


@unittest.skipUnless(os.environ.get("TC_ORDINARY_TEST_LIBRARY"),"explicit ordinary native build/local fixtures required")
class RuntimeBoundaryOrdinaryGateTests(unittest.TestCase):
    def test_public_release_rejects_control_before_loading_weights(self):
        library=str(Path(os.environ["TC_ORDINARY_TEST_LIBRARY"]).resolve(strict=True))
        model=ROOT/"models/Comfy-Org-z_image_turbo"
        self.assertTrue(model.is_dir(),"local base fixture missing")
        base={k:v for k,v in os.environ.items() if not k.startswith("TURBOCIDER_")}
        for value,error in (("1","qe_capability_unqualified"),("yes","qe_config_conflict"),("0",None)):
            with self.subTest(value=value):
                run=subprocess.run([sys.executable,"-c",CREATE_PROBE,library,str(model)],cwd=ROOT,
                    env={**base,"TURBOCIDER_Z_DENSE_SPLIT_GPU_CONTROL":value},capture_output=True,text=True,timeout=20)
                self.assertEqual(run.returncode,0,run.stderr)
                row=json.loads(run.stdout)
                if error:
                    self.assertNotEqual(row["status"],0)
                    self.assertIn(error,row["error"])
                else:self.assertEqual(row["status"],0,row["error"])


if __name__=="__main__":unittest.main(verbosity=2)
