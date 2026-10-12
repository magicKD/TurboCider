"""Full Public W8 FFN exporter/shared executor, not end-to-end qualification."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
S=importlib.util.spec_from_file_location("public_w8_ffn_export",ROOT/"tools/coreml/export_runtime_w8a8_ffn.py")
E=importlib.util.module_from_spec(S);S.loader.exec_module(E)
try:
    import coremltools as ct
except ImportError:ct=None
CT26=ct is not None and hasattr(ct.target,"macOS26")


class PublicW8HostTests(unittest.TestCase):
    def test_geometry_and_invalid_recipes(self):
        for lora in (False,True):
            spec=E.geometry(4224,4096,12288,lora_inputs=lora)
            self.assertFalse(spec["production_qualified"])
            self.assertEqual(spec["outputs"]["y"],[4097+(12288 if lora else 0),4224])
            self.assertEqual(spec["graph_version"],2 if lora else 1)
        for args in ((True,128,512),(33,127,512),(4225,4096,12288),(33,128,513)):
            with self.assertRaises(ValueError):E.geometry(*args)
        with self.assertRaises(ValueError):E.geometry(33,128,512,basis="comfy_h256")
        self.assertEqual(E.geometry(33,4096,12288,16384)["tile_k"],16384)
        with self.assertRaises(ValueError):E.geometry(33,4096,12288,16385)

    def test_sylvester_rotation_matches_independent_unsigned64_sign_recipe(self):
        value=E.rotation(512,"sylvester_dh").reshape(512,512)
        mask=(1<<64)-1
        for out in (0,1,31,255,511):
            for col in range(512):
                x=(20260930+col*0x9e3779b97f4a7c15)&mask
                x=((x^(x>>30))*0xbf58476d1ce4e5b9)&mask
                x=((x^(x>>27))*0x94d049bb133111eb)&mask
                sign=-1 if (x^(x>>31))&1 else 1
                expected=np.float16(sign*(-1 if (out&col).bit_count()%2 else 1)/np.sqrt(512))
                self.assertEqual(value[out,col],expected)
        h4=np.array([[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]])
        full=h4
        for _ in range(3):full=np.kron(full,h4)
        np.testing.assert_array_equal(E.rotation(512,"comfy_h256").reshape(512,256),np.tile((full/16).astype(np.float16),(2,1)))

    @unittest.skipUnless(CT26,"isolated coremltools>=9 required")
    def test_graph_has_runtime_code_weights_scales_headroom_and_lora(self):
        from coremltools.converters.mil.mil import types
        for lora in (False,True):
            spec=E.geometry(33,128,512,128,lora)
            f=E.make_program(spec).functions["main"]
            for name in ("x","wg","wu","wd"):self.assertEqual(f.inputs[name].dtype,types.int8)
            for name in ("tx","sg","su","headroom"):self.assertEqual(f.inputs[name].dtype,types.fp16)
            self.assertEqual(set(f.inputs),set(spec["inputs"]))
            self.assertEqual(list(f.outputs)[0].shape,(129+(512 if lora else 0),33))
            self.assertEqual(sum(op.op_type=="quantize" for op in f.operations),1)

    @unittest.skipUnless(CT26,"isolated coremltools>=9 required")
    def test_full_k_is_explicit_three_projection_control_not_default(self):
        default=E.geometry(33,128,512,128,True)
        full=E.geometry(33,128,512,16384,True)
        a=E.make_program(default).functions["main"]
        b=E.make_program(full).functions["main"]
        self.assertEqual(sum(op.op_type=="matmul" for op in a.operations),6)
        self.assertEqual(sum(op.op_type=="matmul" for op in b.operations),3)
        self.assertEqual(a.outputs[0].shape,b.outputs[0].shape)
        self.assertEqual(set(a.inputs),set(b.inputs))
        self.assertEqual(E.geometry(33,128,512)["tile_k"],1024)
        self.assertFalse(full["production_qualified"])


@unittest.skipUnless(CT26 and os.environ.get("TURBOCIDER_TEST_PUBLIC_W8_FFN")=="1",
                     "isolated coremltools>=9/macOS26 and TURBOCIDER_TEST_PUBLIC_W8_FFN=1 required")
class PublicW8IntegrationTests(unittest.TestCase):
    def test_complete_gpu_coreml_gpu_executor_real_corrections_and_recovery(self):
        with tempfile.TemporaryDirectory(prefix="tc-public-w8-") as directory:
            root=Path(directory).resolve();build=root/"build";graph=root/"graph"
            E.export(graph,E.geometry(33,128,512,512,True))
            subprocess.run(["bash","tools/native/build_ane_public_w8_test.sh"],cwd=ROOT,check=True,
                env={**os.environ,"TURBOCIDER_NATIVE_OUT":str(build)},capture_output=True,text=True,timeout=120)
            for name in ("ane-public-w8-test","ane-public-w8-executor-test"):
                result=subprocess.run([str(build/name),str(graph/"manifest.json")],cwd=ROOT,check=True,
                    capture_output=True,text=True,timeout=120)
                self.assertIn("PASS",result.stdout)
                if name=="ane-public-w8-executor-test" and int(os.environ.get("TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_BYTES","0"))>0:
                    self.assertIn("PASS Public converted weight cache",result.stdout)
                print(result.stdout)
                strings=subprocess.check_output(["strings",str(build/name)],text=True)
                for private in ("_ANEClient","_ANERequest","_ANEIOSurfaceObject","_ANESharedEvents"):
                    self.assertNotIn(private,strings)
            # A same-size artifact corruption must fail before any inference.
            payload=graph/"graph.mlmodelc/coremldata.bin"
            data=payload.read_bytes();payload.write_bytes(bytes([data[0]^1])+data[1:])
            result=subprocess.run([str(build/"ane-public-w8-test"),str(graph/"manifest.json")],cwd=ROOT,
                capture_output=True,text=True,timeout=30)
            self.assertNotEqual(result.returncode,0)
            self.assertIn("digest mismatch",result.stderr)


if __name__=="__main__":unittest.main()
