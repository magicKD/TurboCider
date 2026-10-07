"""Public INT8 IO controls; SDK/type acceptance is not engine residency proof."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location("public_int8_io_export",ROOT/"tools/coreml/export_runtime_int8_io_probe.py")
EXPORT=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(EXPORT)
try:
    import coremltools as ct
except ImportError:
    ct=None
HAS_CT26=ct is not None and hasattr(ct.target,"macOS26")


class PublicInt8IOHostTests(unittest.TestCase):
    def test_geometry_never_claims_production_or_arithmetic_and_bounds_growth(self):
        for arm in ("int8","fp16"):
            spec=EXPORT.geometry(4224,4096,7168,arm)
            self.assertEqual(spec["inputs"],{"x":[4096,4224],"w":[7168,4096]})
            self.assertFalse(spec["production_qualified"])
            self.assertEqual(spec["arithmetic_evidence"],"unknown")
            self.assertEqual(spec["minimum_macos"],"26.0")
        for dims in ((True,128,64),(0,128,64),(4225,4096,7168),(33,4097,64),(33,128,16385)):
            with self.assertRaises(ValueError):EXPORT.geometry(*dims,"int8")
        with self.assertRaises(ValueError):EXPORT.geometry(33,128,64,"unknown")

    def test_legacy_toolchain_declines_without_touching_destination(self):
        if ct is None or HAS_CT26:self.skipTest("requires retained pre-macOS26 coremltools control")
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"not-published"
            with self.assertRaisesRegex(RuntimeError,"coremltools>=9"):
                EXPORT.export(output,EXPORT.geometry(33,128,64,"int8"))
            self.assertFalse(output.exists())

    @unittest.skipUnless(HAS_CT26,"isolated coremltools>=9 required for declared INT8 IO")
    def test_graph_inputs_are_dynamic_codes_not_checkpoint_constants(self):
        from coremltools.converters.mil.mil import types
        for arm in ("int8","fp16"):
            program=EXPORT.make_program(EXPORT.geometry(33,128,64,arm))
            f=program.functions["main"]
            self.assertEqual(set(f.inputs),{"x","w"})
            self.assertEqual(f.inputs["x"].dtype,types.int8 if arm=="int8" else types.fp16)
            ops=list(f.operations)
            self.assertEqual(sum(op.op_type=="dequantize" for op in ops),2 if arm=="int8" else 0)
            self.assertEqual(sum(op.op_type=="quantize" for op in ops),0)
            for op in ops:
                if op.op_type=="const":self.assertLess(len(op.outputs[0].shape),2)
                if op.op_type=="dequantize":self.assertEqual(float(op.scale.val),1/128)


class PrivateQdqRewriteHostTests(unittest.TestCase):
    def test_rewrite_preserves_graph_boundary_and_rejects_nonidentical_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            binary=Path(directory)/"graph-test"
            subprocess.run(["clang++","-std=c++20","-O2","-Wall","-Wextra","-Werror",
                "tests/native/ane_qdq_share_graph_test.cpp","native/backends/private/ane_mil.cpp","-o",str(binary)],
                cwd=ROOT,check=True,capture_output=True,text=True,timeout=120)
            result=subprocess.run([str(binary)],check=True,capture_output=True,text=True,timeout=30)
            self.assertIn("PASS 12 shared-QDQ graph controls",result.stdout)


@unittest.skipUnless(sys.platform=="darwin" and HAS_CT26 and os.environ.get("TURBOCIDER_TEST_PUBLIC_INT8_IO")=="1",
                     "isolated coremltools>=9/macOS26 and TURBOCIDER_TEST_PUBLIC_INT8_IO=1 required")
class PublicInt8IOIntegrationTests(unittest.TestCase):
    def test_actual_public_signed_codes_aba_and_scoped_output_backing(self):
        with tempfile.TemporaryDirectory(prefix="tc-public-int8-io-") as directory:
            root=Path(directory).resolve();build=root/"build"
            subprocess.run(["bash","tools/native/build_public_runtime_int8_io_probe.sh"],cwd=ROOT,check=True,
                env={**os.environ,"TURBOCIDER_NATIVE_OUT":str(build)},capture_output=True,text=True,timeout=120)
            strings=subprocess.check_output(["strings",str(build/"probe")],text=True)
            for name in ("_ANEClient","_ANERequest","_ANEIOSurfaceObject","_ANESharedEvents"):
                self.assertNotIn(name,strings)
            slots={}
            for arm in ("fp16","int8"):
                graph=root/arm
                EXPORT.export(graph,EXPORT.geometry(33,128,64,arm))
                for policy in ("cpu","ne"):
                    result=subprocess.run([str(build/"probe"),str(graph/"manifest.json"),policy],cwd=ROOT,
                        check=True,capture_output=True,text=True,timeout=120)
                    receipt=json.loads(result.stdout)
                    self.assertEqual(receipt["actual_input_dtype"],arm)
                    self.assertEqual(receipt["calls"],21)
                    self.assertEqual(receipt["checked_values"],44352)
                    self.assertEqual(receipt["max_abs"],0)
                    self.assertTrue(receipt["signed_minus128_checked"])
                    self.assertTrue(receipt["weight_switch_aba_checked"])
                    self.assertEqual(receipt["output_backing_identity_hits"],21)
                    self.assertFalse(receipt["production_qualified"])
                    self.assertEqual(receipt["observed_ane_residency"],"unknown")
                    self.assertEqual(len(receipt["samples"]),15)
                    slots[arm]=receipt["slot_bytes"]
            self.assertLess(slots["int8"],slots["fp16"])


if __name__=="__main__":unittest.main()
