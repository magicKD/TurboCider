"""Program/export semantics, not ANE residency or arithmetic certification."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location('runtime_qdq_export',ROOT/'tools/coreml/export_runtime_w8a8_probe.py')
E=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(E)


class RuntimeExportTests(unittest.TestCase):
    def test_geometry_and_no_false_qualification(self):
        s=E.geometry(32,256,512,'runtime_qdq')
        self.assertFalse(s['production_qualified']);self.assertEqual(s['arithmetic_evidence'],'unknown')
        for dims in [(True,256,512),(0,256,512),(32,-1,512),(32,256,32769)]:
            with self.assertRaises(ValueError):E.geometry(*dims,'runtime_qdq')
        with self.assertRaises(ValueError):E.geometry(1,1,1,'unknown')
        with self.assertRaises(ValueError):E.geometry(32,10240,3840,'runtime_qdq',1,1)

    def test_dynamic_graph_does_not_capture_checkpoint_and_fixed_scale(self):
        for arm in ('fp16','runtime_qdq'):
            spec=E.geometry(33,256,65,arm,128,32);program=E.make_program(spec)
            function=program.functions['main'];self.assertEqual(set(function.inputs),{'x','w'})
            ops=list(function.operations)
            if arm=='runtime_qdq':
                self.assertEqual(sum(op.op_type=='quantize' for op in ops),2)
                for op in ops:
                    if op.op_type in ('quantize','dequantize'):
                        self.assertEqual(float(op.scale.val),1/128)
            self.assertEqual(sum(op.op_type=='matmul' for op in ops),6)

    def test_frozen_control_identity_is_separate(self):
        spec=E.geometry(32,256,512,'frozen_qdq')
        w=np.random.default_rng(4).integers(-128,128,size=(512,256),dtype=np.int8)
        p=E.make_program(spec,w);self.assertEqual(set(p.functions['main'].inputs),{'x'})
        with self.assertRaises(ValueError):E.make_program(spec,None)
        with self.assertRaises(ValueError):E.make_program(spec,w.astype(np.float16))

if __name__=='__main__':unittest.main()
