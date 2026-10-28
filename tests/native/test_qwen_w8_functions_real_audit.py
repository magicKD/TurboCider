"""Small host-only tensor reader/join rounder checks; no hardware or weights."""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/profiling/qwen_base'))
import w8_functions_real_audit as audit


def tensor_file(path,values,dtype='BF16',extra=b''):
    raw=values.tobytes();header=json.dumps({'x':dict(dtype=dtype,shape=list(values.shape),data_offsets=[0,len(raw)])}).encode()
    path.write_bytes(struct.pack('<Q',len(header))+header+raw+extra)


class RealFunctionAuditTests(unittest.TestCase):
    def test_bf16_rne_ties_sign_and_nonfinite_rejection(self):
        v=np.array([1.,-2.,1.+1./256,1.+3./256,-0.],dtype='<f4')
        self.assertEqual(audit.bf16_rne(v).tolist(),[0x3f80,0xc000,0x3f80,0x3f82,0x8000])
        for bad in (np.inf,-np.inf,np.nan):
            with self.assertRaises(ValueError):audit.bf16_rne(np.array([bad],dtype='<f4'))

    def test_exact_typed_extent_finite_quality_and_extra_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'a.safetensors';v=np.full((1,4,3),0x3f80,dtype='<u2');spec={'x':('BF16',(1,4,3))}
            tensor_file(path,v);t=audit.Tensors(path,spec)
            self.assertEqual(t.floats('x').tolist(),np.ones((1,4,3)).tolist())
            self.assertEqual(audit.quality(t,'x','x')['relative_l2'],0.)
            tensor_file(path,v,extra=b'x')
            with self.assertRaises(ValueError):audit.Tensors(path,spec)
            tensor_file(path,np.zeros((1,4,3),dtype='<f4'),'F32')
            with self.assertRaises(ValueError):audit.Tensors(path,spec)
            v[0,0,0]=0x7fc0;tensor_file(path,v);t=audit.Tensors(path,spec)
            with self.assertRaises(ValueError):audit.quality(t,'x','x')

    def test_full_payload_schema_keeps_f32_tail_and_original_bf16(self):
        s=audit.specifications()
        self.assertEqual(len(s),45)
        self.assertEqual(s['family_tail_1024_0'],('F32',(1,1024,4096)))
        self.assertEqual(s['hidden_1024_2'],('BF16',(1,1024,6144)))
        self.assertEqual(s['gpu_2560_0'],('BF16',(1,2560,4096)))


if __name__=='__main__':unittest.main()
