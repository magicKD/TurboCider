"""Independent W8 consumer-oracle negatives; CPU only, no model/ANE."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/profiling/qwen_base'))
from multi_function_collect import audit_w8_arrays


def fixture(root):
    trial=root/'trial';trial.mkdir()
    report=dict(same_slot_bytes=True,column_view_negative_controls=3,runs=[])
    for variant in range(2):
        wg=np.zeros((512,128),dtype='<f4');wu=wg.copy();wd=np.zeros((128,512),dtype='<f4')
        for c in range(512):wg[c,c%128]=wu[c,c%128]=.25 if variant else .125
        for c in range(128):wd[c,c]=-.5 if variant else .25
        for j,value in enumerate((wg,wu,wd)):value.tofile(trial/f'weight-{variant}-{j}.bin')
    for i,count in enumerate((32,64,96,32)):
        variant=int(i==2);x=np.zeros((128,128),dtype='<f4');x[:count]=4.
        x.tofile(trial/f'run-{i}-input.bin')
        g=x[:count].astype(np.float64)*(.25 if variant else .125)
        expected=g/(1.+np.exp(-g))*g*(-.5 if variant else .25)
        arms=[]
        for arm in ('legacy','family'):
            launch=count if arm=='family' else ((count+63)//64)*64
            y=np.full((128,128),1234,dtype='<f4');y[:launch]=0;y[:count]=expected
            y.tofile(trial/f'run-{i}-{arm}.bin')
            arms.append(dict(launch_rows=launch,calls=(count+63)//64,
                             policy=dict(fixed=True,instances=1,blocked_rebuilds=0,disabled=False)))
        report['runs'].append(dict(index=i,rows=count,weight_variant=variant,arms=arms))
    return report


class W8ConsumerOracleTests(unittest.TestCase):
    def test_complete_nonlinear_full_k_reference_and_restoration(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=fixture(root);self.assertEqual(len(audit_w8_arrays(root,r)),8)

    def test_nonfinite_source_output_wrong_tail_policy_and_delta_fail(self):
        for kind in ('source','output','tail','policy','delta','signed-zero'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                root=Path(d);r=fixture(root)
                if kind=='source':
                    p=root/'trial/weight-0-0.bin';v=np.fromfile(p,dtype='<f4');v[0]=np.nan;v.tofile(p)
                elif kind=='policy':r['runs'][0]['arms'][0]['policy']['instances']=2
                elif kind=='signed-zero':
                    p=root/'trial/run-3-legacy.bin';v=np.fromfile(p,dtype='<f4').reshape(128,128)
                    v[40,0]=np.float32(-0.);v.tofile(p)
                else:
                    p=root/'trial/run-0-family.bin';v=np.fromfile(p,dtype='<f4').reshape(128,128)
                    if kind=='output':v[0,0]=np.inf
                    if kind=='tail':v[32,0]=0.
                    if kind=='delta':v[:32]*=1.02
                    v.tofile(p)
                with self.assertRaises(ValueError):audit_w8_arrays(root,r)


if __name__=='__main__':unittest.main()
