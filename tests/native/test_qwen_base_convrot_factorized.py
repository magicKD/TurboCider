"""CPU-only ready-domain factored-Q8 projection accounting and no-bank contracts."""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/profiling/qwen_base'))
import convrot_factorized_collect as screen


def report(real=False,n=16,bm=32,bn=64):
    m,k,out,wp,rb,cb=(1024,6144,4096,12288,0,0) if real else (33,128,257,320,5,32)
    return dict(schema='tc-convrot-row-scale-dot-v1',status='complete',real_weights=real,complete_request=False,image_export=False,ane_predictions=0,
        candidate_owned_dense_weight_bytes=0,source_unchanged=True,rejected=8,bm=bm,bn=bn,bk=64,repeats=n,warmups=2,m=m,k=k,n=out,
        physical_pitch=wp,row_begin=rb,column_begin=cb,strong_f32_control_weight_bytes=2*out*k*4,
        checks={v:{a:dict(relative_l2=1e-7,max_abs=1e-6,elements=m*out) for a in ('packed','factored')} for v in ('a','b','w')},
        samples=[dict(arm=screen.ARMS[(j+rep+3)%3],rep=rep,warmup=rep<0,input_variant=bool(rep&1),layer=int((rep+2)%4==2),seconds=.004)
            for rep in range(-2,n) for j in range(3)])


class FactoredRowScaleTests(unittest.TestCase):
    def test_original_physical_views_and_all_ready_controls(self):
        for real in (False,True):
            for bm,bn in ((32,64),(32,128),(64,64)):
                self.assertEqual(screen.validate(report(real,16,bm,bn),16,real,bm,bn)['measured_dots'],48)
        for change in (dict(physical_pitch=6144),dict(column_begin=1),dict(ane_predictions=1),dict(candidate_owned_dense_weight_bytes=1),dict(strong_f32_control_weight_bytes=0)):
            with self.assertRaises(ValueError):screen.validate({**report(True),**change},16,True,32,64)

    def test_full_f32_quality_is_not_bf16_output_tolerance(self):
        r=report(True);r['checks']['w']['factored']['relative_l2']=1e-4
        with self.assertRaises(ValueError):screen.validate(r,16,True,32,64)
        r=report();r['checks']['a']['packed']['elements']=1
        with self.assertRaises(ValueError):screen.validate(r,16,False,32,64)

    def test_warm_rotations_true_weight_changes_and_finite_completed_times(self):
        for change in (lambda r:r['samples'].reverse(),lambda r:r['samples'].pop(),lambda r:r['samples'][0].update(layer=1),
            lambda r:r['samples'][0].update(warmup=False),lambda r:r['samples'][0].update(seconds=float('nan'))):
            r=report();change(r)
            with self.assertRaises(ValueError):screen.validate(r,16,False,32,64)

    def test_native_kernel_keeps_f32_row_scale_outside_dot_and_no_global_w_bank(self):
        h=(ROOT/'tools/profiling/qwen_base/convrot_factorized.hpp').read_text()
        for token in ('row_scale.dtype()!=mx::float32','threadgroup T tile[BN*BK]','T(signed_code)','acc[i]*row_scale[RB+col+c[0]]'):
            self.assertIn(token,h)
        self.assertNotIn('mx::astype(words',h);self.assertNotIn('mx::contiguous(words',h)


if __name__=='__main__':unittest.main()
