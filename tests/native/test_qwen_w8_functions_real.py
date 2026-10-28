"""Host-only real-consumer report coverage and timing-denominator negatives."""
import copy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/profiling/qwen_base'))
import w8_functions_real_collect as screen


def fixture(reverse=False):
    r=dict(schema='tc-real-w8-function-ffn-v1',status='complete',whole_ffn=True,whole_dit_block=False,image_export=False,
           h=4096,f=12288,ane_channels=6144,repeats=8,warmups=2,source_unchanged=True,restored_bits_exact=True,
           reverse=reverse,checks=[],samples=[])
    for n in screen.ROWS:
        for state in range(3 if n==1024 else 1):
            q=dict(elements=n*4096,relative_l2=.001,max_abs=.1)
            r['checks'].append(dict(rows=n,state=state,qualities=[q.copy() for _ in screen.ARMS],family_vs_single2048=q.copy()))
        for rep in range(-2,8):
            for j in range(4):
                arm=screen.ARMS[3-(j+rep+4)%4 if reverse else (j+rep+4)%4]
                r['samples'].append(dict(rows=n,arm=arm,rep=rep,warmup=rep<0,input_variant=bool(rep&1),layer=(rep+2)%4==2,
                                         calls=screen.predictions(arm,n),seconds=.02,scope=100+len(r['samples'])))
    return r


class RealFunctionConsumerTests(unittest.TestCase):
    def test_complete_forward_reverse_schedule_and_actual_calls(self):
        for reverse in (False,True):self.assertEqual(screen.validate(fixture(reverse),8)['measured_consumers'],128)
        self.assertEqual([screen.predictions(a,2560) for a in screen.ARMS],[0,3,2,2])

    def test_missing_invalid_wrong_call_or_scope_never_counts(self):
        for change in (lambda r:r['samples'].pop(),lambda r:r['samples'][0].update(scope=99),
                       lambda r:r['samples'][0].update(calls=99),lambda r:r['samples'][0].update(seconds=float('nan')),
                       lambda r:r['samples'][0].update(warmup=False),lambda r:r['checks'].pop(),
                       lambda r:r['checks'][0]['qualities'][1].update(relative_l2=.1),
                       lambda r:r['checks'][0]['family_vs_single2048'].update(relative_l2=.02),
                       lambda r:r.update(whole_dit_block=True),lambda r:r.update(image_export=True)):
            r=copy.deepcopy(fixture());change(r)
            with self.assertRaises(ValueError):screen.validate(r,8)


if __name__=='__main__':unittest.main()
