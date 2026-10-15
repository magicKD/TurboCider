import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('real_ffn_screen',ROOT/'tools/validation/qwen21_real_ffn_screen.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class RealFfnScreenTest(unittest.TestCase):
    def fixture(self):
        header=dict(samples=9,warmup_passes=1,measured_passes=2,calibration_step=1,lora=True,width=4096,recipe='hadamard',setup_seconds=1)
        points=[]
        for run in range(3):
            for step in (1,3,5):
                for layer in (0,15,31):
                    p=dict(pass_=run,step=step,layer=layer,held_out=step!=1,input_sha256='a'*64,
                           gpu_seconds=.03,hybrid_seconds=.02,relative_rmse=.01,cosine=.9999,
                           calls=1,headroom=1,fallback_blocks=0,overflow_retries=0,scale_cache_hits=0,scale_cache_misses=0)
                    p['pass']=p.pop('pass_')
                    for name in ('stage','stage_wait','pre','lora_input_ready','a8_wait','async_ane_wait','post_join','wall'):
                        p[name+'_seconds']=.001
                    points.append(p)
        return [header]+points
    def result(self,rows):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'sample.jsonl';p.write_text('\n'.join(map(json.dumps,rows)))
            return m.summarize(p)
    def test_complete_path_gate(self):self.assertTrue(self.result(self.fixture())['eligible_for_next_e2e'])
    def test_cold_retry_cannot_be_hidden(self):
        r=self.fixture();r[1]['overflow_retries']=1
        self.assertFalse(self.result(r)['eligible_for_next_e2e'])
    def test_holdout_error_rejects(self):
        r=self.fixture();r[-1]['relative_rmse']=.04
        self.assertFalse(self.result(r)['eligible_for_next_e2e'])
    def test_slow_depth_rejects(self):
        r=self.fixture()
        for p in r[1:]:
            if p['layer']==31:p['hybrid_seconds']=.04
        self.assertFalse(self.result(r)['eligible_for_next_e2e'])
    def test_missing_duplicate_changed_nonfinite_reject(self):
        original=self.fixture()
        for mode in ('missing','duplicate','changed','nan'):
            with self.subTest(mode=mode):
                r=copy.deepcopy(original)
                if mode=='missing':r.pop()
                if mode=='duplicate':r[-1]=r[-2]
                if mode=='changed':r[-1]['input_sha256']='b'*64
                if mode=='nan':r[-1]['hybrid_seconds']=float('nan')
                with self.assertRaises(ValueError):self.result(r)
