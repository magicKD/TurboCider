"""CPU-only cache approximation, restoration and non-qualified speed guards."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/profiling/qwen_base'))
import runtime_cache_audit as audit


def records():
    values=[]
    for cache in ('off','conservative'):
        summary=dict(case='generate',cache=cache,route='runtime',mode='stock',requests=3,
            model_snapshot={'source':'original'},assets={'library':'exact','reference':'original'},runtime_build_id='same')
        values.append(dict(summary=summary,requests=[dict(seed=n,steps=40,output=f'{i}.png',qwen21_dit_cache=cache) for i,n in enumerate((73,74,73))],
            images=[np.full((16,16,4),i,dtype=np.uint8) for i in (1,2,1)],load=dict(passed=True),
            report=dict(runs=[dict(request_wall_including_export=n,metrics={}) for n in ((40.,30.,30.) if cache=='off' else (30.,20.,20.))]),
            coverage=[dict(saved_middle_blocks=0 if cache=='off' else 360)]*3))
    return values


class RuntimeCacheAuditTests(unittest.TestCase):
    def test_speed_requires_clean_matched_stock_three_requests(self):
        off,on=records();self.assertEqual(audit.compare(off,on)['speed']['ratio'],1.5)
        for change in (lambda x:x['load'].update(passed=False),lambda x:x['summary'].update(mode='timeline'),lambda x:x['summary'].update(requests=4)):
            x,y=records();change(x);change(y)
            self.assertIsNone(audit.compare(x,y)['speed'])
        for change in (lambda x:x['summary']['assets'].update(reference='changed'),lambda x:x['requests'][1].update(steps=8),
            lambda x:x['summary'].update(case='edit')):
            x,y=records();change(y)
            with self.assertRaises(ValueError):audit.compare(x,y)
        on['summary']['mode']='timeline'
        with self.assertRaises(ValueError):audit.compare(off,on)
        self.assertIsNone(audit.compare(off,on,True)['speed'])

    def test_provisional_budget_requires_all_rgb_and_alpha_conditions(self):
        q=dict(rgb=dict(ssim=.99,mae_255=1.),alpha_max_abs=0)
        self.assertTrue(audit.numerical_budget(q))
        for change in (dict(rgb=dict(ssim=.97,mae_255=1.)),dict(rgb=dict(ssim=.99,mae_255=3.01)),
            dict(alpha_max_abs=5),dict(rgb=dict(ssim=float('nan'),mae_255=1.))):
            self.assertFalse(audit.numerical_budget({**q,**change}))

    def test_prime_and_warm_restoration_are_not_the_same_route(self):
        q=[dict(seed=n,steps=40,output=f'{i}.png') for i,n in enumerate((73,73,74,73))]
        images=[np.full((16,16,4),i,dtype=np.uint8) for i in (2,1,3,1)]
        self.assertTrue(audit.restoration(q,images)['hot_seed_restore_rgba_exact'])
        self.assertFalse(audit.restoration(q,images)['prime_hit_quality']['rgba_exact'])
        heldout=[dict(item,seed=seed) for item,seed in zip(q,(307,307,308,307))]
        self.assertTrue(audit.restoration(heldout,images)['hot_seed_restore_rgba_exact'])
        bad=copy.deepcopy(images);bad[3][0,0,3]=9
        with self.assertRaises(ValueError):audit.restoration(q,bad)
        with self.assertRaises(ValueError):audit.restoration(q,images[:3])
        bad=copy.deepcopy(images);bad[2]=bad[1].copy()
        with self.assertRaises(ValueError):audit.restoration(q,bad)

    def test_original_busy_or_gapped_load_cannot_become_speed_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);file=root/'load.jsonl'
            data=[dict(monotonic_ns=n,competing_processes=[]) for n in (1_000_000_000,1_500_000_000,2_000_000_000)]
            def write():file.write_text(''.join(json.dumps(x)+'\n' for x in data))
            write();s=dict(load=dict(complete=True,samples=3,evidence_sha256=audit.sha256_file(file)))
            self.assertTrue(audit.load_contract(root,s)['passed'])
            bad=copy.deepcopy(s);bad['load_error']='captured sampling issue';self.assertFalse(audit.load_contract(root,bad)['passed'])
            data[1]['competing_processes']=[dict(pid=1,cpu_percent=50)]
            write();s['load']['evidence_sha256']=audit.sha256_file(file)
            self.assertFalse(audit.load_contract(root,s)['passed'])
            data[2]['monotonic_ns']=5_000_000_000;write();s['load']['evidence_sha256']=audit.sha256_file(file)
            self.assertFalse(audit.load_contract(root,s)['passed'])
            bad=copy.deepcopy(s);bad['load']['evidence_sha256']='changed'
            with self.assertRaises(ValueError):audit.load_contract(root,bad)


if __name__=='__main__':unittest.main()
