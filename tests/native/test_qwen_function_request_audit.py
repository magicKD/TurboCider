"""Host-only complete-request comparison/denominator safety, no model/ANE."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/profiling/qwen_base'))
import function_request_audit as audit


class FunctionRequestAuditTests(unittest.TestCase):
    def test_stock_actual_counters_and_timeline_progress_have_distinct_proof(self):
        r=dict(runs=[dict(metrics=dict(actual_denoise_steps=40,hybrid=dict(runtime_weight=dict(hybrid_blocks_session_total=1280))))],progress=[])
        self.assertIn('native_actual_steps',audit.sampler_evidence(r,0,'stock'))
        with self.assertRaises(ValueError):audit.sampler_evidence(r,0,'timeline')
        for i in range(41):r['progress'].append(dict(request=0,event=dict(phase='denoise',completed=i,total=40)))
        self.assertIn('complete_progress',audit.sampler_evidence(r,0,'timeline'))
        r['progress'].pop()
        with self.assertRaises(ValueError):audit.sampler_evidence(r,0,'stock')
        r['progress']=[];r['runs'][0]['metrics']['actual_denoise_steps']=39
        with self.assertRaises(ValueError):audit.sampler_evidence(r,0,'stock')

    def test_no_timeline_single_or_contaminated_speed_ratio(self):
        control=dict(summary=dict(mode='stock',requests=3,load=dict(complete=True)),
                     report=dict(runs=[dict(request_wall_including_export=t) for t in (10.,8.,8.)]))
        candidate=copy.deepcopy(control)
        candidate['report']['runs'][1:]=[dict(request_wall_including_export=4.)]*2
        self.assertEqual(audit.speed(control,candidate)['ratio'],2.)
        for key,value in (('mode','timeline'),('requests',1),('load_error','external CPU load'),('load',dict(complete=False))):
            bad=copy.deepcopy(candidate);bad['summary'][key]=value
            self.assertIsNone(audit.speed(control,bad))

    def test_normalization_retains_all_model_sampling_and_reference_work(self):
        q=dict(model='qwen-image-2.1',steps=40,seed=73,prompt='example',width=512,height=512,
               inputs=[dict(kind='image',role='reference',path='original.png')],output='a.png',ane_manifest='a.json')
        other=dict(q,output='b.png',ane_manifest='b.json')
        self.assertEqual(audit.normalized_request(q),audit.normalized_request(other))
        for change in (dict(steps=8),dict(seed=74),dict(inputs=[]),dict(width=768),dict(prompt='changed')):
            self.assertNotEqual(audit.normalized_request(q),audit.normalized_request(dict(other,**change)))

    def test_gpu_step0_requires_restoration_but_legacy_records_approximation(self):
        requests=[dict(seed=seed,output=f'{i}.png') for i,seed in enumerate((73,74,73))]
        first=np.arange(16*16*4,dtype=np.uint8).reshape(16,16,4)
        second=first.copy();second[1,1,0]^=127
        changed=first.copy();changed[2,2,1]^=1
        self.assertTrue(audit.image_contract(requests,[first,second,first],'gpu')['rgba_exact'])
        self.assertFalse(audit.image_contract(requests,[first,second,changed])['rgba_exact'])
        with self.assertRaisesRegex(ValueError,'restoration'):
            audit.image_contract(requests,[first,second,changed],'gpu')
        with self.assertRaisesRegex(ValueError,'stale'):
            audit.image_contract(requests,[first,first,first],'gpu')
        with self.assertRaises(ValueError):audit.image_contract(requests,[first])
        self.assertIsNone(audit.image_contract(requests[:1],[first],'gpu'))

    def test_replay_only_ignores_observer_toggle_and_export_paths(self):
        requests=[dict(seed=73,width=512,output='a.png',ane_manifest='a.json')]
        summary=dict(runtime_build_id='exact-build',model_snapshot=dict(weight='exact-source'),
                     environment=dict(algorithm='fixed',TURBOCIDER_QWEN21_PROFILE_STEPS='1'))
        stock=copy.deepcopy(summary);stock['environment'].pop('TURBOCIDER_QWEN21_PROFILE_STEPS')
        other=[dict(requests[0],output='b.png',ane_manifest='b.json')]
        self.assertEqual(audit.replay_key(summary,requests),audit.replay_key(stock,other))
        for value in (dict(stock,runtime_build_id='other'),dict(stock,model_snapshot=dict(weight='other')),
                      dict(stock,environment=dict(algorithm='different'))):
            self.assertNotEqual(audit.replay_key(summary,requests),audit.replay_key(value,requests))
        self.assertNotEqual(audit.replay_key(stock,requests),audit.replay_key(stock,[dict(requests[0],seed=74)]))
        original=np.zeros((16,16,4),dtype=np.uint8)
        self.assertEqual(audit.replay_contract([original],[original.copy()]),dict(images=1,rgba_exact=True))
        with self.assertRaises(ValueError):audit.replay_contract([original],[original+1])
        with self.assertRaises(ValueError):audit.replay_contract([original],[])


if __name__=='__main__':unittest.main()
