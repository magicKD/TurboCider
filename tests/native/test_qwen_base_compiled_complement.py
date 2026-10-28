"""CPU-only contracts for fixed original/compiled dynamic-W GPU complement."""
from collections import Counter
import copy
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/profiling/qwen_base'))
import compiled_complement_collect as screen
import compiled_complement_hybrid_collect as hybrid
import compiled_complement_audit as audit
from test_qwen_base_dual_gup import real_report
from test_qwen_base_profiling import fixture


def report(n=16):
    c=dict(partial=dict(relative_l2=0.,max_abs=0.,elements=4194304),hidden=dict(relative_l2=0.,max_abs=0.,elements=6291456),
        partial_bits_exact=True,hidden_bits_exact=True)
    return dict(schema='tc-compiled-complement-v1',status='complete',complete_request=False,image_export=False,ane_predictions=0,extra_owned_weight_bytes=0,
        repeats=n,warmups=2,source_unchanged=True,explicit_body_native_bits_exact=True,rejected=7,m=1024,h=4096,f=12288,channels=6144,
        trace_count_after_sources=1,trace_count_final=1,checks={k:copy.deepcopy(c) for k in ('a','b','w')},
        samples=[dict(arm=screen.ARMS[(j+rep+2)%2],rep=rep,warmup=rep<0,input_variant=bool(rep&1),layer=int((rep+2)%4==2),seconds=.01)
            for rep in range(-2,n) for j in range(2)])


def hybrid_report(n=16):
    r=report(n);r.update(schema='tc-compiled-complement-hybrid-v1',complete_dit_block=False,metrics=real_report(n)['metrics'],
        checks={v:dict(bits_exact=True,incremental=dict(relative_l2=0.,max_abs=0.,elements=4194304),
            gpu_quality=dict(relative_l2=.007,max_abs=.03,elements=4194304)) for v in ('a','b','w')})
    for s in r['samples']:s['completion_deltas']=dict(prediction_wait_seconds=.005,output_wait_seconds=.0001,caller_wait_seconds=.004)
    return r


class CompiledComplementTests(unittest.TestCase):
    def test_fixed_geometry_explicit_weights_and_one_graph(self):
        for n in (16,32):self.assertEqual(screen.validate(report(n),n)['measured_complements'],2*n)
        for change in (dict(channels=5632),dict(ane_predictions=1),dict(extra_owned_weight_bytes=1),dict(complete_request=True),
            dict(trace_count_after_sources=2),dict(trace_count_final=2),dict(rejected=0)):
            with self.assertRaises(ValueError):screen.validate({**report(),**change},16)

    def test_full_array_bits_and_both_input_weights_controls_required(self):
        for change in (lambda r:r['checks'].pop('w'),lambda r:r['checks']['b'].update(hidden_bits_exact=False),
            lambda r:r['checks']['w']['partial'].update(relative_l2=1e-9),lambda r:r['checks']['a']['hidden'].update(elements=1)):
            r=report();change(r)
            with self.assertRaises(ValueError):screen.validate(r,16)

    def test_rotated_schedule_actual_weights_and_complete_times(self):
        for change in (lambda r:r['samples'].reverse(),lambda r:r['samples'].pop(),lambda r:r['samples'][0].update(layer=1),
            lambda r:r['samples'][0].update(warmup=False),lambda r:r['samples'][0].update(seconds=float('nan'))):
            r=report();change(r)
            with self.assertRaises(ValueError):screen.validate(r,16)

    def test_gpu_only_observer_requires_every_completion(self):
        raw,_=fixture();template=next(x for x in raw['events'] if x['request']==-2);r=report()
        raw['events']=[template]+[dict(template,request=100+i) for i in range(len(r['samples']))];raw['counts']=dict(Counter(x['name'] for x in raw['events']))
        screen.observer_contract(raw,r);raw['events'].pop();raw['counts']=dict(Counter(x['name'] for x in raw['events']))
        with self.assertRaises(ValueError):screen.observer_contract(raw,r)

    def test_body_uses_original_buffers_and_existing_typed_arithmetic(self):
        h=(ROOT/'tools/profiling/qwen_base/compiled_complement.hpp').read_text();self.assertIn('silu(g)*u',h);self.assertIn('dense_gpu::projection_range(hidden,a[2]',h)
        self.assertNotIn('mx::astype',h);self.assertNotIn('mx::contiguous',h)
        p=(ROOT/'tools/profiling/qwen_base/compiled_complement_probe.mm').read_text();self.assertIn('mx::compile([traces]',p)
        self.assertIn('weights[layer]->project_slice',p);self.assertIn('explicit body changed actual native callback arithmetic',p)

    def test_hybrid_exactness_workload_and_scope_cannot_be_dropped(self):
        for n in (16,32):self.assertEqual(hybrid.validate(hybrid_report(n),n)['native_workload_predictions'],2*n+12)
        for change in (lambda r:r.update(complete_dit_block=True),lambda r:r['checks']['w'].update(bits_exact=False),
            lambda r:r['checks']['a']['gpu_quality'].update(relative_l2=.03),lambda r:r['checks']['a']['incremental'].update(elements=1),
            lambda r:r['metrics'].update(runtime_calls_session_total=1),lambda r:r['samples'][0].pop('completion_deltas')):
            r=hybrid_report();change(r)
            with self.assertRaises(ValueError):hybrid.validate(r,16)

    def test_hybrid_selector_every_actual_prediction_one_model_no_hot_compile(self):
        raw,_=fixture();template=next(x for x in raw['events'] if x['name']=='ane.evaluate_api');r=hybrid_report()
        scopes=set(range(1,7))|{10,11}|set(range(100,100+len(r['samples'])))
        raw['events']=[x for x in raw['events'] if x['request']==-2]+[dict(template,request=n,model=1) for n in sorted(scopes)];raw['counts']=dict(Counter(x['name'] for x in raw['events']))
        self.assertEqual(hybrid.observer_contract(raw,r)['actual_workload_evaluations'],44)
        bad=copy.deepcopy(raw);bad['events'][-1]['model']=2
        with self.assertRaises(ValueError):hybrid.observer_contract(bad,r)
        bad=copy.deepcopy(raw);bad['events'][-1]['name']='ane.compile_api';bad['counts']=dict(Counter(x['name'] for x in bad['events']))
        with self.assertRaises(ValueError):hybrid.observer_contract(bad,r)

    def test_independent_component_full_bits_audit_rejects_one_value(self):
        r=report();v={f'input_{t}':np.array([i+1.],dtype='<f4') for i,t in enumerate(('a','b'))}
        for i,t in enumerate(('a','b','w')):
            for arm in ('raw','compiled'):
                for part in ('partial','hidden'):v[arm+'_'+part+'_'+t]=np.array([i+1.],dtype='<f4')
        self.assertTrue(audit.check_arrays(v,r,False)['all_incremental_output_bits_exact'])
        bad=copy.deepcopy(v);bad['compiled_partial_w'][0]+=1e-9 # below F32 precision: unchanged
        audit.check_arrays(bad,r,False);bad['compiled_partial_w'][0]+=1e-5
        with self.assertRaises(ValueError):audit.check_arrays(bad,r,False)

    def test_hybrid_uses_real_native_owner_and_waits_not_gpu_placeholder(self):
        p=(ROOT/'tools/profiling/qwen_base/compiled_complement_hybrid_probe.mm').read_text()
        for token in ('ane::HybridFfn runtime','runtime.stage_weights','runtime.run','source_scope.finish()',
            'runtime_weight_overflow_retries==0','runtime_weight_program_instances==1','runtime_weight_fallback_blocks==0'):
            self.assertIn(token,p)


if __name__=='__main__':unittest.main()
