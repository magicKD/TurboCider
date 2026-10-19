"""CPU-only W producer accounting and independent H128/RNE oracle gates."""
from collections import Counter
import copy
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/profiling/qwen_base'))
import qkv_stage_collect as screen
import qkv_stage_audit as audit
import qkv_stage_consumer_collect as consumer
import qkv_stage_consumer_audit as caudit
from test_qwen_base_profiling import fixture


def stage_report(n=32):
    r=dict(schema='tc-qkv-w-stage-v1',status='complete',complete_request=False,image_export=False,ane_predictions=0,real_source_layers=[0,1],
        repeats=n,warmups=2,extra_owned_weight_bytes=0,scale_cache_capacity=2,metadata_bytes=49152,checksum_policy='preflight-and-post-loop-only',
        source_unchanged=True,all_stage_bytes_exact=True,calls=4*n+23,native_calls=n+7,custom_calls=3*n+16,command_buffers=6*n+37,
        compute_dispatches=16*n+99,native_scale_misses=6,native_scale_hits=3*(n+5),rejected=5,setup_seconds=.1,prime_seconds=.01)
    r['arrays']=[dict(layer=l,arm=a) for l in (0,1) for a in screen.ARMS]
    r['samples']=[dict(arm=screen.ARMS[(j+rep+4)%4],rep=rep,warmup=rep<0,layer=int(bool(rep&1)),seconds=.001,
        command_buffers=3 if (j+rep+4)%4==0 else 1,compute_dispatches=(6,6,3,1)[(j+rep+4)%4]) for rep in range(-2,n) for j in range(4)]
    return r


def consumer_report(n=32):
    r=dict(schema='tc-qkv-stage-consumer-v1',status='complete',complete_request=False,image_export=False,synthetic_inputs=True,repeats=n,warmups=2,
        extra_owned_weight_bytes=0,source_unchanged=True,incremental_bits_exact=True,program_key='a'*64,predictions=4*n+24,stage_calls=4*n+26,
        stage_command_buffers=6*n+42,stage_dispatches=16*n+114)
    r['samples']=[dict(arm=screen.ARMS[(j+rep+4)%4],rep=rep,warmup=rep<0,input_variant=bool(rep&1),seconds=.005,
        weight_stage_seconds=.001,activation_seconds=.0002,predict_restore_join_seconds=.0036) for rep in range(-2,n) for j in range(4)]
    return r


class QkvStageTests(unittest.TestCase):
    def test_complete_cold_metadata_work_and_measured_cb_dispatch_credits(self):
        for n in (32,64):screen.validate(stage_report(n),n)
        for change in (dict(compute_dispatches=1),dict(native_scale_misses=0),dict(rejected=4),dict(ane_predictions=1),dict(complete_request=True)):
            with self.assertRaises(ValueError):screen.validate({**stage_report(),**change},32)

    def test_all_rotated_sources_warmup_and_positive_completed_times(self):
        for change in (lambda r:r['samples'].pop(),lambda r:r['samples'].reverse(),lambda r:r['arrays'].pop(),
            lambda r:r['samples'][0].update(layer=1),lambda r:r['samples'][0].update(seconds=float('nan')),
            lambda r:r['samples'][0].update(compute_dispatches=1)):
            r=stage_report();change(r)
            with self.assertRaises(ValueError):screen.validate(r,32)

    def test_cpu_h128_scale_and_codes_include_zero_and_nonzero_rows(self):
        bits=np.zeros((2,128),dtype='<u2');bits[0,0]=0x3f80
        scale,code,q=audit.quantize(bits,audit.signs());self.assertTrue(np.all(np.abs(code[0].astype(int))==127));self.assertTrue(np.all(code[1]==0))
        self.assertEqual(scale[1],np.float16(128).view(np.uint16));self.assertTrue(audit.oracle_check(bits,scale,code,audit.signs())['code_bits_exact'])
        bad=code.copy();bad[1,0]=1
        with self.assertRaises(ValueError):audit.oracle_check(bits,scale,bad,audit.signs())

    def test_complete_producer_bits_not_only_sizes_or_finite(self):
        values={(l,a):(np.full((2,128),l,dtype='int8'),np.full(2,1,dtype='<u2')) for l in (0,1) for a in screen.ARMS};audit.compare(values)
        values[0,screen.ARMS[3]][0][0,0]=1
        with self.assertRaises(ValueError):audit.compare(values)

    def test_full_consumer_keeps_one_model_actual_predictions_and_primary_wall(self):
        for n in (32,64):self.assertEqual(consumer.validate(consumer_report(n),n)['workload_predictions'],4*n+24)
        for change in (dict(predictions=128),dict(stage_calls=128),dict(synthetic_inputs=False),dict(complete_request=True)):
            with self.assertRaises(ValueError):consumer.validate({**consumer_report(),**change},32)
        raw,_=fixture();template=next(x for x in raw['events'] if x['name']=='ane.evaluate_api');raw['events']=[x for x in raw['events'] if x['request']==-2]
        raw['events'] += [dict(template,request=i,model=1) for i in list(range(1,17))+list(range(100,100+4*34))]
        raw['counts']=dict(Counter(x['name'] for x in raw['events']));consumer.observer_contract(raw,32)
        raw['events'][-1]['model']=2
        with self.assertRaises(ValueError):consumer.observer_contract(raw,32)

    def test_w_only_observer_cannot_hide_ane_prediction_and_requires_actual_cbs(self):
        raw,_=fixture();template=next(x for x in raw['events'] if x['name']=='metal.command_buffer' and x['request']==-2)
        raw['events']=[template]
        counts={100+(rep+2)*4+j:(3 if (j+rep+4)%4==0 else 1) for rep in range(-2,32) for j in range(4)}
        counts.update({1+p*4+a:(3 if a==0 else 1) for p in range(3) for a in range(4)})
        raw['events'] += [dict(template,request=n) for n,c in counts.items() for _ in range(c)];raw['counts']=dict(Counter(x['name'] for x in raw['events']))
        screen.observer_contract(raw,32);raw['events'].pop();raw['counts']=dict(Counter(x['name'] for x in raw['events']))
        with self.assertRaises(ValueError):screen.observer_contract(raw,32)

    def test_complete_consumer_bytes_cannot_be_replaced_by_close_quality(self):
        values={'input_a':np.array([0.,1.],dtype=np.float32),'input_b':np.array([2.,3.],dtype=np.float32)}
        for i,v in enumerate(('a','b','w')):
            values['gpu_'+v]=np.array([i+1.,i+2.],dtype=np.float32)
            for a in screen.ARMS:values[a+'_'+v]=values['gpu_'+v].copy()
        caudit.compare(values)
        bad=copy.deepcopy(values);bad[screen.ARMS[-1]+'_a'][0]+=.00001
        with self.assertRaises(ValueError):caudit.compare(bad)
        bad=copy.deepcopy(values);bad['input_b']=bad['input_a'].copy()
        with self.assertRaises(ValueError):caudit.compare(bad)


if __name__=='__main__':unittest.main()
