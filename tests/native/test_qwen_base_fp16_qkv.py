"""CPU-only Private FP16 QKV shapes, real-source and candidate-count gates."""
from collections import Counter
import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/profiling/qwen_base'))
import fp16_qkv_collect as screen
import fp16_qkv_audit as audit
from test_qwen_base_profiling import fixture


def report(n=16,single=False,prepack=False):
    active=screen.arms(single,prepack)
    return dict(schema='tc-private-fp16-qkv-screen-v4' if prepack else 'tc-private-fp16-qkv-screen-v3',status='complete',complete_request=False,image_export=False,frozen_real_input=True,
        repeats=n,warmups=2,single_only=single,compare_prepack=prepack,extra_owned_weight_bytes=0,source_unchanged=True,restored_outputs_verified=True,
        fp16_slot_bytes=117440512,w8_slot_bytes=65814528,w8_predictions=n+6,fp16_predictions=(len(active)-2)*(n+6),
        fp16_tiled_predictions=0 if single else n+6,fp16_single_predictions=(2 if prepack else 1)*(n+6),w8_validation_calls=1,w8_validation_relative_l2=.0035,
        fp16_program_key='' if single else 'a'*64,fp16_single_program_key='b'*64,w8_program_key='c'*64,
        samples=[dict(arm=active[(j+rep+len(active))%len(active)],rep=rep,warmup=rep<0,input_variant=bool(rep&1),seconds=.005)
                 for rep in range(-2,n) for j in range(len(active))])


def values(single=False,prepack=False):
    active=screen.arms(single,prepack);v={'input_a':np.ones((1,1024,2),dtype=np.float32),'input_b':np.full((1,1024,2),2,dtype=np.float32)}
    for i,variant in enumerate(('a','b','w')):
        for arm in active:v[arm+'_'+variant]=np.full((1,1024,2),10.+i,dtype=np.float32)
    return v


class PrivateFp16QkvTests(unittest.TestCase):
    def test_both_and_single_residency_have_exact_per_operator_calls(self):
        for single in (False,True):
            for n in (16,32):self.assertEqual(screen.validate(report(n,single),n)['predictions_per_operator'],n+6)
        for change in (dict(fp16_predictions=22),dict(w8_slot_bytes=0),dict(complete_request=True),dict(w8_validation_calls=2)):
            with self.assertRaises(ValueError):screen.validate({**report(),**change},16)

    def test_single_only_must_not_claim_loaded_tiled_program_or_calls(self):
        for change in (dict(fp16_program_key='a'*64),dict(fp16_tiled_predictions=22),dict(fp16_predictions=44)):
            with self.assertRaises(ValueError):screen.validate({**report(single=True),**change},16)
        with self.assertRaises(ValueError):screen.validate({**report(), 'fp16_single_program_key':'a'*64},16)

    def test_rotations_completed_positive_times_and_warmups_are_strict(self):
        for change in (lambda r:r['samples'].pop(),lambda r:r['samples'].reverse(),lambda r:r['samples'][0].update(warmup=False),
            lambda r:r['samples'][0].update(seconds=float('nan')),lambda r:r['samples'][0].update(input_variant=True)):
            r=report();change(r)
            with self.assertRaises(ValueError):screen.validate(r,16)

    def test_actual_model_ids_scopes_and_no_hot_reload_not_just_flags(self):
        for single in (False,True):
            active=screen.arms(single);count=len(active);raw,_=fixture();template=next(x for x in raw['events'] if x['name']=='ane.evaluate_api')
            pairs=[(1+p*count+a,a) for p in range(4) for a in range(1,count)]
            pairs += [(100+(rep+2)*count+j,(j+rep+count)%count) for rep in range(-2,16) for j in range(count) if (j+rep+count)%count]
            raw['events']=[x for x in raw['events'] if x['request']==-2]+[dict(template,request=n,model=m) for n,m in pairs]
            raw['counts']=dict(Counter(x['name'] for x in raw['events']));screen.observer_contract(raw,16,single)
            raw['events'][-1]['name']='ane.load_api';raw['counts']=dict(Counter(x['name'] for x in raw['events']))
            with self.assertRaises(ValueError):screen.observer_contract(raw,16,single)

    def test_prepack_reuses_same_single_model_and_both_fp16_route_credits(self):
        r=report(single=True,prepack=True);screen.validate(r,16)
        with self.assertRaises(ValueError):screen.validate({**r,'single_only':False},16)
        with self.assertRaises(ValueError):screen.validate({**r,'fp16_single_predictions':22},16)
        count=4;raw,_=fixture();template=next(x for x in raw['events'] if x['name']=='ane.evaluate_api')
        pairs=[(1+p*count+a,1 if a==1 else 2) for p in range(4) for a in range(1,count)]
        pairs += [(100+(rep+2)*count+j,1 if (j+rep+count)%count==1 else 2) for rep in range(-2,16) for j in range(count) if (j+rep+count)%count]
        raw['events']=[x for x in raw['events'] if x['request']==-2]+[dict(template,request=n,model=m) for n,m in pairs]
        raw['counts']=dict(Counter(x['name'] for x in raw['events']));self.assertEqual(screen.observer_contract(raw,16,True,True)['runtime_models'],2)
        raw['events'][-1]['model']=3
        with self.assertRaises(ValueError):screen.observer_contract(raw,16,True,True)
        v=values(True,True);self.assertTrue(audit.compare(v,screen.arms(True,True))['fp16_single_prepack_bits_exact'])
        v['fp16-prepack_a'][0,512,0]+=.00001
        with self.assertRaises(ValueError):audit.compare(v,screen.arms(True,True))

    def test_full_gpu_head_and_ane_tail_must_both_be_valid(self):
        audit.compare(values(),screen.arms(False))
        for change in (lambda v:v['fp16-single_a'].__setitem__((0,0,0),10.001),
            lambda v:v['fp16-single_a'].__setitem__((slice(None),slice(512,None),slice(None)),0),
            lambda v:v.update(input_b=v['input_a'].copy()),lambda v:v.pop('fp16-tiled_w')):
            v=values();change(v)
            with self.assertRaises(ValueError):audit.compare(v,screen.arms(False))

    def test_perturbation_keeps_two_typed_bf16_rounding_boundaries(self):
        def bfloat(value):
            bits=struct.unpack('<I',struct.pack('<f',float(value)))[0];bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000
            return struct.unpack('<f',struct.pack('<I',bits))[0]
        a=np.array([0.,1.,-.5,.00390625,31.5],dtype=np.float32)
        expected=np.array([bfloat(bfloat(x*.984375)+.00390625) for x in a],dtype=np.float32)
        np.testing.assert_array_equal(audit.perturbed_input(a),expected)

    def test_single_mil_has_one_dynamic_matmul_not_twentyfour_slices(self):
        h=(ROOT/'tools/profiling/qwen_base/fp16_qkv.hpp').read_text();body=h.split('inline std::string single_program()',1)[1].split('class Graph',1)[0]
        self.assertEqual(body.count('yn = matmul('),1);self.assertNotIn('slice_by_size',body);self.assertNotIn('dequantize(',body)
        self.assertIn('tensor_buffer<fp16',body);self.assertIn('if(!single_only)',h)


if __name__=='__main__':unittest.main()
