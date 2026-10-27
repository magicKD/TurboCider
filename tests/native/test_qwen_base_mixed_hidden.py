"""CPU-only mixed hidden dtype/geometry/domain/oracle negative controls."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools/profiling/qwen_base'))
import mixed_hidden_collect as screen
import mixed_hidden_audit as audit
from test_qwen_base_profiling import fixture


def report(real=False,n=16):
    m,k,out,pitch,rb,cb=(1024,6144,4096,12288,0,0) if real else (33,128,257,320,5,17)
    q=dict(relative_l2=0.,max_abs=0.)
    return dict(schema='tc-mixed-hidden-down-v1',status='complete',real_hidden=real,complete_request=False,image_export=False,ane_predictions=0,
        extra_owned_weight_bytes=0,repeats=n,warmups=2,source_unchanged=True,rejected=6,m=m,k=k,n=out,physical_weight_pitch=pitch,row_begin=rb,column_begin=cb,
        checks={v:{key:{**q,'elements':m*(k if key=='cast_quality' else out)} for key in ('cast_quality','incremental_partial_quality','same_value_partial_quality')} for v in ('a','b')},
        wide_format_rejected=True,format_domains=dict(underflow=dict(finite=True,vs_original=dict(relative_l2=1.,max_abs=1e-8)),subnormal=dict(finite=True),
            zero=dict(finite=True,vs_original=q.copy()),wide=dict(finite=False,rejected_before_down=True)),
        samples=[dict(arm=screen.ARMS[(j+rep+4)%4],rep=rep,warmup=rep<0,input_variant=bool(rep&1),seconds=.0037) for rep in range(-2,n) for j in range(4)])


def synthetic():
    r=report();m,k,n,pitch,rb,cb=(r[x] for x in ('m','k','n','physical_weight_pitch','row_begin','column_begin'))
    base=np.arange((m+2)*k,dtype=np.int64).reshape(m+2,k)[1:m+1]%29-14
    wi=np.arange((n+rb+4)*pitch,dtype=np.int64).reshape(n+rb+4,pitch);w=(((wi%257)*37+13)%31-15)[2:n+rb+2]
    wf=w.astype('<f4')/128;r['source_weight_sha256']=hashlib.sha256((wf.view('<u4')>>16).astype('<u2').tobytes()).hexdigest()
    values={}
    for tag,ints in (('a',base*4),('b',base*3+2)):
        x=ints.astype('<f4')[None]/128;p=(ints@w[rb:rb+n,cb:cb+k].T).astype('<f4')[None]/16384
        for name in ('input','half','round'):values[name+'_'+tag]=x.copy()
        for arm in screen.ARMS:values[arm+'_'+tag]=p.copy()
    for domain,factor in (('underflow',2**-24),('subnormal',2**-16),('zero',0.)):
        values[domain+'_input']=values['input_a']*np.float32(factor);values[domain+'_half']=values[domain+'_input'].astype('<f2').astype('<f4')
        values[domain+'_original']=values[screen.ARMS[0]+'_a']*np.float32(factor)
        wanted=np.zeros_like(values[domain+'_original']) if domain!='subnormal' else values[domain+'_original']
        values[domain+'_mixed']=wanted.copy();values[domain+'_rounded']=wanted.copy()
    return values,r


class MixedHiddenTests(unittest.TestCase):
    def test_physical_ranges_and_all_four_rotated_controls(self):
        for real in (False,True):
            for n in (16,32):screen.validate(report(real,n),n,real)
        for change in (dict(physical_weight_pitch=6144),dict(column_begin=1),dict(rejected=0),dict(ane_predictions=1),dict(complete_request=True)):
            with self.assertRaises(ValueError):screen.validate({**report(True),**change},16,True)

    def test_expected_underflow_loss_not_zero_exact_and_wide_rejection(self):
        for change in (lambda r:r['format_domains']['underflow']['vs_original'].update(relative_l2=0),
            lambda r:r['format_domains'].pop('subnormal'),lambda r:r.update(wide_format_rejected=False),
            lambda r:r['format_domains']['zero']['vs_original'].update(max_abs=1e-9)):
            r=report();change(r)
            with self.assertRaises(ValueError):screen.validate(r,16,False)

    def test_normal_numeric_gate_and_warmup_order_are_not_optional(self):
        for change in (lambda r:r['samples'].reverse(),lambda r:r['samples'].pop(),lambda r:r['samples'][0].update(warmup=False),
            lambda r:r['checks']['a']['incremental_partial_quality'].update(relative_l2=1e-4),lambda r:r['samples'][0].update(seconds=float('nan'))):
            r=report(True);change(r)
            with self.assertRaises(ValueError):screen.validate(r,16,True)

    def test_reader_keeps_f16_subnormal_signedzero_and_strict_dtype(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.safetensors';arrays=dict(out=np.array([1.,2.],dtype='<f4'),bf=np.array([0x3f80,0xbf80,0],dtype='<u2'),half=np.array([.5,2**-20,-0.],dtype='<f2'))
            h={};payload=b''
            for name,a in arrays.items():h[name]=dict(dtype='F32' if name=='out' else 'BF16' if name=='bf' else 'F16',shape=[1,1,a.size],data_offsets=[len(payload),len(payload)+a.nbytes]);payload+=a.tobytes()
            encoded=json.dumps(h).encode();encoded+=b' '*((-len(encoded))%8);p.write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)
            specs={name:(entry['dtype'],tuple(entry['shape'])) for name,entry in h.items()};v=audit.read(p,specs)
            np.testing.assert_array_equal(v['half'].view('<u4'),arrays['half'].astype('<f4').reshape(1,1,3).view('<u4'))
            bad=specs.copy();bad['half']=('BF16',(1,1,3))
            with self.assertRaises(ValueError):audit.read(p,bad)

    def test_full_integer_oracle_covers_source_offsets_tails_and_tiny_domains(self):
        v,r=synthetic();q=audit.synthetic_oracle(v,r);self.assertTrue(q['all_exact_dyadic_integer_oracle']);self.assertEqual(q['underflow_relative_l2'],1)
        bad=copy.deepcopy(v);bad['mixed-ready_a'][0,-1,-1]+=1
        with self.assertRaises(ValueError):audit.synthetic_oracle(bad,r)
        bad=copy.deepcopy(v);bad['subnormal_mixed'][0,0,0]+=1e-6
        with self.assertRaises(ValueError):audit.synthetic_oracle(bad,r)

    def test_gpu_only_observer_requires_completed_work_and_no_ane(self):
        raw,_=fixture();template=next(x for x in raw['events'] if x['request']==-2);r=report()
        raw['events']=[template]+[dict(template,request=100+i) for i in range(len(r['samples']))];raw['counts']=dict(Counter(x['name'] for x in raw['events']))
        screen.observer_contract(raw,r);raw['events'][-1]['name']='ane.evaluate_api';raw['counts']=dict(Counter(x['name'] for x in raw['events']))
        with self.assertRaises(ValueError):screen.observer_contract(raw,r)

    def test_mixed_source_keeps_separate_tensor_types_and_original_weight_pitch(self):
        h=(ROOT/'tools/profiling/qwen_base/mixed_hidden.hpp').read_text();self.assertIn('device A*',h);self.assertIn('device B*',h)
        self.assertNotIn('auto aa=a.slice(0,row),bb=',h);self.assertIn('RB*WP+CB',h);self.assertIn('execution_simdgroups<4>',h)


if __name__=='__main__':unittest.main()
