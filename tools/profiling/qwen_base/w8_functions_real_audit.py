#!/usr/bin/env python3
"""Post-window independent real-FFN source/full-array/join/down-witness audit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import tarfile
import numpy as np

import w8_functions_real_collect as screen
from collect import ROOT, model_snapshot, sha256_file, write_json
from expanded_audit import archive_binding, PINS
from gpu_dual_request_audit import memory_contract


class Tensors:
    def __init__(self,path,spec=None):
        self.path=Path(path)
        with self.path.open('rb') as f:
            self.n=struct.unpack('<Q',f.read(8))[0]
            if not 2<=self.n<=16<<20:raise ValueError('bounded complete tensor header required')
            self.header={k:v for k,v in json.loads(f.read(self.n)).items() if k!='__metadata__'}
        if spec is not None and set(spec)!=set(self.header):raise ValueError('exact complete tensor names required')
        data_bytes=self.path.stat().st_size-8-self.n;extents=[];self.values={}
        for name,info in self.header.items():
            if spec is not None and (info['dtype'],info['shape'])!=(spec[name][0],list(spec[name][1])):
                raise ValueError('original BF16/F32 tensor geometry differs')
            begin,end=info['data_offsets'];item={'BF16':2,'F32':4}.get(info['dtype'])
            # The original model may contain other metadata tensors; only the
            # output fixture requires a complete all-BF16/F32 tensor domain.
            if item is None:
                if spec is not None:raise ValueError('unsupported output tensor dtype')
                continue
            if (any(type(v) is not int or v<=0 for v in info['shape']) or not 0<=begin<=end<=data_bytes or
                begin%item or end-begin!=math.prod(info['shape'])*item):raise ValueError('complete typed tensor extent required')
            self.values[name]=np.memmap(self.path,mode='r',dtype='<u2' if item==2 else '<f4',offset=8+self.n+begin,shape=tuple(info['shape']))
            extents.append((begin,end))
        if spec is not None:
            cursor=0
            for begin,end in sorted(extents):
                if begin!=cursor:raise ValueError('overlapping or omitted tensor bytes')
                cursor=end
            if cursor!=data_bytes:raise ValueError('extra output bytes')

    def floats(self,name,sl=slice(None)):
        value=self.values[name][sl]
        return (value.astype(np.uint32)<<16).view(np.float32) if self.header[name]['dtype']=='BF16' else np.asarray(value)

    def digest(self,name):
        raw=self.values[name].reshape(-1).view(np.uint8);h=hashlib.sha256()
        for start in range(0,raw.size,4<<20):h.update(raw[start:start+(4<<20)])
        return h.hexdigest()


def specifications():
    result={}
    for n in screen.ROWS:
        for state in range(3 if n==1024 else 1):
            tag=f'{n}_{state}'
            for name in ('input',)+screen.ARMS:result[name+'_'+tag]=('BF16',(1,n,4096))
            if n==1024:
                for arm in screen.ARMS[1:]:result[arm+'_tail_'+tag]=('F32',(1,n,4096))
                result['head_'+tag]=('F32',(1,n,4096));result['hidden_'+tag]=('BF16',(1,n,6144))
    return result


def quality(t,left,right):
    if t.values[left].shape!=t.values[right].shape:raise ValueError('same full shape required')
    energy=error=maximum=0.;changed=0;n=t.values[left].shape[1]
    for first in range(0,n,64):
        sl=(slice(None),slice(first,min(n,first+64)),slice(None))
        x=t.floats(left,sl).astype(np.float64);y=t.floats(right,sl).astype(np.float64)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):raise ValueError('complete finite outputs required')
        d=x-y;energy+=float(np.sum(x*x));error+=float(np.sum(d*d));maximum=max(maximum,float(np.max(np.abs(d))));changed+=int(np.count_nonzero(d))
    value=math.sqrt(error/max(energy,1e-30))
    if not math.isfinite(value):raise ValueError('finite independent quality required')
    return dict(relative_l2=value,max_abs=maximum,changed=changed,elements=t.values[left].size)


def bf16_rne(value):
    v=np.asarray(value,dtype='<f4')
    if not np.all(np.isfinite(v)):raise ValueError('finite FP32 join required')
    bits=v.view('<u4')
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype('<u2')


def join_and_down(t,model):
    witnesses=[]
    coordinates=[(0,0),(1023,4095),(0,4095),(1023,0)]+[((i*137+11)%1024,(i*977+31)%4096) for i in range(28)]
    for state in range(3):
        tag=f'1024_{state}';layer=int(state==2)
        for arm in screen.ARMS[1:]:
            for first in range(0,1024,64):
                sl=(slice(None),slice(first,first+64),slice(None))
                h=t.floats('head_'+tag,sl);tail=t.floats(arm+'_tail_'+tag,sl)
                if not np.array_equal(bf16_rne(h+tail),t.values[arm+'_'+tag][sl]):
                    raise ValueError('entire FP32 partial join/final BF16 bits differ')
        w=model.values[f'transformer_blocks.{layer}.img_mlp.out.weight']
        if w.dtype!=np.uint16 or w.shape!=(4096,12288):raise ValueError('original physical full down W required')
        for row,col in coordinates:
            hidden=t.floats('hidden_'+tag,(0,row,slice(None))).astype(np.float64)
            weight=(np.asarray(w[col,:6144],dtype=np.uint32)<<16).view(np.float32).astype(np.float64)
            expected=float(np.sum(hidden*weight,dtype=np.float64));actual=float(t.values['head_'+tag][0,row,col])
            if not math.isfinite(expected) or not math.isfinite(actual) or abs(actual-expected)>5e-4*max(1.,abs(expected)):
                raise ValueError('independent complete-K6144 GPU down witness failed')
            witnesses.append(dict(state=state,row=row,column=col,k=6144,expected=expected,actual=actual))
    return witnesses


def audit_arrays(t,r,frozen,model):
    digests={name:t.digest(name) for name in t.values};checks=[]
    for check in r['checks']:
        n,state=check['rows'],check['state'];tag=f'{n}_{state}';variant=int(state==1)
        actual=t.values['input_'+tag]
        for first in range(0,n,1024):
            count=min(1024,n-first);original=frozen.values['input_'+('b' if (variant+first//1024)%2 else 'a')]
            if not np.array_equal(actual[:,first:first+count],original[:,:count]):raise ValueError('actual constructed rows differ from original frozen bits')
        row=[]
        for index,arm in enumerate(screen.ARMS):
            q=quality(t,'gpu_'+tag,arm+'_'+tag);reported=check['qualities'][index]
            if q['relative_l2']>=.03 or not math.isclose(q['relative_l2'],reported['relative_l2'],rel_tol=1e-4,abs_tol=1e-10) or q['max_abs']!=reported['max_abs']:
                raise ValueError('independent complete GPU/ANE quality differs')
            row.append(q)
        incremental=quality(t,'single2048_'+tag,'family_'+tag)
        if incremental['relative_l2']>=.005 or not math.isclose(incremental['relative_l2'],check['family_vs_single2048']['relative_l2'],rel_tol=1e-4,abs_tol=1e-10):
            raise ValueError('independent incremental family quality differs')
        checks.append(dict(rows=n,state=state,qualities=row,incremental=incremental))
    for arm in screen.ARMS:
        if len({digests[f'{arm}_1024_{state}'] for state in range(3)})!=3:raise ValueError('changed X/W returned stale complete output')
    for name in t.values:
        for first in range(0,t.values[name].shape[1],64):
            if not np.all(np.isfinite(t.floats(name,(slice(None),slice(first,first+64),slice(None))))):
                raise ValueError('any saved full tensor must be finite')
    return dict(full_array_sha256=digests,checks=checks,fp32_join_all_bits_verified=True,
                full_k_down_witnesses=join_and_down(t,model))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-library',type=Path,required=True);p.add_argument('--native-source-archive',type=Path,required=True)
    p.add_argument('--observer',type=Path,default=ROOT/'build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib')
    p.add_argument('--window',nargs=2,type=Path,action='append',required=True,metavar=('ROOT','HELPER'))
    p.add_argument('--tool-source-archive',type=Path,action='append',default=[]);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('fresh independent output required')
    native=json.loads((a.native_library.parent/'runtime-build/runtime-build-manifest.json').read_text());archive_binding(a.native_source_archive,native)
    historical={}
    for path in a.tool_source_archive:
        with tarfile.open(path,'r:gz') as archive:
            for member in archive.getmembers():
                if member.isfile():historical.setdefault(member.name,set()).add(archive.extractfile(member).read())
    full={};records=[];frozen_cache={};models={}
    for root,helper in a.window:
        s=json.loads((root/'summary.json').read_text());r=json.loads((root/'trial/report.json').read_text())
        if (s['status']!='complete' or r['status']!='complete' or s['runtime_build_id']!=native['runtime_build_id'] or
            r['loaded_runtime_id']!=native['runtime_build_id'] or r['compiled_native_id']!=native['runtime_build_id'] or
            s['assets']['native_library']!=sha256_file(a.native_library) or s['assets']['library']!=sha256_file(helper) or
            s['assets']['observer']!=sha256_file(a.observer)):
            raise ValueError('complete exact native/helper source binding required')
        snapshot=json.loads((root/'source.json').read_text())
        if set(snapshot)!=set(s['source_sha256']):raise ValueError('complete captured tool source required')
        for name,digest in s['source_sha256'].items():
            raw=snapshot[name].encode()
            if hashlib.sha256(raw).hexdigest()!=digest or not(raw==(ROOT/name).read_bytes() or raw in historical.get(name,set())):
                raise ValueError('actual current-or-archived tool bytes differ')
        if r['cpp_sha256']!=s['source_sha256']['tools/profiling/qwen_base/w8_functions_real_probe.mm']:raise ValueError('actual compiled CPP differs')
        model_path=Path(s['model']);identity=model_snapshot(model_path)
        if identity!=s['model_snapshot']:raise ValueError('original model generation changed')
        for name in identity:
            path=(model_path/name).resolve(strict=True);key=str(path)
            if key not in full:
                full[key]=sha256_file(path)
                if path.name in PINS and full[key]!=PINS[path.name]:raise ValueError('full original published asset SHA differs')
        key=str(model_path)
        if key not in models:models[key]=Tensors(model_path/'diffusion_models/qwen_image_2.1_bf16.safetensors')
        model=models[key]
        for layer in (0,1):
            for leaf in ('gate_up','out'):
                name=f'transformer_blocks.{layer}.img_mlp.{leaf}'
                if model.digest(name+'.weight')!=r['source_hashes'][name]:raise ValueError('original complete BF16 W payload differs')
        cap_root=Path(s['capture']['root']);cap=cap_root/'trial/gpu-complement-arrays.safetensors';cap_key=str(cap)
        if sha256_file(cap_root/'summary.json')!=s['capture']['summary_sha256'] or sha256_file(cap)!=s['capture']['arrays_sha256']:
            raise ValueError('original frozen capture changed')
        old=json.loads((cap_root/'summary.json').read_text())
        if old.get('model_snapshot')!=identity or old.get('case')!=s['case'] or old.get('status')!='complete':
            raise ValueError('original frozen capture is not from the same actual source/task')
        if cap_key not in frozen_cache:frozen_cache[cap_key]=Tensors(cap)
        frozen=frozen_cache[cap_key]
        if frozen.digest('input_a')!=r['input_a_sha256'] or frozen.digest('input_b')!=r['input_b_sha256']:raise ValueError('actual frozen input bytes differ')
        analysis=screen.validate(r,s['repeats'])
        if analysis!=s['analysis']:raise ValueError('independent balanced timings differ')
        memory_contract(root,s)
        array_path=root/'trial/arrays.safetensors'
        if sha256_file(array_path)!=s['arrays_sha256']:raise ValueError('original full result payload changed')
        arrays=Tensors(array_path,specifications());data=audit_arrays(arrays,r,frozen,model)
        if model_snapshot(model_path)!=identity:raise ValueError('model generation changed during independent audit')
        observed=screen.selectors(json.loads((root/'trial/observer.json').read_text()),r) if s['mode']=='timeline' else None
        if s['mode']=='timeline' and observed!=s['observer']:raise ValueError('independent actual selector counters differ')
        records.append(dict(root=str(root),case=s['case'],mode=s['mode'],reverse=s['reverse'],analysis=analysis,arrays=data,observer=observed,
                            performance_load_gate_passed=s['performance_load_gate_passed'],load_error=s.get('load_error')))
        print(json.dumps(dict(window=str(root),full_arrays_verified=True,load_gate=s['performance_load_gate_passed'])),flush=True)
    write_json(a.output,dict(status='complete',qualification_passed=False,scope='real weights/frozen whole FFN only; no natural long prefill, DiT block or image speed qualification',
        full_asset_sha256=full,native_source_archive_sha256=sha256_file(a.native_source_archive),windows=records,audit_script_sha256=sha256_file(Path(__file__))))


if __name__=='__main__':main()
