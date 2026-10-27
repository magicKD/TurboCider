#!/usr/bin/env python3
"""Independent I8/F32 sources, full outputs and complete-K CPU dot witnesses."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np

import convrot_factorized_collect as screen
from collect import ROOT,sha256_file,write_json
from down_audit import read_components
from gpu_dual_request_audit import memory_contract
from qkv_audit import quality


def read_weights(path):
    with path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        if not 2<=n<=16<<20:raise ValueError('bounded original source header required')
        h=json.loads(f.read(n))
    q=[];s=[]
    for layer in (0,1):
        p=f'transformer_blocks.{layer}.img_mlp.out'
        for name,dtype,shape,target in ((p+'.weight','I8',(4096,12288),q),(p+'.weight_scale','F32',(4096,1),s)):
            info=h[name];begin,end=info['data_offsets'];item=1 if dtype=='I8' else 4
            if info['dtype']!=dtype or info['shape']!=list(shape) or end-begin!=int(np.prod(shape))*item:raise ValueError('original complete I8/F32 extent differs')
            target.append(np.memmap(path,mode='r',dtype='i1' if dtype=='I8' else '<f4',offset=8+n+begin,shape=shape))
    return q,s


def synthetic(r):
    m,k,n,wp,rb,cb=(r[x] for x in ('m','k','n','physical_pitch','row_begin','column_begin'));rows=n+rb+7
    ids=np.arange((rows+2)*wp,dtype=np.int64).reshape(rows+2,wp)
    q0=((ids*37+13)%256-128)[1:rows+1].astype('i1');q1=(((q0.astype(np.int32)+128)*7+11)%256-128).astype('i1')
    s0=(np.arange(rows,dtype=np.int32)%17-8).astype('<f4')[:,None]*np.float32(.00012345)
    a=(np.arange(m*k).reshape(1,m,k)%29-14).astype('<f4')/32
    return [q0,q1],[s0,s0*np.float32(.875)],dict(a=a,b=a*np.float32(.75)+np.float32(.015625))


def oracle(q,s,x,y,rb,cb,points=None):
    n=y.shape[2];k=x.shape[2]
    if points is None:
        weights=q[rb:rb+n,cb:cb+k].astype('<f4')*s[rb:rb+n]
        expected=x[0].astype(np.float64)@weights.astype(np.float64).T
        actual=y[0].astype(np.float64);err=actual-expected
        if not np.isfinite(expected).all() or not np.isfinite(actual).all() or not np.isfinite(err).all():raise ValueError('finite independent F64 oracle and complete output required')
        relative=float(np.linalg.norm(err)/max(np.linalg.norm(expected),1e-30))
        if not np.isfinite(relative) or relative>=1e-4 or np.max(np.abs(err))>1e-4:raise ValueError('independent full synthetic complete-K dot differs')
        return dict(points=int(y.size),relative_l2=relative,max_abs=float(np.max(np.abs(err))),scope='full synthetic output, original rounded-F32 dequantized W, F64 CPU dot')
    records=[];energy=error=0.
    for row,col in points:
        w=q[rb+col,cb:cb+k].astype('<f4')*np.float32(s[rb+col,0]);expected=float(np.dot(x[0,row].astype(np.float64),w.astype(np.float64)));actual=float(y[0,row,col]);delta=actual-expected
        if not np.isfinite(expected) or not np.isfinite(actual) or abs(delta)>5e-4*max(1.,abs(expected)):raise ValueError('actual complete-K original-source F64 witness differs')
        records.append(dict(row=row,column=col,expected=expected,actual=actual,absolute_error=abs(delta)));energy+=expected*expected;error+=delta*delta
    relative=(error/max(energy,1e-30))**.5
    if relative>=1e-4:raise ValueError('sampled complete-K F64 relative error exceeds1e-4')
    return dict(points=len(records),relative_l2=relative,witnesses=records,scope='sampled output coordinates, complete logical K; not a full dense-model CPU oracle')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('native-library','probe-library','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--cohort',type=Path,action='append',required=True);a=p.parse_args()
    if a.output.exists():p.error('fresh independent row-scale audit required')
    manifest=json.loads((a.native_library.parent/'runtime-build/runtime-build-manifest.json').read_text());records=[];real_weights={};controls={};replay={}
    for root in a.cohort:
        s=json.loads((root/'summary.json').read_text());r=json.loads((root/'trial/report.json').read_text())
        if (s['status']!='complete' or s['assets']['native_library']!=sha256_file(a.native_library) or s['assets']['library']!=sha256_file(a.probe_library) or
            r['compiled_native_id']!=manifest['runtime_build_id'] or r['loaded_runtime_id']!=manifest['runtime_build_id']):raise ValueError('actual complete helper/native binding differs')
        snapshot=json.loads((root/'source.json').read_text())
        if set(snapshot)!=set(s['source_sha256']):raise ValueError('complete row-scale source snapshot required')
        for name,digest in s['source_sha256'].items():
            if hashlib.sha256(snapshot[name].encode()).hexdigest()!=digest or sha256_file(ROOT/name)!=digest:raise ValueError('bound row-scale source differs')
        if r['cpp_sha256']!=s['source_sha256']['tools/profiling/qwen_base/convrot_factorized_probe.mm'] or r['header_sha256']!=s['source_sha256']['tools/profiling/qwen_base/convrot_factorized.hpp']:raise ValueError('actual compiled kernel/helper source differs')
        analysis=screen.validate(r,s['repeats'],s['real'],s['bm'],s['bn']);file=root/'trial/arrays.safetensors'
        if analysis!=s['analysis'] or sha256_file(file)!=s['arrays_sha256']:raise ValueError('actual full timings or array bytes changed')
        m,k,n=r['m'],r['k'],r['n'];specs={f'input_{t}':('BF16',(1,m,k)) for t in ('a','b')}
        specs.update({arm+'_'+t:('F32',(1,m,n)) for arm in screen.ARMS for t in ('a','b','w')});v=read_components(file,specs)
        if any(not np.isfinite(x).all() for x in v.values()):raise ValueError('all complete finite outputs required')
        hashes={key:hashlib.sha256(value.astype('<f4',copy=False).tobytes()).hexdigest() for key,value in v.items()}
        if s['real']:
            path=Path(s['original_weight_file'])
            if str(path) not in real_weights:
                if sha256_file(path)!='cb74113cb03faecd79611b01fd7fd642f0aa60d6f0b95086abee214d75eaa57d':raise ValueError('full original ConvRot ModelScope source SHA differs')
                real_weights[str(path)]=read_weights(path)
            q,sc=real_weights[str(path)];cap=Path(s['capture']['root']);old_s=json.loads((cap/'summary.json').read_text());oldfile=cap/'trial/gpu-complement-arrays.safetensors'
            if sha256_file(cap/'summary.json')!=s['capture']['summary_sha256'] or sha256_file(oldfile)!=s['capture']['arrays_sha256'] or old_s['status']!='complete':raise ValueError('complete frozen input source changed')
            names={'input_a','input_b'}|{f'{part}_{kind}_{t}' for part in ('hidden','partial') for kind in ('original','fused') for t in ('a','b')}
            old=read_components(oldfile,{name:('F32' if name.startswith('partial') else 'BF16',(1,1024,6144 if name.startswith('hidden') else 4096)) for name in names})
            if any(not np.array_equal(v['input_'+t].view('<u4'),old['hidden_original_'+t].view('<u4')) for t in ('a','b')):raise ValueError('actual ready BF16 input differs from complete frozen bytes')
            points=[(0,0),(m-1,n-1),(0,n-1),(m-1,0)]+[((i*137+11)%m,(i*977+31)%n) for i in range(28)]
        else:
            q,sc,inputs=synthetic(r)
            if any(not np.array_equal(v['input_'+t],inputs[t]) for t in ('a','b')):raise ValueError('independent dyadic synthetic input recipe differs')
            points=None
        for layer in (0,1):
            if hashlib.sha256(q[layer].tobytes()).hexdigest()!=r['source_hashes'][str(layer)]['codes'] or hashlib.sha256(sc[layer].tobytes()).hexdigest()!=r['source_hashes'][str(layer)]['row_scale']:
                raise ValueError('actual complete I8 codes/F32 row scale payload differs')
        checks={};oracles={}
        for t in ('a','b','w'):
            x=v['input_b' if t=='b' else 'input_a'];layer=int(t=='w')
            for arm in screen.ARMS:
                oracles[arm+'_'+t]=oracle(q[layer],sc[layer],x,v[arm+'_'+t],r['row_begin'],r['column_begin'],points)
            checks[t]={}
            for arm,label in ((screen.ARMS[1],'packed'),(screen.ARMS[2],'factored')):
                metric=quality(v[screen.ARMS[0]+'_'+t],v[arm+'_'+t]);checks[t][label]=metric
                if any(not np.isclose(metric[key],r['checks'][t][label][key],rtol=2e-5,atol=1e-12) for key in ('relative_l2','max_abs')):raise ValueError('independent full-array quality differs')
        common={key:val for key,val in hashes.items() if not key.startswith('factored_')};anchor=controls.setdefault(s['real'],common)
        if anchor!=common:raise ValueError('strong original controls changed across tiles/repeats')
        key=(s['real'],s['bm'],s['bn']);prior=replay.setdefault(key,hashes)
        if prior!=hashes:raise ValueError('same tile repeat/timeline complete outputs changed')
        observed=None
        if s['mode']=='timeline':
            raw=json.loads((root/'trial/observer.json').read_text())
            if raw['source_sha256']!=s['source_sha256']['tools/profiling/qwen_base/observer.mm']:raise ValueError('actual observer source differs')
            observed=screen.observer_contract(raw,r)
            if observed!=s['observer']:raise ValueError('actual completed GPU scopes differ')
        records.append(dict(cohort=root.name,analysis=analysis,full_array_sha256=hashes,quality=checks,oracles=oracles,observer=observed,memory=memory_contract(root,s),load_error=s.get('load_error')))
    result=dict(schema='tc-convrot-row-scale-dot-audit-v1',status='complete',qualification_passed=False,cohorts=records,
        native_sha256=sha256_file(a.native_library),probe_sha256=sha256_file(a.probe_library),script_sha256=sha256_file(Path(__file__)),
        scope='ready source dot only; F32 reassociation not globally lossless; NO H256 rotation/FFN/ANE/image gain or adoption qualification')
    a.output.mkdir(parents=True);write_json(a.output/'audit.json',result);print(json.dumps(dict(status='complete',cohorts=len(records))))


if __name__=='__main__':main()
