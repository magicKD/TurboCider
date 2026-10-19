#!/usr/bin/env python3
"""Independent mixed FP16-hidden/BF16-W down arrays and source audit."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np

import mixed_hidden_collect as screen
import joint_audit
from collect import ROOT,model_snapshot,sha256_file,write_json
from down_audit import original_down,sampled_dot,read_components
from gpu_dual_request_audit import memory_contract
from qkv_audit import quality
from vpipe_audit import trace


def read(path,specs):
    with path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        if not 2<=n<=16<<20:raise ValueError('bounded complete tensor header required')
        h={k:v for k,v in json.loads(f.read(n)).items() if k!='__metadata__'}
    if set(h)!=set(specs):raise ValueError('exact mixed BF16/F16/F32 tensor set required')
    result={};extents=[];data_bytes=path.stat().st_size-8-n
    for name,(dtype,shape) in specs.items():
        if dtype not in ('BF16','F16','F32'):raise ValueError('unsupported mixed component dtype')
        info=h[name];begin,end=info['data_offsets'];item=4 if dtype=='F32' else 2
        if (info['dtype']!=dtype or info['shape']!=list(shape) or type(begin) is not int or type(end) is not int or
            not 0<=begin<=end<=data_bytes or begin%item or end-begin!=int(np.prod(shape))*item):raise ValueError('actual typed complete extent/shape differs')
        value=np.memmap(path,mode='r',dtype='<f4' if dtype=='F32' else '<u2',offset=8+n+begin,shape=shape)
        result[name]=(value.astype(np.uint32)<<16).view('<f4') if dtype=='BF16' else value.view('<f2').astype('<f4') if dtype=='F16' else value
        extents.append((begin,end))
    cursor=0
    for begin,end in sorted(extents):
        if begin!=cursor:raise ValueError('overlapping/gapped tensor payload')
        cursor=end
    if cursor!=data_bytes:raise ValueError('extra tensor payload')
    return result


def bf16(v):
    bits=np.asarray(v,dtype='<f4').view('<u4');return ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view('<f4')


def verify_arrays(v,r):
    checks={};hashes={n:hashlib.sha256(x.astype('<f4',copy=False).tobytes()).hexdigest() for n,x in v.items()}
    if any(not np.isfinite(x).all() for x in v.values()):raise ValueError('complete exported normal/domain arrays must be finite')
    for tag in ('a','b'):
        expected_half=v['input_'+tag].astype('<f2').astype('<f4')
        if not np.array_equal(expected_half.view('<u4'),v['half_'+tag].view('<u4')) or not np.array_equal(bf16(expected_half).view('<u4'),v['round_'+tag].view('<u4')):
            raise ValueError('actual complete F16 encoding/BF16 roundtrip differs from CPU IEEE recipe')
        if hashes['mixed-ready_'+tag]!=hashes['mixed-with-cast_'+tag]:raise ValueError('same mixed values but per-call cast changed complete partial bits')
        checks[tag]={}
        for key,left,right in (('cast_quality','input_','round_'),('incremental_partial_quality','original-bf16_','mixed-ready_'),('same_value_partial_quality','rounded-bf16_','mixed-ready_')):
            q=quality(v[left+tag],v[right+tag]);checks[tag][key]=q
            for name in ('relative_l2','max_abs'):
                if not np.isclose(q[name],r['checks'][tag][key][name],rtol=2e-5,atol=1e-12):raise ValueError('full independently recomputed quality differs from GPU reduction')
    if hashes['input_a']==hashes['input_b'] or hashes['original-bf16_a']==hashes['original-bf16_b']:raise ValueError('changed actual source hidden returned stale output')
    return dict(full_array_sha256=hashes,checks=checks,mixed_ready_cast_bits_exact=True,
        same_value_bf16_mixed_bits_exact=all(hashes['rounded-bf16_'+tag]==hashes['mixed-ready_'+tag] for tag in ('a','b')))


def synthetic_oracle(values,r):
    m,k,n,pitch,rb,cb=(r[x] for x in ('m','k','n','physical_weight_pitch','row_begin','column_begin'))
    source_ids=np.arange((m+2)*k,dtype=np.int64).reshape(m+2,k)[1:m+1];base=source_ids%29-14
    xints={'a':base*4,'b':base*3+2}
    ids=np.arange((n+rb+4)*pitch,dtype=np.int64).reshape(n+rb+4,pitch);wints=(((ids%257)*37+13)%31-15)[2:n+rb+2]
    w=(wints.astype('<f4')/128);bits=(w.view('<u4')>>16).astype('<u2')
    if hashlib.sha256(bits.tobytes()).hexdigest()!=r['source_weight_sha256']:raise ValueError('independent full synthetic W recipe/physical slice differs')
    logical=wints[rb:rb+n,cb:cb+k];expected={}
    for tag in ('a','b'):
        x=xints[tag].astype('<f4')/128
        if not np.array_equal(x[None],values['input_'+tag]):raise ValueError('synthetic source offset/periodic input differs')
        expected[tag]=(xints[tag]@logical.T).astype('<f4')[None]/np.float32(16384)
        for arm in screen.ARMS:
            if not np.array_equal(expected[tag].view('<u4'),values[arm+'_'+tag].view('<u4')):raise ValueError('all synthetic normal partials must match complete exact integer-dot oracle')
    for domain,factor in (('underflow',2**-24),('subnormal',2**-16),('zero',0.)):
        x=values['input_a']*np.float32(factor);half=x.astype('<f2').astype('<f4');wanted=expected['a']*np.float32(factor)
        if not np.array_equal(x.view('<u4'),values[domain+'_input'].view('<u4')) or not np.array_equal(half.view('<u4'),values[domain+'_half'].view('<u4')):
            raise ValueError('domain source/cast bytes differ')
        if not np.array_equal(wanted,values[domain+'_original']):raise ValueError('nonzero tiny/zero original reference differs from exact scaled oracle')
        target=np.zeros_like(wanted) if domain in ('underflow','zero') else wanted
        for suffix in ('mixed','rounded'):
            if not np.array_equal(target,values[domain+'_'+suffix]):raise ValueError('actual mixed domain output differs from explicit format-loss oracle')
    return dict(full_normal_outputs=2*m*n*len(screen.ARMS),full_domain_outputs=3*m*n*3,all_exact_dyadic_integer_oracle=True,
        underflow_relative_l2=quality(values['underflow_original'],values['underflow_mixed'])['relative_l2'],scope='complete synthetic full-K integer dot and exact power-of-two domains, not actual dense FFN full CPU oracle')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('native-library','probe-library','native-source-archive','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cohort',type=Path,action='append',required=True);p.add_argument('--model',type=Path,default=ROOT/'models/Comfy-Org-Qwen-Image-2.1');args=p.parse_args()
    if args.output.exists():p.error('fresh independent mixed down audit required')
    manifest=json.loads((args.native_library.parent/'runtime-build/runtime-build-manifest.json').read_text());joint_audit.archive_contract(args.native_source_archive,manifest)
    weights=original_down(args.model);payload_hash=hashlib.sha256(weights.tobytes()).hexdigest();records=[];traces=[];anchors={}
    for root in args.cohort:
        s=json.loads((root/'summary.json').read_text());r=json.loads((root/'trial/report.json').read_text());real=s['fixture']!='synthetic'
        if (s['status']!='complete' or s['assets']['library']!=sha256_file(args.probe_library) or s['assets']['native_library']!=sha256_file(args.native_library) or
            r['compiled_native_id']!=manifest['runtime_build_id'] or r['loaded_runtime_id']!=manifest['runtime_build_id'] or
            sha256_file(Path(s['installed_mpp_header']))!=s['assets']['installed_mpp_header']):raise ValueError('actual native/helper/installed SDK binding differs')
        snapshot=json.loads((root/'source.json').read_text())
        if set(snapshot)!=set(s['source_sha256']):raise ValueError('complete source snapshot required')
        for name,digest in s['source_sha256'].items():
            if hashlib.sha256(snapshot[name].encode()).hexdigest()!=digest or sha256_file(ROOT/name)!=digest:raise ValueError('actual shader/helper/tool source differs')
        if r['cpp_sha256']!=s['source_sha256']['tools/profiling/qwen_base/mixed_hidden_probe.mm'] or r['header_sha256']!=s['source_sha256']['tools/profiling/qwen_base/mixed_hidden.hpp']:raise ValueError('actual compiled kernel/header differs')
        analysis=screen.validate(r,s['repeats'],real)
        if analysis!=s['analysis'] or sha256_file(root/'trial/arrays.safetensors')!=s['arrays_sha256']:raise ValueError('actual complete timings or array payload differs')
        m,k,n=r['m'],r['k'],r['n'];specs={f'{part}_{v}':('F16' if part=='half' else 'BF16',(1,m,k)) for part in ('input','half','round') for v in ('a','b')}
        specs.update({a+'_'+v:('F32',(1,m,n)) for a in screen.ARMS for v in ('a','b')})
        if not real:specs.update({domain+'_'+suffix:('BF16' if suffix=='input' else 'F16' if suffix=='half' else 'F32',(1,m,k if suffix in ('input','half') else n)) for domain in ('underflow','subnormal','zero') for suffix in ('input','half','original','mixed','rounded')})
        values=read(root/'trial/arrays.safetensors',specs);checked=verify_arrays(values,r);oracle=None;frozen=None
        if not real:oracle=synthetic_oracle(values,r)
        else:
            if model_snapshot(args.model)!=s['model_snapshot'] or r['source_weight_sha256']!=payload_hash:raise ValueError('actual original BF16 down matrix changed')
            cap=Path(s['capture']['root']);old_s=json.loads((cap/'summary.json').read_text());old_r=json.loads((cap/'trial/report.json').read_text());file=cap/'trial/gpu-complement-arrays.safetensors'
            request=old_r['actual_source_request']
            if (old_s['status']!='complete' or sha256_file(cap/'summary.json')!=s['capture']['summary_sha256'] or sha256_file(file)!=s['capture']['arrays_sha256'] or
                (request['model'],request['width'],request['height'],request['steps'],request['seed'],request['lora_count'])!=('qwen-image-2.1',512,512,40,73,0) or
                request['operation']!='image.'+s['fixture'] or old_r['source_arrays_sha256']['transformer_blocks.0.img_mlp.out']!=payload_hash):raise ValueError('actual full-base40 frozen capture/current source differs')
            old_names={'input_a','input_b'}|{f'{part}_{kind}_{v}' for part in ('hidden','partial') for kind in ('original','fused') for v in ('a','b')}
            old=read_components(file,{name:('F32',(1,1024,4096)) if name.startswith('partial') else ('BF16',(1,1024,6144 if name.startswith('hidden') else 4096)) for name in old_names})
            for v in ('a','b'):
                if not np.array_equal(values['input_'+v].view('<u4'),old['hidden_original_'+v].view('<u4')) or not np.array_equal(values['original-bf16_'+v].view('<u4'),old['partial_original_'+v].view('<u4')):
                    raise ValueError('current actual hidden or strong original F32 partial changed frozen control bits')
            coordinates=[(0,0),(1023,4095),(0,4095),(1023,0)]+[((i*137+11)%1024,(i*977+31)%4096) for i in range(28)]
            oracle={a+'_'+v:sampled_dot(weights,values[('input_' if a=='original-bf16' else 'round_' if a=='rounded-bf16' else 'half_')+v],values[a+'_'+v],coordinates) for a in screen.ARMS for v in ('a','b')}
            frozen=dict(original_hidden_and_f32_partial_bits_exact=True,source_sha256=sha256_file(file),scope='historical arrays/source only, not timing denominator')
        old=anchors.setdefault(s['fixture'],checked['full_array_sha256'])
        if old!=checked['full_array_sha256']:raise ValueError('same source real/synthetic repeat/timeline complete arrays changed')
        observed=None
        if s['mode']=='timeline':
            raw=json.loads((root/'trial/observer.json').read_text())
            if raw['source_sha256']!=s['source_sha256']['tools/profiling/qwen_base/observer.mm']:raise ValueError('actual observer differs')
            observed=screen.observer_contract(raw,r)
            if observed!=s['observer']:raise ValueError('actual completed GPU work scopes differ')
            traces.append((root.name,trace(raw)))
        records.append(dict(cohort=root.name,fixture=s['fixture'],mode=s['mode'],analysis=analysis,arrays=checked,oracle=oracle,frozen=frozen,
            observer=observed,memory=memory_contract(root,s),load_error=s.get('load_error'),summary_sha256=sha256_file(root/'summary.json')))
    result=dict(schema='tc-mixed-hidden-down-audit-v1',status='complete',qualification_passed=False,native_changed=False,cohorts=records,
        runtime_build_id=manifest['runtime_build_id'],native_sha256=sha256_file(args.native_library),probe_sha256=sha256_file(args.probe_library),
        native_source_archive_sha256=sha256_file(args.native_source_archive),original_down_sha256=payload_hash,script_sha256=sha256_file(Path(__file__)),
        scope='actual frozen FFN GPU complement hidden ready/down-only and synthetic domains, no W cast/copy, no complete hybrid/image/physical precision claim')
    args.output.mkdir(parents=True);write_json(args.output/'audit.json',result)
    for name,events in traces:write_json(args.output/(name+'-trace.json'),events)
    print(json.dumps(dict(status='complete',cohorts=len(records)),indent=2))


if __name__=='__main__':main()
