#!/usr/bin/env python3
"""Independent full arrays, frozen source, native ABI, memory and selector audit."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import tarfile
import numpy as np

import compiled_complement_collect as gpu
import compiled_complement_hybrid_collect as hybrid
import ffn_projection_audit
import joint_audit
from collect import ROOT,model_snapshot,sha256_file,write_json
from down_audit import read_components,sampled_dot
from gpu_dual_request_audit import memory_contract
from qkv_audit import quality
from vpipe_audit import trace


def hashes(values):return {name:hashlib.sha256(v.astype('<f4',copy=False).tobytes()).hexdigest() for name,v in values.items()}


def check_arrays(values,r,is_hybrid):
    if any(not np.isfinite(v).all() for v in values.values()):raise ValueError('all complete finite arrays required')
    h=hashes(values);checks={}
    for tag in ('a','b','w'):
        if is_hybrid:
            left,right=values['raw_'+tag],values['compiled_'+tag]
            if h['raw_'+tag]!=h['compiled_'+tag]:raise ValueError('compiled GPU callback changed actual complete hybrid output bits')
            checks[tag]=dict(incremental=quality(left,right),gpu_quality=quality(values['gpu_'+tag],right))
            for kind in ('incremental','gpu_quality'):
                for key in ('relative_l2','max_abs'):
                    if not np.isclose(checks[tag][kind][key],r['checks'][tag][kind][key],rtol=2e-5,atol=1e-12):raise ValueError('independent complete hybrid quality differs')
        else:
            checks[tag]={}
            for part in ('hidden','partial'):
                a,b='raw_'+part+'_'+tag,'compiled_'+part+'_'+tag
                if h[a]!=h[b]:raise ValueError('compiled GPU callback changed complete BF16/F32 bits')
                checks[tag][part]=quality(values[a],values[b])
                for key in ('relative_l2','max_abs'):
                    if checks[tag][part][key]!=r['checks'][tag][part][key]:raise ValueError('independent complete component quality differs')
    outputs=['raw_'+tag for tag in ('a','b','w')] if is_hybrid else ['raw_partial_'+tag for tag in ('a','b','w')]
    if len({h[n] for n in outputs})!=3 or h['input_a']==h['input_b']:raise ValueError('changed X/W controls returned stale data')
    return dict(full_array_sha256=h,checks=checks,all_incremental_output_bits_exact=True)


def weights(model):
    path=model/'diffusion_models/qwen_image_2.1_bf16.safetensors'
    with path.open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0]
        if not 2<=size<=16<<20:raise ValueError('bounded original checkpoint header required')
        header=json.loads(f.read(size))
    result={}
    for layer in (0,1):
        info=header[f'transformer_blocks.{layer}.img_mlp.out.weight'];begin,end=info['data_offsets']
        if info['dtype']!='BF16' or info['shape']!=[4096,12288] or end-begin!=4096*12288*2:raise ValueError('original physical BF16 down extent required')
        result[layer]=np.memmap(path,mode='r',dtype='<u2',offset=8+size+begin,shape=(4096,12288))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('native-library','native-source-archive','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('gpu-cohort','hybrid-cohort'):p.add_argument('--'+k,nargs=2,type=Path,metavar=('ROOT','HELPER'),action='append',default=[])
    p.add_argument('--tool-source-archive',type=Path,action='append',default=[])
    p.add_argument('--model',type=Path,default=ROOT/'models/Comfy-Org-Qwen-Image-2.1');args=p.parse_args()
    if args.output.exists() or not args.gpu_cohort:p.error('fresh audit and actual GPU controls required')
    manifest=json.loads((args.native_library.parent/'runtime-build/runtime-build-manifest.json').read_text());joint_audit.archive_contract(args.native_source_archive,manifest)
    historical={}
    for path in args.tool_source_archive:
        with tarfile.open(path,'r:gz') as archive:
            for member in archive.getmembers():
                if member.isfile():historical.setdefault(member.name,set()).add(archive.extractfile(member).read())
    identity=model_snapshot(args.model);payloads=ffn_projection_audit.original_hashes(args.model);w=weights(args.model);records=[];traces=[];anchors={}
    for is_hybrid,items in ((False,args.gpu_cohort),(True,args.hybrid_cohort)):
        for root,helper in items:
            s=json.loads((root/'summary.json').read_text());r=json.loads((root/'trial/report.json').read_text());screen=hybrid if is_hybrid else gpu
            if (s['status']!='complete' or s['assets']['library']!=sha256_file(helper) or s['assets']['native_library']!=sha256_file(args.native_library) or
                s['model_snapshot']!=identity or s['runtime_build_id']!=manifest['runtime_build_id'] or r['compiled_native_id']!=manifest['runtime_build_id'] or
                r['loaded_runtime_id']!=manifest['runtime_build_id'] or r['source_hashes']!=payloads):raise ValueError('actual original FFN/native/helper/source identity differs')
            source=json.loads((root/'source.json').read_text())
            if set(source)!=set(s['source_sha256']):raise ValueError('complete helper/tool snapshot required')
            for name,digest in s['source_sha256'].items():
                raw=source[name].encode()
                if hashlib.sha256(raw).hexdigest()!=digest or not (raw==(ROOT/name).read_bytes() or raw in historical.get(name,set())):raise ValueError('actual compiled/current-or-archived source differs')
            name='compiled_complement_hybrid_probe.mm' if is_hybrid else 'compiled_complement_probe.mm'
            if r['cpp_sha256']!=s['source_sha256']['tools/profiling/qwen_base/'+name] or r['header_sha256']!=s['source_sha256']['tools/profiling/qwen_base/compiled_complement.hpp']:raise ValueError('compiled header/translation-unit identity differs')
            analysis=screen.validate(r,s['repeats']);file=root/'trial/arrays.safetensors'
            if analysis!=s['analysis'] or sha256_file(file)!=s['arrays_sha256']:raise ValueError('actual complete timings/arrays changed')
            specs={f'input_{v}':('BF16',(1,1024,4096)) for v in ('a','b')}
            if is_hybrid:specs.update({a+'_'+v:('BF16',(1,1024,4096)) for a in ('raw','compiled','gpu') for v in ('a','b','w')})
            else:specs.update({a+'_'+part+'_'+v:('BF16' if part=='hidden' else 'F32',(1,1024,6144 if part=='hidden' else 4096))
                for a in ('raw','compiled') for part in ('hidden','partial') for v in ('a','b','w')})
            values=read_components(file,specs);checked=check_arrays(values,r,is_hybrid)
            cap=Path(s['capture']['root']);old_s=json.loads((cap/'summary.json').read_text());old_r=json.loads((cap/'trial/report.json').read_text());old_file=cap/'trial/gpu-complement-arrays.safetensors'
            request=old_r['actual_source_request']
            if (old_s['status']!='complete' or sha256_file(cap/'summary.json')!=s['capture']['summary_sha256'] or sha256_file(old_file)!=s['capture']['arrays_sha256'] or
                old_r['source_arrays_sha256']!=payloads or (request['model'],request['width'],request['height'],request['steps'],request['seed'],request['lora_count'])!=('qwen-image-2.1',512,512,40,73,0) or
                request['operation']!='image.'+s['case']):raise ValueError('actual original base40 capture differs')
            old_names={'input_a','input_b'}|{f'{part}_{kind}_{v}' for part in ('hidden','partial') for kind in ('original','fused') for v in ('a','b')}
            old=read_components(old_file,{n:('F32' if n.startswith('partial') else 'BF16',(1,1024,6144 if n.startswith('hidden') else 4096)) for n in old_names})
            for v in ('a','b'):
                if not np.array_equal(values['input_'+v].view('<u4'),old['input_'+v].view('<u4')):raise ValueError('frozen complete actual input changed')
                if not is_hybrid:
                    for part in ('hidden','partial'):
                        if not np.array_equal(values['raw_'+part+'_'+v].view('<u4'),old[part+'_original_'+v].view('<u4')):raise ValueError('strong original native callback changed historical complete output bits')
            anchor=anchors.setdefault((is_hybrid,s['case']),checked['full_array_sha256'])
            if anchor!=checked['full_array_sha256']:raise ValueError('same case repeat/timeline complete arrays changed')
            oracle=None
            if not is_hybrid:
                coords=[(0,0),(1023,4095),(0,4095),(1023,0)]+[((i*137+11)%1024,(i*977+31)%4096) for i in range(28)]
                oracle={a+'_'+v:sampled_dot(w[int(v=='w')],values[a+'_hidden_'+v],values[a+'_partial_'+v],coords) for a in ('raw','compiled') for v in ('a','b','w')}
            observed=None
            if s['mode']=='timeline':
                raw=json.loads((root/'trial/observer.json').read_text())
                if raw['source_sha256']!=s['source_sha256']['tools/profiling/qwen_base/observer.mm']:raise ValueError('actual observer source differs')
                observed=screen.observer_contract(raw,r)
                if observed!=s['observer']:raise ValueError('actual GPU/ANE observation differs')
                traces.append((root.name,trace(raw)))
            records.append(dict(cohort=root.name,case=s['case'],mode=s['mode'],hybrid_ffn=is_hybrid,analysis=analysis,arrays=checked,down_oracle=oracle,observer=observed,
                memory=memory_contract(root,s),load_error=s.get('load_error'),summary_sha256=sha256_file(root/'summary.json')))
    result=dict(schema='tc-compiled-complement-audit-v1',status='complete',qualification_passed=False,native_changed=False,cohorts=records,
        original_ffn_payload_sha256=payloads,runtime_build_id=manifest['runtime_build_id'],native_sha256=sha256_file(args.native_library),
        native_source_archive_sha256=sha256_file(args.native_source_archive),script_sha256=sha256_file(Path(__file__)),
        tool_source_archive_sha256={str(p):sha256_file(p) for p in args.tool_source_archive},
        scope='complete GPU complement and staged hybrid FFN on frozen actual input; no attention/QKV/whole DiT/full40 image or physical precision/overlap qualification')
    args.output.mkdir(parents=True);write_json(args.output/'audit.json',result)
    for name,events in traces:write_json(args.output/(name+'-trace.json'),events)
    print(json.dumps(dict(status='complete',cohorts=len(records))))


if __name__=='__main__':main()
