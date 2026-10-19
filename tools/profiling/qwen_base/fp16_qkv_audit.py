#!/usr/bin/env python3
"""Independent real-input QKV FP16/W8 arrays, provenance and timeline audit."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import tarfile
import numpy as np

import fp16_qkv_collect as screen
import qkv_stage_audit as source
import joint_audit
from collect import ROOT,model_snapshot,sha256_file,write_json
from down_audit import read_components
from gpu_dual_request_audit import memory_contract
from qkv_audit import quality,cpu_projection_oracle
from vpipe_audit import trace


def tensor_bits(path,name,shape):
    with path.open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0]
        if not 2<=size<=16<<20:raise ValueError('bounded raw tensor header required')
        h=json.loads(f.read(size))[name]
    begin,end=h['data_offsets']
    if h['dtype']!='BF16' or h['shape']!=list(shape) or end-begin!=int(np.prod(shape))*2 or end>path.stat().st_size-8-size:
        raise ValueError('complete original BF16 source/target tensor required')
    return np.memmap(path,mode='r',dtype='<u2',offset=8+size+begin,shape=shape)


def compare(values,arms):
    expected={'input_a','input_b'}|{a+'_'+v for a in arms for v in ('a','b','w')}
    if set(values)!=expected or any(not np.isfinite(v).all() for v in values.values()):raise ValueError('complete finite real-input raw projections required')
    hashes={n:hashlib.sha256(v.astype('<f4',copy=False).tobytes()).hexdigest() for n,v in values.items()};checks={}
    for variant in ('a','b','w'):
        for arm in arms[1:]:
            reference=values['gpu_'+variant];value=values[arm+'_'+variant];q=quality(reference,value);tail=quality(reference[:,512:],value[:,512:]);checks[arm+'_'+variant]=dict(full=q,ane_tail=tail)
            if q['relative_l2']>=(.004 if arm.startswith('fp16') else .02) or tail['relative_l2']>=(.006 if arm.startswith('fp16') else .03):
                raise ValueError('actual FP16/W8 complete or ANE-tail projection exceeds predeclared component bound')
            if not np.array_equal(value[:,:512],values['native-w8_'+variant][:,:512]):raise ValueError('candidate changed native GPU head arithmetic')
    if hashes['input_a']==hashes['input_b'] or any(hashes[a+'_a'] in (hashes[a+'_b'],hashes[a+'_w']) for a in arms):
        raise ValueError('changed input/weight returned stale raw QKV')
    prepack_exact=all(hashes['fp16-single_'+v]==hashes['fp16-prepack_'+v] for v in ('a','b','w')) if 'fp16-prepack' in arms else None
    if prepack_exact is False:raise ValueError('same FP16 Program prepack timing control changed complete output bits')
    return dict(full_array_sha256=hashes,quality=checks,gpu_head_preserved=True,
        fp16_single_prepack_bits_exact=prepack_exact,
        fp16_tiled_single_bits_exact=(all(hashes['fp16-tiled_'+v]==hashes['fp16-single_'+v] for v in ('a','b','w')) if 'fp16-tiled' in arms else None),
        scope='complete real frozen source and raw QKV, numeric FP16/W8 comparison; not full-block/image or physical accumulator qualification')


def bf16(value):
    bits=np.asarray(value,dtype='<f4').view('<u4')
    rounded=(bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)
    return rounded.view('<f4')


def perturbed_input(a):
    # This fixture perturbs normalized X, whereas the historical component
    # perturbed hidden BEFORE layer_norm. Its old B projection is not a match.
    return bf16(bf16(a*np.float32(.984375))+np.float32(.00390625))


def witness(weights,x,y):
    coordinates=[(0,0),(511,4095),(512,8192),(1023,12287)]+[((i*137+11)%1024,(i*977+31)%12288) for i in range(28)]
    errors=[]
    for row,col in coordinates:
        bits=np.asarray(weights[col//4096][col%4096],dtype=np.uint32)<<16;w=bits.view(np.float32).astype(np.float64)
        expected=float(np.dot(x[0,row].astype(np.float64),w));actual=float(y[0,row,col]);error=abs(expected-actual)
        if not np.isfinite(expected) or error>.008*max(1.,abs(expected)):raise ValueError('sampled complete-K F64 GPU reference witness failed')
        errors.append(dict(row=row,column=col,expected_f64=expected,gpu_bf16=actual,abs_error=error))
    return dict(points=len(errors),witnesses=errors,scope='sampled coordinates, complete K4096 F64 dot from original BF16 W/X; not full CPU dense QKV')


def timeline_summary(raw,r,active):
    by={}
    for e in raw['events']:by.setdefault(e['request'],[]).append(e)
    import statistics
    result={}
    for arm in active:
        queues={};api=[];ordinals={}
        for i,s in enumerate(r['samples']):
            if s['warmup'] or s['arm']!=arm:continue
            current={}
            for e in by.get(100+i,[]):
                if e['name']=='ane.evaluate_api':api.append((e['end_ns']-e['begin_ns'])/1e6)
                if e['name']=='metal.command_buffer':
                    queues.setdefault(e['queue'],[]).append(dict(gpu=(e['gpu_end_ns']-e['gpu_begin_ns'])/1e6,life=(e['end_ns']-e['begin_ns'])/1e6));current.setdefault(e['queue'],[]).append(e)
            for q,events in current.items():
                for ordinal,e in enumerate(sorted(events,key=lambda x:x['begin_ns'])):
                    ordinals.setdefault((q,ordinal,len(events)),[]).append(dict(gpu=(e['gpu_end_ns']-e['gpu_begin_ns'])/1e6,life=(e['end_ns']-e['begin_ns'])/1e6))
        result[arm]=dict(evaluate_api_median_ms=statistics.median(api) if api else None,
            queues={str(q):dict(command_buffers=len(v),median_gpu_cb_elapsed_ms=statistics.median(x['gpu'] for x in v),median_cb_lifetime_ms=statistics.median(x['life'] for x in v)) for q,v in queues.items()},
            queue_ordinals={f'{q}:{ordinal}/{count}':dict(samples=len(v),median_gpu_cb_elapsed_ms=statistics.median(x['gpu'] for x in v),median_cb_lifetime_ms=statistics.median(x['life'] for x in v)) for (q,ordinal,count),v in ordinals.items()})
    return dict(arms=result,scope='CB boundaries/lifetimes and async evaluate API spans; never sum or equate with physical ANE compute duration')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('native-library','native-source-archive','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cohort',nargs=2,metavar=('ROOT','PROBE_LIBRARY'),action='append',required=True)
    p.add_argument('--tool-source-archive',type=Path,action='append',default=[]);p.add_argument('--model',type=Path,default=ROOT/'models/Comfy-Org-Qwen-Image-2.1');args=p.parse_args()
    if args.output.exists():p.error('fresh independent actual-input audit required')
    manifest=json.loads((args.native_library.parent/'runtime-build/runtime-build-manifest.json').read_text());joint_audit.archive_contract(args.native_source_archive,manifest)
    weights,payloads=source.original(args.model);identity=model_snapshot(args.model);historical={};records=[];traces=[];anchors={};b_oracles={}
    for path in args.tool_source_archive:
        with tarfile.open(path,'r:gz') as arc:
            for m in arc.getmembers():
                if m.isfile():historical.setdefault(m.name,set()).add(arc.extractfile(m).read())
    for root,probe in args.cohort:
        root,probe=Path(root),Path(probe);s=json.loads((root/'summary.json').read_text());r=json.loads((root/'trial/report.json').read_text());single=r.get('single_only',False);prepack=r.get('compare_prepack',False);active=screen.arms(single,prepack)
        if (s['status']!='complete' or s['assets']['library']!=sha256_file(probe) or s['assets']['native_library']!=sha256_file(args.native_library) or
            r['compiled_native_id']!=manifest['runtime_build_id'] or r['loaded_runtime_id']!=manifest['runtime_build_id'] or s['model_snapshot']!=identity or
            r['source_arrays_sha256']!=payloads or s.get('single_only',False)!=single or s.get('compare_prepack',False)!=prepack):raise ValueError('actual native/helper/source/candidate owners differ')
        snapshot=json.loads((root/'source.json').read_text())
        if set(snapshot)!=set(s['source_sha256']):raise ValueError('complete tool snapshot required')
        for name,digest in s['source_sha256'].items():
            raw=snapshot[name].encode()
            if hashlib.sha256(raw).hexdigest()!=digest or not(raw==(ROOT/name).read_bytes() or raw in historical.get(name,set())):raise ValueError('actual current/archive source differs')
        if r['cpp_sha256']!=s['source_sha256']['tools/profiling/qwen_base/fp16_qkv_probe.mm'] or r['header_sha256']!=s['source_sha256']['tools/profiling/qwen_base/fp16_qkv.hpp']:
            raise ValueError('actual compiled helper/header source differs')
        capture=Path(s['capture']['root']);bound=screen.capture_contract(capture,s['case'])
        if bound!=s['capture']:raise ValueError('frozen actual source request/hash differs')
        for l in (0,1):
            for j,leaf in enumerate(('q','k','v')):
                if bound['source_weight_sha256'][f'block{l}-{j}']!=payloads[f'transformer_blocks.{l}.attn.to_{leaf}.weight']:raise ValueError('frozen source weights differ from current original BF16')
        arrays_path=root/'trial/arrays.safetensors';old=capture/'trial/arrays.safetensors';original=tensor_bits(old,'normalized',(1,1024,4096));actual=tensor_bits(arrays_path,'input_a',(1,1024,4096))
        if not np.array_equal(original,actual) or hashlib.sha256(actual.tobytes()).hexdigest()!=r['input_sha256']:raise ValueError('actual normalized source bytes were changed or synthetic')
        analysis=screen.validate(r,s['repeats'])
        if analysis!=s['analysis'] or sha256_file(arrays_path)!=s['arrays_sha256']:raise ValueError('actual complete timing/array payload differs')
        specifications={n:('BF16',(1,1024,4096 if n.startswith('input_') else 12288)) for n in {'input_a','input_b'}|{a+'_'+v for a in active for v in ('a','b','w')}}
        values=read_components(arrays_path,specifications);checked=compare(values,active)
        if not np.array_equal(values['input_b'],perturbed_input(values['input_a'])):raise ValueError('full typed BF16 normalized-input B recipe differs')
        for variant,old_name in (('a','raw_reference'),('w','other_weight_reference')):
            frozen=tensor_bits(old,old_name,(1,1024,12288));new=tensor_bits(arrays_path,'gpu_'+variant,(1,1024,12288))
            if not np.array_equal(frozen,new):raise ValueError('ordinary GPU projection changed from frozen original controls')
        if s['case'] not in b_oracles:b_oracles[s['case']]=cpu_projection_oracle(args.model,values['input_b'],values['gpu_b'])
        oracle={v:witness(weights[1 if v=='w' else 0],values['input_b' if v=='b' else 'input_a'],values['gpu_'+v]) for v in ('a','b','w')}
        if s['case'] in anchors:
            shared=set(checked['full_array_sha256'])&set(anchors[s['case']])
            if any(checked['full_array_sha256'][n]!=anchors[s['case']][n] for n in shared):raise ValueError('same-case candidate replay/residency control changed complete bytes')
        else:anchors[s['case']]=checked['full_array_sha256']
        observed=timeline=None
        if s['mode']=='timeline':
            raw=json.loads((root/'trial/observer.json').read_text())
            if raw['source_sha256']!=s['source_sha256']['tools/profiling/qwen_base/observer.mm']:raise ValueError('actual compiled observer differs')
            observed=screen.observer_contract(raw,s['repeats'],single,prepack)
            if observed!=s['observer']:raise ValueError('actual fixed-program predictor counts differ')
            timeline=timeline_summary(raw,r,active);traces.append((root.name,trace(raw)))
        records.append(dict(cohort=root.name,probe_sha256=sha256_file(probe),case=s['case'],mode=s['mode'],single_only=single,compare_prepack=prepack,analysis=analysis,arrays=checked,
            frozen_original_gpu_bytes_exact=['a','w'],b_perturbation_domain='normalized-input-not-historical-hidden',full_cpu_b_projection_oracle=b_oracles[s['case']],
            sampled_f64_gpu_witnesses=oracle,observer=observed,timeline=timeline,memory=memory_contract(root,s),load_error=s.get('load_error'),
            summary_sha256=sha256_file(root/'summary.json')))
    result=dict(schema='tc-private-fp16-qkv-audit-v1',status='complete',qualification_passed=False,native_changed=False,runtime_build_id=manifest['runtime_build_id'],
        native_sha256=sha256_file(args.native_library),native_source_archive_sha256=sha256_file(args.native_source_archive),script_sha256=sha256_file(Path(__file__)),
        cohorts=records,original_payloads=payloads,scope='frozen actual base40 gen/edit first-decode QKV source; full projection numeric/performance/resource/candidate-count study, not new images or physical precision/overlap qualification')
    args.output.mkdir(parents=True);write_json(args.output/'audit.json',result)
    for name,events in traces:write_json(args.output/(name+'-trace.json'),events)
    print(json.dumps(dict(status='complete',cohorts=len(records)),indent=2))


if __name__=='__main__':main()
