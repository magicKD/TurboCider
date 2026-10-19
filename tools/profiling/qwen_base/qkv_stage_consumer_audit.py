#!/usr/bin/env python3
"""Independent complete QKV consumer arrays/source/fixed-program audit."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np

import qkv_stage_consumer_collect as screen
import qkv_stage_audit as weights
import joint_audit
from collect import ROOT,model_snapshot,sha256_file,write_json
from down_audit import read_components
from gpu_dual_request_audit import memory_contract
from qkv_audit import quality
from vpipe_audit import trace


def compare(values):
    names={'input_a','input_b'}|{'gpu_'+v for v in ('a','b','w')}|{a+'_'+v for a in screen.ARMS for v in ('a','b','w')}
    if set(values)!=names or any(not np.isfinite(v).all() for v in values.values()):raise ValueError('all seventeen complete finite BF16 inputs/projections required')
    hashes={n:hashlib.sha256(v.astype('<f4',copy=False).tobytes()).hexdigest() for n,v in values.items()}
    checks={}
    for v in ('a','b','w'):
        anchor=screen.ARMS[0]+'_'+v
        if any(hashes[a+'_'+v]!=hashes[anchor] for a in screen.ARMS):raise ValueError('batch W producer changed complete raw QKV bits')
        q=quality(values['gpu_'+v],values[anchor]);checks[v]=q
        if not 0<=q['relative_l2']<.02:raise ValueError('raw QKV exceeds predeclared original GPU component bound')
    if hashes['input_a']==hashes['input_b'] or hashes[screen.ARMS[0]+'_a'] in (hashes[screen.ARMS[0]+'_b'],hashes[screen.ARMS[0]+'_w']):
        raise ValueError('changed W/input returned stale full projection')
    return dict(full_array_sha256=hashes,incremental_bits_exact=True,quality_vs_gpu=checks,
        scope='complete synthetic-input raw QKV and actual original W, not natural FFN/attention/full40 image or independent full CPU GEMM')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('native-library','probe-library','native-source-archive','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cohort',type=Path,action='append',required=True);p.add_argument('--model',type=Path,default=ROOT/'models/Comfy-Org-Qwen-Image-2.1')
    p.add_argument('--cohort-library',nargs=2,action='append',default=[]);p.add_argument('--tool-source-archive',type=Path,action='append',default=[]);args=p.parse_args()
    if args.output.exists():p.error('fresh consumer audit required')
    manifest=json.loads((args.native_library.parent/'runtime-build/runtime-build-manifest.json').read_text());joint_audit.archive_contract(args.native_source_archive,manifest)
    historical={};libraries={Path(root):Path(lib) for root,lib in args.cohort_library}
    for path in args.tool_source_archive:
        with tarfile.open(path,'r:gz') as arc:
            for m in arc.getmembers():
                if m.isfile():historical.setdefault(m.name,set()).add(arc.extractfile(m).read())
    _,payloads=weights.original(args.model);identity=model_snapshot(args.model);records=[];traces=[]
    for root in args.cohort:
        s=json.loads((root/'summary.json').read_text());r=json.loads((root/'trial/report.json').read_text())
        probe=libraries.get(root,args.probe_library)
        if (s['status']!='complete' or s['assets']['library']!=sha256_file(probe) or s['assets']['native_library']!=sha256_file(args.native_library) or
            r['compiled_native_id']!=manifest['runtime_build_id'] or r['loaded_runtime_id']!=manifest['runtime_build_id'] or s['model_snapshot']!=identity or r['source_arrays_sha256']!=payloads):
            raise ValueError('actual bound helper/native/original payload differs')
        snapshot=json.loads((root/'source.json').read_text())
        if set(snapshot)!=set(s['source_sha256']):raise ValueError('complete actual source snapshot required')
        for name,digest in s['source_sha256'].items():
            raw=snapshot[name].encode()
            if hashlib.sha256(raw).hexdigest()!=digest or not(raw==(ROOT/name).read_bytes() or raw in historical.get(name,set())):raise ValueError('actual current/archive producer/consumer tool source differs')
        if r['cpp_sha256']!=s['source_sha256']['tools/profiling/qwen_base/qkv_stage_consumer_probe.mm'] or r['header_sha256']!=s['source_sha256']['tools/profiling/qwen_base/qkv_stage_batch.hpp']:
            raise ValueError('actual compiled consumer/header source differs')
        analysis=screen.validate(r,s['repeats'])
        if analysis!=s['analysis'] or sha256_file(root/'trial/arrays.safetensors')!=s['arrays_sha256']:raise ValueError('completed timing/array payload changed')
        specifications={n:('BF16',(1,1024,4096 if n.startswith('input_') else 12288)) for n in
            {'input_a','input_b'}|{'gpu_'+v for v in ('a','b','w')}|{a+'_'+v for a in screen.ARMS for v in ('a','b','w')}}
        arrays=compare(read_components(root/'trial/arrays.safetensors',specifications))
        if records and arrays['full_array_sha256']!=records[0]['arrays']['full_array_sha256']:raise ValueError('repeat/timeline complete projection bytes changed')
        observed=None
        if s['mode']=='timeline':
            raw=json.loads((root/'trial/observer.json').read_text())
            if raw['source_sha256']!=s['source_sha256']['tools/profiling/qwen_base/observer.mm']:raise ValueError('compiled observer differs')
            observed=screen.observer_contract(raw,s['repeats'])
            if observed!=s['observer']:raise ValueError('actual single fixed-model selector contract differs')
            traces.append((root.name,trace(raw)))
        records.append(dict(cohort=root.name,probe_sha256=sha256_file(probe),mode=s['mode'],analysis=analysis,arrays=arrays,observer=observed,memory=memory_contract(root,s),
            load_error=s.get('load_error'),summary_sha256=sha256_file(root/'summary.json')))
    result=dict(schema='tc-qkv-stage-consumer-audit-v1',status='complete',qualification_passed=False,native_changed=False,cohorts=records,original_payloads=payloads,
        runtime_build_id=manifest['runtime_build_id'],probe_sha256=sha256_file(args.probe_library),native_sha256=sha256_file(args.native_library),
        native_source_archive_sha256=sha256_file(args.native_source_archive),script_sha256=sha256_file(Path(__file__)),
        scope='fixed QKV graph four W producers, real W synthetic target1024; complete consumer wall/array/selector/memory audit, not complete40 image speed')
    args.output.mkdir(parents=True);write_json(args.output/'audit.json',result)
    for name,events in traces:write_json(args.output/(name+'-trace.json'),events)
    print(json.dumps(dict(status='complete',cohorts=len(records)),indent=2))


if __name__=='__main__':main()
