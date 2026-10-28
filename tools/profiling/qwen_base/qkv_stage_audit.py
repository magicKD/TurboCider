#!/usr/bin/env python3
"""Independent full QKV code/scales, source, CPU oracle and CB audit."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import tarfile

import numpy as np
import qkv_stage_collect as screen
import joint_audit
from collect import ROOT,model_snapshot,sha256_file,write_json
from gpu_dual_request_audit import memory_contract
from vpipe_audit import trace


def signs(seed=20260930):
    # Same versioned splitmix64 specification, independently evaluated here.
    result=[];mask=(1<<64)-1
    for i in range(128):
        x=(seed+i*0x9e3779b97f4a7c15)&mask;x=((x^(x>>30))*0xbf58476d1ce4e5b9)&mask;x=((x^(x>>27))*0x94d049bb133111eb)&mask
        result.append(-1 if (x^(x>>31))&1 else 1)
    return np.array(result,dtype=np.float32)


def quantize(bits,rotation_signs):
    if bits.dtype!=np.dtype('uint16') or bits.ndim!=2 or bits.shape[1]%128:raise ValueError("BF16 complete H128 rows required")
    x=(bits.astype(np.uint32)<<16).view(np.float32).copy();x=x.reshape(x.shape[0],-1,128)
    x*=rotation_signs
    for span in (1,2,4,8,16,32,64):
        y=x.reshape(x.shape[0],-1,2*span);a=y[...,:span].copy();b=y[...,span:].copy();y[...,:span]=a+b;y[...,span:]=a-b
    x*=np.float32(1)/np.sqrt(np.float32(128));x=x.reshape(bits.shape)
    peak=np.max(np.abs(x),axis=1);scale=np.where(peak==0,np.float32(128),np.maximum((peak/np.float32(127))*np.float32(128),np.float32(2**-24))).astype('<f2')
    q=np.clip((x/scale.astype(np.float32)[:,None])*np.float32(128),-127,127)
    return scale.view('<u2'),np.rint(q).astype('int8'),q


def oracle_check(bits,scale,code,rotation_signs):
    expected_scale,expected,q=quantize(bits,rotation_signs)
    if not np.array_equal(expected_scale,scale):raise ValueError("complete independent CPU row-scale bits differ")
    changed=expected!=code;count=int(np.count_nonzero(changed));delta=np.abs(expected.astype(np.int16)-code.astype(np.int16))
    # Metal/CPU division may straddle a nearest-even half-integer. Never
    # permit a distant code mismatch or claim CPU bit-exact if any exist.
    distance=np.abs(np.abs(q[changed]-np.floor(q[changed]))-.5)
    if count:
        raise ValueError("complete CPU quantization code bits differ; near-half differences are not silently accepted")
    return dict(codes=int(code.size),scale_bits_exact=True,code_bits_exact=count==0,near_half_code_changes=count,
        max_half_distance=float(distance.max()) if count else 0.,scope="independent complete CPU H128/FP32/RNE codes and scale bits exact on actual original W; not physical INT8 proof")


def original(model):
    path=model/'diffusion_models/qwen_image_2.1_bf16.safetensors'
    with path.open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0]
        if not 2<=size<=16<<20:raise ValueError("bounded original checkpoint header required")
        header=json.loads(f.read(size))
    result={};hashes={}
    for layer in (0,1):
        values=[]
        for leaf in ('q','k','v'):
            name=f'transformer_blocks.{layer}.attn.to_{leaf}.weight';info=header[name];begin,end=info['data_offsets']
            if info['dtype']!='BF16' or info['shape']!=[4096,4096] or end-begin!=4096*4096*2:raise ValueError("original full dense BF16 QKV payload required")
            bits=np.memmap(path,mode='r',dtype='<u2',offset=8+size+begin,shape=(4096,4096));hashes[name]=hashlib.sha256(bits.tobytes()).hexdigest();values.append(bits)
        result[layer]=values
    return result,hashes


def compare(values):
    if set(values)!={(l,a) for l in (0,1) for a in screen.ARMS}:raise ValueError("all code and row-scale producers required")
    for l in (0,1):
        anchor=values[l,screen.ARMS[0]]
        for a in screen.ARMS:
            if any(not np.array_equal(x,y) for x,y in zip(anchor,values[l,a])):raise ValueError("complete batch/fused codes or scales changed native bits")
    if np.array_equal(values[0,screen.ARMS[0]][0],values[1,screen.ARMS[0]][0]):raise ValueError("different source W returned stale full codes")
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('native-library','probe-library','native-source-archive','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cohort',type=Path,action='append',required=True);p.add_argument('--model',type=Path,default=ROOT/'models/Comfy-Org-Qwen-Image-2.1')
    p.add_argument('--cohort-library',nargs=2,action='append',default=[]);p.add_argument('--tool-source-archive',type=Path,action='append',default=[]);args=p.parse_args()
    if args.output.exists():p.error('fresh independent audit required')
    manifest=json.loads((args.native_library.parent/'runtime-build/runtime-build-manifest.json').read_text());joint_audit.archive_contract(args.native_source_archive,manifest)
    historical={};libraries={Path(root):Path(lib) for root,lib in args.cohort_library}
    for path in args.tool_source_archive:
        with tarfile.open(path,'r:gz') as arc:
            for m in arc.getmembers():
                if m.isfile():historical.setdefault(m.name,set()).add(arc.extractfile(m).read())
    weights,payloads=original(args.model);identity=model_snapshot(args.model);rotation_signs=signs();records=[];traces=[];oracles={}
    for root in args.cohort:
        s=json.loads((root/'summary.json').read_text());r=json.loads((root/'trial/report.json').read_text())
        probe=libraries.get(root,args.probe_library)
        if (s['status']!='complete' or s['assets']['library']!=sha256_file(probe) or s['assets']['native_library']!=sha256_file(args.native_library) or
            r['compiled_native_id']!=manifest['runtime_build_id'] or r['loaded_runtime_id']!=manifest['runtime_build_id'] or s['model_snapshot']!=identity or
            r['source_arrays_sha256']!=payloads):raise ValueError('actual complete source/native/helper identity differs')
        snapshot=json.loads((root/'source.json').read_text())
        if set(snapshot)!=set(s['source_sha256']):raise ValueError('complete source snapshot required')
        for name,digest in s['source_sha256'].items():
            raw=snapshot[name].encode()
            if hashlib.sha256(raw).hexdigest()!=digest or not(raw==(ROOT/name).read_bytes() or raw in historical.get(name,set())):raise ValueError('actual current/archive tool source differs')
        if r['cpp_sha256']!=s['source_sha256']['tools/profiling/qwen_base/qkv_stage_probe.mm'] or r['header_sha256']!=s['source_sha256']['tools/profiling/qwen_base/qkv_stage_batch.hpp']:
            raise ValueError('actual compiled helper/header source differs')
        analysis=screen.validate(r,s['repeats'])
        if analysis!=s['analysis']:raise ValueError('complete producer wall analysis differs')
        values={};digests={}
        for row in r['arrays']:
            l,a=row['layer'],row['arm'];code=root/'trial'/row['codes_file'];scale=root/'trial'/row['scales_file']
            if (code.stat().st_size!=12288*4096 or scale.stat().st_size!=12288*2 or sha256_file(code)!=row['codes_sha256'] or sha256_file(scale)!=row['scales_sha256']):raise ValueError('complete raw I8/FP16 payload differs')
            values[l,a]=(np.memmap(code,mode='r',dtype='int8',shape=(12288,4096)),np.memmap(scale,mode='r',dtype='<u2',shape=(12288,)))
            digests[f'{l}-{a}']=(row['codes_sha256'],row['scales_sha256'])
        compare(values)
        if not oracles:
            for l in (0,1):
                code,scale=values[l,screen.ARMS[0]]
                for j in range(3):oracles[f'{l}-{j}']=oracle_check(weights[l][j],scale[j*4096:(j+1)*4096],code[j*4096:(j+1)*4096],rotation_signs)
        if records and digests!=records[0]['raw_payload_sha256']:raise ValueError('repeat/timeline full producer bytes changed')
        observed=None
        if s['mode']=='timeline':
            raw=json.loads((root/'trial/observer.json').read_text())
            if raw['source_sha256']!=s['source_sha256']['tools/profiling/qwen_base/observer.mm']:raise ValueError('compiled observer differs')
            observed=screen.observer_contract(raw,s['repeats'])
            if observed!=s['observer']:raise ValueError('actual GPU CB evidence differs')
            traces.append((root.name,trace(raw)))
        records.append(dict(cohort=root.name,probe_sha256=sha256_file(probe),mode=s['mode'],analysis=analysis,raw_payload_sha256=digests,all_stage_bytes_exact=True,
            memory=memory_contract(root,s),observer=observed,load_error=s.get('load_error'),summary_sha256=sha256_file(root/'summary.json')))
    result=dict(schema='tc-qkv-w-stage-audit-v1',status='complete',qualification_passed=False,native_changed=False,cohorts=records,
        original_payloads=payloads,full_cpu_oracles=oracles,runtime_build_id=manifest['runtime_build_id'],probe_sha256=sha256_file(args.probe_library),
        native_sha256=sha256_file(args.native_library),native_source_archive_sha256=sha256_file(args.native_source_archive),script_sha256=sha256_file(Path(__file__)),
        scope='W producers alone, cached compact scales preprimed; full raw codes/scales and CPU oracle, no ANE placement/physical overlap/full40 speed claim')
    args.output.mkdir(parents=True);write_json(args.output/'audit.json',result)
    for name,events in traces:write_json(args.output/(name+'-trace.json'),events)
    print(json.dumps(dict(status='complete',cohorts=len(records)),indent=2))


if __name__=='__main__':main()
