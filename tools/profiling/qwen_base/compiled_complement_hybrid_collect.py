#!/usr/bin/env python3
"""Same fixed ANE owner: raw vs compiled GPU complement, complete FFN only."""
import argparse
from collections import Counter
import ctypes as C
import json
import math
import os
from pathlib import Path
import shutil
import sys

import analyze
import collect
import compiled_complement_collect as gpu
from collect import ROOT,Observer,model_snapshot,sha256_file,write_json
from runtime_ane_common import benchmark_environment
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

SOURCES=tuple(n for n in gpu.SOURCES if not n.endswith(('compiled_complement_probe.mm','build_qwen_compiled_complement_probe.sh')))+(
    'tools/profiling/qwen_base/compiled_complement_hybrid_probe.mm','tools/profiling/qwen_base/compiled_complement_hybrid_collect.py',
    'tools/native/build_qwen_compiled_complement_hybrid_probe.sh')


def validate(r,n):
    if r.get('schema')!='tc-compiled-complement-hybrid-v1' or r.get('complete_dit_block') is not False:raise ValueError('complete hybrid FFN is not a whole DiT block')
    checks=r.get('checks') or {}
    if set(checks)!={'a','b','w'}:raise ValueError('all real hybrid controls required')
    for c in checks.values():
        q=c['gpu_quality']
        if q.get('elements')!=4194304 or any(type(q.get(k)) not in (int,float) or not math.isfinite(q[k]) or q[k]<0 for k in ('relative_l2','max_abs')) or q['relative_l2']>=.03:
            raise ValueError('full GPU/ANE FFN approximation exceeds fixed screen')
        if c.get('bits_exact') is not True or c['incremental'].get('elements')!=4194304 or any(c['incremental'].get(k)!=0 for k in ('relative_l2','max_abs')):raise ValueError('GPU compile must not change full hybrid FFN output bits')
    # Reuse the exact schedule/geometry/trace contract without pretending this
    # report measured the GPU-only stage or exported hidden/partial arrays.
    adapted={**r,'schema':'tc-compiled-complement-v1','ane_predictions':0,'explicit_body_native_bits_exact':True,'rejected':7,
        'checks':{v:dict(partial=dict(relative_l2=0.,max_abs=0.,elements=4194304),hidden=dict(relative_l2=0.,max_abs=0.,elements=6291456),partial_bits_exact=True,hidden_bits_exact=True) for v in checks}}
    result=gpu.validate(adapted,n)
    for s in r['samples']:
        d=s.get('completion_deltas') or {}
        if set(d)!={'prediction_wait_seconds','output_wait_seconds','caller_wait_seconds'} or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in d.values()):
            raise ValueError('actual complete FFN wait deltas required')
    m=r['metrics'];w=m['runtime_weight'];calls=2*n+12
    if (m.get('runtime_failed') is not False or m.get('runtime_calls_session_total')!=calls or w.get('hybrid_blocks_session_total')!=calls or
        w.get('ane_channels')!=6144 or w.get('gpu_channels')!=6144 or w.get('partition_axis')!='intermediate_channels' or w.get('executor_backend')!='private_ane' or
        w.get('data_path')!='w8a8_hadamard' or w.get('fp32_channel_join_enabled') is not True or w.get('prefetch_enabled') is not False or w['weight_code_cache']['enabled'] is not False):
        raise ValueError('one healthy actual fixed ANE owner and no hidden fallback required')
    collect.validate_fixed_program([dict(hybrid=m)]);collect.validate_activation_receipt([dict(hybrid=m)],'sigmoid-complement')
    result.update(native_workload_predictions=calls,scope='complete staged GPU/ANE FFN incl weight preparation, execution, output join; ready frozen input, no attention/QKV or complete image')
    return result


def observer_contract(raw,r):
    rows=analyze.observer_evidence(raw);hot=[x for x in rows if x['request']>=1]
    calls=[x for x in hot if x['name']=='ane.evaluate_api'];expected=set(range(1,7))|{10,11}|set(range(100,100+len(r['samples'])))
    if (Counter(x['request'] for x in calls)!=Counter({n:1 for n in expected}) or len({x['model'] for x in calls})!=1 or
        len(calls)!=2*r['repeats']+12 or any(x['name'] in ('ane.compile_api','ane.load_api') for x in hot)):
        raise ValueError('one actual ANE model, hot compile/load0 and every complete workload prediction required')
    return dict(runtime_model_count=1,actual_workload_evaluations=len(calls),hot_compile_load_zero=True,
        scope='actual Private selectors, not physical INT8 MAC or hardware-overlap proof')


def child(args):
    env=dict(TURBOCIDER_ANE_BACKEND='private',TURBOCIDER_ALLOW_PRIVATE_ANE='1',TURBOCIDER_PRIVATE_ANE_DATA_PATH='w8a8',TURBOCIDER_PRIVATE_ANE_CHANNELS='6144',
        TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM='1',TURBOCIDER_PRIVATE_ANE_W8_PROJECTION='descriptor',TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION='sigmoid-complement',
        TURBOCIDER_PRIVATE_ANE_SCALE_CACHE='1',TURBOCIDER_PRIVATE_ANE_STAGE_SPECIALIZE='1',TURBOCIDER_PRIVATE_ANE_PREFETCH='0',TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE='1',
        TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC='1',TURBOCIDER_RUNTIME_ANE_CHUNKS='1',TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN='1',
        TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_BYTES='0',TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_MODE='copy')
    os.environ.update(env);observer=Observer(args.observer) if args.mode=='timeline' else None
    if observer:observer.lib.tc_qwen_profile_scope(-1)
    C.CDLL(str(args.native_library),mode=C.RTLD_GLOBAL);lib=C.CDLL(str(args.library));scope_type=C.CFUNCTYPE(None,C.c_int)
    scope=scope_type(lambda i:observer.lib.tc_qwen_profile_scope(i)) if observer else scope_type()
    fn=lib.tc_compiled_complement_hybrid_probe;fn.argtypes=[C.c_char_p]*4+[C.c_int,scope_type];fn.restype=C.c_int
    try:
        if fn(str(args.model).encode(),str(args.capture/'trial/gpu-complement-arrays.safetensors').encode(),str(args.manifest).encode(),str(args.output).encode(),args.repeats,scope):
            raise RuntimeError('complete hybrid FFN fixture failed; inspect report/stderr')
    finally:
        if observer:write_json(args.output/'observer.json',observer.take('tc_qwen_profile_finish'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('library','native-library','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--case',choices=('generate','edit'),default='generate');p.add_argument('--mode',choices=('stock','timeline'),default='stock')
    p.add_argument('--repeats',type=int,choices=(16,32),default=16);p.add_argument('--model',type=Path,default=ROOT/'models/Comfy-Org-Qwen-Image-2.1')
    p.add_argument('--observer',type=Path,default=ROOT/'build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib');p.add_argument('--capture',type=Path)
    p.add_argument('--manifest',type=Path,default=ROOT/'outputs/qwen-base-runtime-c1024-h4096-f12288-20261018.json');p.add_argument('--child',action='store_true');args=p.parse_args()
    if args.capture is None:args.capture=ROOT/('outputs/qwen-dual-gup-register-c6144-stock-v1-20261019' if args.case=='generate' else 'outputs/qwen-dual-gup-register-c6144-edit-v3-20261019')
    for k in ('library','native_library','model','observer','capture','manifest'):setattr(args,k,getattr(args,k).resolve(strict=True))
    args.output=args.output.resolve()
    if args.child:child(args);return
    native=json.loads((args.native_library.parent/'runtime-build/runtime-build-manifest.json').read_text())
    def sources():
        if any(sha256_file(ROOT/x['path'])!=x['sha256'] for x in native['inputs']['sources']):raise ValueError('native sources differ from bound ABI')
    sources()
    if shutil.disk_usage(ROOT).free<(2<<30)+(104<<20):raise ValueError('2GiB reserve plus bounded104MiB output required')
    old=json.loads((args.capture/'summary.json').read_text());cap=args.capture/'trial/gpu-complement-arrays.safetensors'
    if old['status']!='complete' or old['case']!=args.case or sha256_file(cap)!=old['arrays_sha256']['gpu-complement-arrays.safetensors']:raise ValueError('complete frozen original base40 capture required')
    capture=dict(root=str(args.capture),summary_sha256=sha256_file(args.capture/'summary.json'),arrays_sha256=sha256_file(cap),scope='data-only actual frozen first-decode input; no historical timing denominator')
    hashes={x:sha256_file(ROOT/x) for x in SOURCES};assets={k:sha256_file(getattr(args,k)) for k in ('library','native_library','observer','manifest')};identity=model_snapshot(args.model)
    args.output.mkdir(parents=True,exist_ok=False);trial=args.output/'trial';trial.mkdir();s=dict(schema='tc-compiled-complement-hybrid-profile-v1',status='running',qualification_passed=False,
        case=args.case,mode=args.mode,repeats=args.repeats,capture=capture,source_sha256=hashes,assets=assets,model_snapshot=identity,runtime_build_id=native['runtime_build_id'])
    write_json(args.output/'summary.json',s);write_json(args.output/'source.json',{x:(ROOT/x).read_text() for x in SOURCES});load=LoadObservation(args.output/'load.jsonl')
    try:
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--child','--library',str(args.library),'--native-library',str(args.native_library),'--case',args.case,
            '--mode',args.mode,'--repeats',str(args.repeats),'--model',str(args.model),'--observer',str(args.observer),'--manifest',str(args.manifest),'--capture',str(args.capture),'--output',str(trial)]
        env={k:v for k,v in benchmark_environment().items() if not k.startswith('TURBOCIDER_') and k!='MLX_DISABLE_COMPILE'}
        with (args.output/'stdout.log').open('x') as stdout,(args.output/'stderr.log').open('x') as stderr:
            s['memory']=run_sampled(command,repo=ROOT,output=args.output,stem='trial',env=env,stdout=stdout,stderr=stderr,timeout=300,interval_ms=100,max_gap_ms=500,observer=load)
        r=json.loads((trial/'report.json').read_text())
        if (r['compiled_native_id']!=native['runtime_build_id'] or r['loaded_runtime_id']!=native['runtime_build_id'] or
            r['cpp_sha256']!=hashes['tools/profiling/qwen_base/compiled_complement_hybrid_probe.mm'] or r['header_sha256']!=hashes['tools/profiling/qwen_base/compiled_complement.hpp']):raise ValueError('actual compiled helper/native identity differs')
        s['analysis']=validate(r,args.repeats);s['arrays_sha256']=sha256_file(trial/'arrays.safetensors')
        if args.mode=='timeline':
            raw=json.loads((trial/'observer.json').read_text())
            if raw['source_sha256']!=hashes['tools/profiling/qwen_base/observer.mm']:raise ValueError('actual observer source differs')
            s['observer']=observer_contract(raw,r)
        sources();s['status']='complete'
    except Exception as e:s.update(status='failed',error=str(e))
    try:s['load']=load.verify()
    except ValueError as e:s['load_error']=str(e)
    if any(sha256_file(ROOT/x)!=v for x,v in hashes.items()) or any(sha256_file(getattr(args,k))!=assets[k] for k in assets) or model_snapshot(args.model)!=identity:s.update(status='failed',error='source/assets/model changed')
    write_json(args.output/'summary.json',s);print(json.dumps(s,indent=2))
    if s['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
