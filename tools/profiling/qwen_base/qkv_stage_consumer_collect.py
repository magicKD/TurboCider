#!/usr/bin/env python3
"""Full QKV projection consumer of four W producers on one fixed graph."""
import argparse
from collections import Counter
import ctypes as C
import json
import math
from pathlib import Path
import shutil
import sys
import statistics

import analyze
from qkv_stage_collect import ARMS
from collect import ROOT,Observer,model_snapshot,sha256_file,write_json
from runtime_ane_common import benchmark_environment
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

SOURCES=tuple("tools/profiling/qwen_base/"+x for x in ("qkv_stage_consumer_collect.py","qkv_stage_consumer_probe.mm","qkv_stage_batch.hpp","qkv_ane.hpp","qkv_stage_collect.py","collect.py","analyze.py","observer.mm"))+(
    "tools/native/build_qwen_qkv_stage_consumer_probe.sh","tools/native/process_tree_sampler.py","tools/native/verify_process_tree_samples.py",
    "tools/validation/runtime_ane_memory.py","tools/validation/runtime_ane_load.py")


def validate(r,n):
    if (type(n) is not int or n not in (32,64) or r.get('schema')!='tc-qkv-stage-consumer-v1' or r.get('status')!='complete' or
        r.get('complete_request') is not False or r.get('image_export') is not False or r.get('synthetic_inputs') is not True or
        r.get('repeats')!=n or r.get('warmups')!=2 or r.get('extra_owned_weight_bytes')!=0 or r.get('source_unchanged') is not True or r.get('incremental_bits_exact') is not True or
        not isinstance(r.get('program_key'),str) or len(r['program_key'])!=64):raise ValueError('actual completed synthetic-input original-W QKV fixture required')
    for key,want in dict(predictions=4*n+24,stage_calls=4*n+26,stage_command_buffers=6*n+42,stage_dispatches=16*n+114).items():
        if type(r.get(key)) is not int or r[key]!=want:raise ValueError('actual fixed projection and W producer counts differ: '+key)
    samples=r['samples'];order=[(ARMS[(j+rep+4)%4],rep) for rep in range(-2,n) for j in range(4)]
    if [(s['arm'],s['rep']) for s in samples]!=order:raise ValueError('four arms fully rotated with all warm/measured projections required')
    for s in samples:
        if (s.get('warmup') is not (s['rep']<0) or s.get('input_variant') is not bool(s['rep']&1) or
            type(s.get('seconds')) not in (int,float) or not math.isfinite(s['seconds']) or s['seconds']<=0 or
            any(type(s.get(k)) not in (int,float) or not math.isfinite(s[k]) or s[k]<0 for k in ('weight_stage_seconds','activation_seconds','predict_restore_join_seconds'))):
            raise ValueError('positive completed QKV consumer wall and finite subspans required')
    medians={a:statistics.median(s['seconds'] for s in samples if s['arm']==a and not s['warmup']) for a in ARMS}
    return dict(median_seconds=medians,reduction_vs_native={a:1-v/medians[ARMS[0]] for a,v in medians.items()},
        median_spans={a:{k:statistics.median(s[k] for s in samples if s['arm']==a and not s['warmup']) for k in ('weight_stage_seconds','activation_seconds','predict_restore_join_seconds')} for a in ARMS},
        measured_projections=4*n,workload_predictions=4*n+24,
        scope='same fixed full QKV projection with real W and synthetic1024 input, includes W preparation and restore; no FFN/lookahead/full40 speed claim')


def observer_contract(raw,n):
    rows=analyze.observer_evidence(raw);hot=[x for x in rows if x['request']>=1]
    evals=[x for x in hot if x['name']=='ane.evaluate_api'];scopes=set(range(1,17))|set(range(100,100+4*(n+2)))
    if Counter(x['request'] for x in evals)!=Counter({i:1 for i in scopes}) or len({x['model'] for x in evals})!=1:
        raise ValueError('one actual QKV model and one predict per fixture scope required')
    if any(x['name'] in ('ane.compile_api','ane.load_api') for x in hot):raise ValueError('changing producer/input/W compiled or loaded an ANE graph')
    return dict(runtime_models=1,workload_evaluations=len(evals),hot_compile_load_zero=True,scope='actual Private selectors and completed Metal records; not physical INT8/overlap')


def child(args):
    observer=Observer(args.observer) if args.mode=='timeline' else None
    if observer:observer.lib.tc_qwen_profile_scope(-1)
    C.CDLL(str(args.native_library),mode=C.RTLD_GLOBAL);lib=C.CDLL(str(args.library));scope_type=C.CFUNCTYPE(None,C.c_int)
    scope=scope_type(lambda i:observer.lib.tc_qwen_profile_scope(i)) if observer else scope_type()
    fn=lib.tc_qkv_stage_consumer_probe;fn.argtypes=[C.c_char_p,C.c_char_p,C.c_int,scope_type];fn.restype=C.c_int
    try:
        if fn(str(args.model).encode(),str(args.output).encode(),args.repeats,scope):raise RuntimeError('QKV consumer fixture failed; inspect report/stderr')
    finally:
        if observer:write_json(args.output/'observer.json',observer.take('tc_qwen_profile_finish'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('library','native-library','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--mode',choices=('stock','timeline'),default='stock');p.add_argument('--repeats',type=int,choices=(32,64),default=32)
    p.add_argument('--model',type=Path,default=ROOT/'models/Comfy-Org-Qwen-Image-2.1')
    p.add_argument('--observer',type=Path,default=ROOT/'build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib');p.add_argument('--child',action='store_true');args=p.parse_args()
    for k in ('library','native_library','model','observer'):setattr(args,k,getattr(args,k).resolve(strict=True))
    args.output=args.output.resolve()
    if args.child:child(args);return
    manifest=json.loads((args.native_library.parent/'runtime-build/runtime-build-manifest.json').read_text())
    def sources():
        if any(sha256_file(ROOT/e['path'])!=e['sha256'] for e in manifest['inputs']['sources']):raise ValueError('actual native source differs')
    sources()
    if shutil.disk_usage(ROOT).free<(2<<30)+(448<<20):raise ValueError('2GiB reserve plus448MiB complete projection arrays required')
    hashes={x:sha256_file(ROOT/x) for x in SOURCES};assets={k:sha256_file(getattr(args,k)) for k in ('library','native_library','observer')};identity=model_snapshot(args.model)
    args.output.mkdir(parents=True,exist_ok=False);trial=args.output/'trial';trial.mkdir();s=dict(schema='tc-qkv-stage-consumer-screen-v1',status='running',qualification_passed=False,
        mode=args.mode,repeats=args.repeats,source_sha256=hashes,assets=assets,model_snapshot=identity,runtime_build_id=manifest['runtime_build_id'])
    write_json(args.output/'summary.json',s);write_json(args.output/'source.json',{x:(ROOT/x).read_text() for x in SOURCES});load=LoadObservation(args.output/'load.jsonl')
    try:
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--child','--library',str(args.library),'--native-library',str(args.native_library),'--mode',args.mode,
            '--repeats',str(args.repeats),'--model',str(args.model),'--observer',str(args.observer),'--output',str(trial)]
        env={k:v for k,v in benchmark_environment().items() if not k.startswith('TURBOCIDER_') and k!='MLX_DISABLE_COMPILE'}
        with (args.output/'stdout.log').open('x') as stdout,(args.output/'stderr.log').open('x') as stderr:
            s['memory']=run_sampled(command,repo=ROOT,output=args.output,stem='trial',env=env,stdout=stdout,stderr=stderr,timeout=300,interval_ms=100,max_gap_ms=500,observer=load)
        r=json.loads((trial/'report.json').read_text())
        if (r['compiled_native_id']!=manifest['runtime_build_id'] or r['loaded_runtime_id']!=manifest['runtime_build_id'] or
            r['cpp_sha256']!=hashes['tools/profiling/qwen_base/qkv_stage_consumer_probe.mm'] or r['header_sha256']!=hashes['tools/profiling/qwen_base/qkv_stage_batch.hpp']):
            raise ValueError('actual compiled helper/header/native ABI differs')
        s['analysis']=validate(r,args.repeats);s['arrays_sha256']=sha256_file(trial/'arrays.safetensors')
        if args.mode=='timeline':
            raw=json.loads((trial/'observer.json').read_text())
            if raw['source_sha256']!=hashes['tools/profiling/qwen_base/observer.mm']:raise ValueError('compiled observer source differs')
            s['observer']=observer_contract(raw,args.repeats)
        sources();s['status']='complete'
    except Exception as e:s.update(status='failed',error=str(e))
    try:s['load']=load.verify()
    except ValueError as e:s['load_error']=str(e)
    if (model_snapshot(args.model)!=identity or any(sha256_file(ROOT/x)!=d for x,d in hashes.items()) or any(sha256_file(getattr(args,k))!=d for k,d in assets.items())):s.update(status='failed',error='source/model/assets changed')
    write_json(args.output/'summary.json',s);print(json.dumps(s,indent=2))
    if s['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
