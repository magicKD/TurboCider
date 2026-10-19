#!/usr/bin/env python3
"""FP16 hidden/original-BF16 W mixed F32 down qualification and profiling."""
import argparse
from collections import Counter
import ctypes as C
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import analyze
from collect import ROOT,Observer,model_snapshot,sha256_file,write_json
from runtime_ane_common import benchmark_environment
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ARMS=('original-bf16','rounded-bf16','mixed-ready','mixed-with-cast')
SOURCES=tuple('tools/profiling/qwen_base/'+n for n in ('mixed_hidden_collect.py','mixed_hidden_probe.mm','mixed_hidden.hpp','collect.py','analyze.py','observer.mm'))+(
    'tools/native/build_qwen_mixed_hidden_probe.sh','tools/native/process_tree_sampler.py','tools/native/verify_process_tree_samples.py','tools/validation/runtime_ane_memory.py','tools/validation/runtime_ane_load.py')


def validate(r,n,real):
    if (type(n) is not int or n not in (16,32) or type(real) is not bool or r.get('schema')!='tc-mixed-hidden-down-v1' or r.get('status')!='complete' or
        r.get('real_hidden') is not real or any(r.get(k) is not False for k in ('complete_request','image_export')) or r.get('ane_predictions')!=0 or
        r.get('extra_owned_weight_bytes')!=0 or r.get('repeats')!=n or r.get('warmups')!=2 or r.get('source_unchanged') is not True or r.get('rejected')!=6):
        raise ValueError('actual mixed down dtype/range/complete-output fixture required')
    shape=tuple(r.get(k) for k in ('m','k','n','physical_weight_pitch','row_begin','column_begin'))
    if shape!=((1024,6144,4096,12288,0,0) if real else (33,128,257,320,5,17)):raise ValueError('actual original physical pitch/offset/tail geometry differs')
    samples=r['samples'];order=[(ARMS[(j+rep+4)%4],rep) for rep in range(-2,n) for j in range(4)]
    if [(s['arm'],s['rep']) for s in samples]!=order:raise ValueError('all four rotated down controls and warmups required')
    for s in samples:
        if (s.get('warmup') is not (s['rep']<0) or s.get('input_variant') is not bool(s['rep']&1) or
            type(s.get('seconds')) not in (int,float) or not math.isfinite(s['seconds']) or s['seconds']<=0):raise ValueError('actual completed positive down wall required')
    for v in ('a','b'):
        for key in ('cast_quality','incremental_partial_quality','same_value_partial_quality'):
            q=r['checks'][v][key]
            if any(type(q.get(k)) not in (int,float) or not math.isfinite(q[k]) or q[k]<0 for k in ('relative_l2','max_abs')):raise ValueError('finite actual format/partial quality required')
        if r['checks'][v]['incremental_partial_quality']['relative_l2']>=1e-4 or r['checks'][v]['same_value_partial_quality']['relative_l2']>=1e-4:
            raise ValueError('mixed real/normal F32 down exceeds predeclared 1e-4 screening bound')
    if not real:
        d=r.get('format_domains') or {}
        if set(d)!={'underflow','subnormal','zero','wide'} or r.get('wide_format_rejected') is not True or d['wide'].get('rejected_before_down') is not True or d['wide'].get('finite') is not False:
            raise ValueError('explicit zero/subnormal/underflow/overflow domains must not be hidden')
        if d['underflow']['vs_original']['relative_l2']<.99:raise ValueError('underflow must be reported as loss, not zero-exact')
        if d['zero']['vs_original']['relative_l2']!=0 or d['zero']['vs_original']['max_abs']!=0:raise ValueError('true zero must remain exact')
    medians={a:statistics.median(s['seconds'] for s in samples if s['arm']==a and not s['warmup']) for a in ARMS}
    return dict(median_seconds=medians,reduction_vs_original={a:1-v/medians[ARMS[0]] for a,v in medians.items()},measured_down_projections=4*n,
        scope='ready captured hidden down only; mixed-with-cast includes per-call X cast, no W casts; not complete GPU complement/block/image speed')


def observer_contract(raw,r):
    rows=analyze.observer_evidence(raw)
    if any(x['name'].startswith('ane.') for x in rows):raise ValueError('GPU down-only screen cannot invoke an ANE model')
    counts=Counter(x['request'] for x in rows if x['name']=='metal.command_buffer')
    expected=set(range(100,100+len(r['samples'])))
    if not all(counts[n]>0 for n in expected):raise ValueError('each measured/warm down must have actual completed GPU work')
    return dict(ane_calls=0,completed_sample_scopes=len(expected),sample_gpu_cb_count=sum(counts[n] for n in expected),scope='actual GPU CB completion/clock evidence, not per-shader counters or physical precision')


def child(args):
    observer=Observer(args.observer) if args.mode=='timeline' else None
    if observer:observer.lib.tc_qwen_profile_scope(-1)
    C.CDLL(str(args.native_library),mode=C.RTLD_GLOBAL);lib=C.CDLL(str(args.library));scope_type=C.CFUNCTYPE(None,C.c_int)
    scope=scope_type(lambda i:observer.lib.tc_qwen_profile_scope(i)) if observer else scope_type()
    fn=lib.tc_mixed_hidden_probe;fn.argtypes=[C.c_char_p,C.c_char_p,C.c_char_p,C.c_int,scope_type];fn.restype=C.c_int
    try:
        if fn(str(args.model).encode(),str(args.capture/'trial/gpu-complement-arrays.safetensors').encode() if args.capture else b'-',str(args.output).encode(),args.repeats,scope):
            raise RuntimeError('mixed down GPU compilation/numeric/ownership fixture failed; inspect report/stderr')
    finally:
        if observer:write_json(args.output/'observer.json',observer.take('tc_qwen_profile_finish'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('library','native-library','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--fixture',choices=('synthetic','generate','edit'),default='synthetic');p.add_argument('--mode',choices=('stock','timeline'),default='stock')
    p.add_argument('--repeats',type=int,choices=(16,32),default=16);p.add_argument('--model',type=Path,default=ROOT/'models/Comfy-Org-Qwen-Image-2.1')
    p.add_argument('--observer',type=Path,default=ROOT/'build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib');p.add_argument('--capture',type=Path);p.add_argument('--child',action='store_true');args=p.parse_args()
    if args.fixture!='synthetic' and args.capture is None:args.capture=ROOT/('outputs/qwen-dual-gup-register-c6144-stock-v1-20261019' if args.fixture=='generate' else 'outputs/qwen-dual-gup-register-c6144-edit-v3-20261019')
    for k in ('library','native_library','model','observer'):setattr(args,k,getattr(args,k).resolve(strict=True))
    if args.capture:args.capture=args.capture.resolve(strict=True)
    args.output=args.output.resolve()
    if args.child:child(args);return
    native=json.loads((args.native_library.parent/'runtime-build/runtime-build-manifest.json').read_text())
    def sources():
        if any(sha256_file(ROOT/x['path'])!=x['sha256'] for x in native['inputs']['sources']):raise ValueError('native source differs from bound helper ABI')
    sources();real=args.capture is not None
    if shutil.disk_usage(ROOT).free<(2<<30)+((224 if real else 16)<<20):raise ValueError('2GiB reserve plus bounded output allowance required')
    capture=None
    if real:
        old=json.loads((args.capture/'summary.json').read_text());capture_file=args.capture/'trial/gpu-complement-arrays.safetensors'
        if old['status']!='complete' or old['case']!=args.fixture or sha256_file(capture_file)!=old['arrays_sha256']['gpu-complement-arrays.safetensors']:raise ValueError('complete original hidden capture required')
        capture=dict(root=str(args.capture),summary_sha256=sha256_file(args.capture/'summary.json'),arrays_sha256=sha256_file(capture_file),scope='data-only frozen first-decode actual base40 GPU complement; no old C++ helper execution/time denominator')
    hashes={x:sha256_file(ROOT/x) for x in SOURCES};assets={k:sha256_file(getattr(args,k)) for k in ('library','native_library','observer')}
    sdk=Path(subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip())/'System/Library/Frameworks/MetalPerformancePrimitives.framework/Versions/A/Headers/__impl/MPPTensorOpsMatMul2dImpl.h'
    assets['installed_mpp_header']=sha256_file(sdk);identity=model_snapshot(args.model) if real else None
    args.output.mkdir(parents=True,exist_ok=False);trial=args.output/'trial';trial.mkdir();s=dict(schema='tc-mixed-hidden-down-profile-v1',status='running',qualification_passed=False,
        fixture=args.fixture,mode=args.mode,repeats=args.repeats,capture=capture,source_sha256=hashes,assets=assets,installed_mpp_header=str(sdk),model_snapshot=identity,runtime_build_id=native['runtime_build_id'])
    write_json(args.output/'summary.json',s);write_json(args.output/'source.json',{x:(ROOT/x).read_text() for x in SOURCES});load=LoadObservation(args.output/'load.jsonl')
    try:
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--child','--library',str(args.library),'--native-library',str(args.native_library),'--fixture',args.fixture,
            '--mode',args.mode,'--repeats',str(args.repeats),'--model',str(args.model),'--observer',str(args.observer),'--output',str(trial)]
        if args.capture:command+=['--capture',str(args.capture)]
        env={k:v for k,v in benchmark_environment().items() if not k.startswith('TURBOCIDER_') and k!='MLX_DISABLE_COMPILE'}
        with (args.output/'stdout.log').open('x') as stdout,(args.output/'stderr.log').open('x') as stderr:
            s['memory']=run_sampled(command,repo=ROOT,output=args.output,stem='trial',env=env,stdout=stdout,stderr=stderr,timeout=300,interval_ms=100,max_gap_ms=500,observer=load)
        r=json.loads((trial/'report.json').read_text())
        if (r['compiled_native_id']!=native['runtime_build_id'] or r['loaded_runtime_id']!=native['runtime_build_id'] or r['cpp_sha256']!=hashes['tools/profiling/qwen_base/mixed_hidden_probe.mm'] or r['header_sha256']!=hashes['tools/profiling/qwen_base/mixed_hidden.hpp']):
            raise ValueError('actual compiled helper/header/native identity differs')
        s['analysis']=validate(r,args.repeats,real);s['arrays_sha256']=sha256_file(trial/'arrays.safetensors')
        if args.mode=='timeline':
            raw=json.loads((trial/'observer.json').read_text())
            if raw['source_sha256']!=hashes['tools/profiling/qwen_base/observer.mm']:raise ValueError('actual compiled observer source differs')
            s['observer']=observer_contract(raw,r)
        sources();s['status']='complete'
    except Exception as e:s.update(status='failed',error=str(e))
    try:s['load']=load.verify()
    except ValueError as e:s['load_error']=str(e)
    if (any(sha256_file(ROOT/x)!=v for x,v in hashes.items()) or any(sha256_file(getattr(args,k))!=assets[k] for k in ('library','native_library','observer')) or
        sha256_file(sdk)!=assets['installed_mpp_header'] or (real and model_snapshot(args.model)!=identity)):s.update(status='failed',error='source/assets/SDK/model changed')
    write_json(args.output/'summary.json',s);print(json.dumps(s,indent=2))
    if s['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
