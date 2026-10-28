#!/usr/bin/env python3
"""Real BF16 Qwen full FFN consumer; fixed vs multi-function, not image speed."""
import argparse
from collections import Counter
import ctypes as C
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys

import collect
from collect import ROOT, Observer, sha256_file, write_json, model_snapshot
from analyze import observer_evidence
from runtime_ane_common import benchmark_environment
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ARMS=('gpu','single1024','single2048','family')
ROWS=(1024,1536,2048,2560)
SOURCES=tuple('tools/profiling/qwen_base/'+x for x in ('w8_functions_real_collect.py','w8_functions_real_probe.mm','collect.py','analyze.py','observer.mm'))+('tools/native/build_ane_w8_functions_real_probe.sh',)


def predictions(arm, rows):
    return 0 if arm=='gpu' else (rows+1023)//1024 if arm=='single1024' else (rows+2047)//2048


def validate(r,repeats):
    if (r.get('schema')!='tc-real-w8-function-ffn-v1' or r.get('status')!='complete' or
        r.get('whole_ffn') is not True or r.get('whole_dit_block') is not False or r.get('image_export') is not False or
        (r.get('h'),r.get('f'),r.get('ane_channels'),r.get('repeats'),r.get('warmups'))!=(4096,12288,6144,repeats,2) or
        r.get('source_unchanged') is not True or r.get('restored_bits_exact') is not True):
        raise ValueError('actual complete source-bound real FFN consumer required')
    checks=r.get('checks') or []
    if [(x['rows'],x['state']) for x in checks]!=[(n,state) for n in ROWS for state in range(3 if n==1024 else 1)]:
        raise ValueError('all full-array source/shape quality states required')
    for c in checks:
        if len(c['qualities'])!=4:raise ValueError('all four full FFN controls required')
        for q in c['qualities']:
            if q.get('elements')!=c['rows']*4096 or any(type(q.get(k)) not in (int,float) or not math.isfinite(q[k]) or q[k]<0 for k in ('relative_l2','max_abs')) or q['relative_l2']>=.03:
                raise ValueError('complete finite FFN approximation exceeds declared budget')
        q=c['family_vs_single2048']
        if q.get('elements')!=c['rows']*4096 or not math.isfinite(q.get('relative_l2',float('nan'))) or not 0<=q['relative_l2']<.005:
            raise ValueError('family incremental error exceeds declared budget')
    if type(r.get('reverse')) is not bool:raise ValueError('actual fixed visit direction required')
    expected=[(n,ARMS[3-(j+rep+4)%4 if r['reverse'] else (j+rep+4)%4],rep) for n in ROWS for rep in range(-2,repeats) for j in range(4)]
    samples=r.get('samples') or []
    if [(x['rows'],x['arm'],x['rep']) for x in samples]!=expected:raise ValueError('balanced complete four-arm schedule differs')
    if [x.get('scope') for x in samples]!=list(range(100,100+len(samples))):raise ValueError('unique actual scope IDs required even without instrumentation')
    for x in samples:
        if (x['warmup'] is not (x['rep']<0) or x['input_variant']!=bool(x['rep']&1) or x['layer']!=((x['rep']+2)%4==2) or
            x['calls']!=predictions(x['arm'],x['rows']) or type(x['seconds']) not in (int,float) or not math.isfinite(x['seconds']) or x['seconds']<=0):
            raise ValueError('actual finite workload timing/call/source coverage differs')
    return dict(per_rows={str(n):{arm:statistics.median(x['seconds'] for x in samples if x['rows']==n and x['arm']==arm and not x['warmup']) for arm in ARMS} for n in ROWS},
                measured_consumers=16*repeats,scope='complete FFN including pad, W stage, GPU complement, ANE restore and FP32 join; not attention/residual or full image')


def selectors(raw,report):
    events=observer_evidence(raw)
    if raw['source_sha256']!=sha256_file(ROOT/'tools/profiling/qwen_base/observer.mm'):raise ValueError('compiled observer source differs')
    if any(e['request']>0 and e['name'] in ('ane.compile_api','ane.load_api') for e in events):
        raise ValueError('warm/check/timed function calls must not compile or load another program')
    models={arm:set() for arm in ARMS}
    for sample in report['samples']:
        owned=[e for e in events if e['request']==sample['scope']];counts=Counter(e['name'] for e in owned)
        if counts['ane.evaluate_api']!=sample['calls'] or counts['ane.compile_api'] or counts['ane.load_api']:
            raise ValueError('actual measured predictions or hot compile/load differs')
        models[sample['arm']].update(e['model'] for e in owned if e['name']=='ane.evaluate_api')
    if models['gpu'] or any(len(models[a])!=1 for a in ARMS[1:]) or len(set.union(*(models[a] for a in ARMS[1:])))!=3:
        raise ValueError('one stable independent model per hybrid arm required')
    return dict(model_counts={a:len(v) for a,v in models.items()},hot_compile_load_zero=True,
                scope='host selectors only; not physical INT8 MAC or hardware overlap')


def child(a):
    os.environ['TURBOCIDER_PROFILE_REVERSE']=str(int(a.reverse))
    observer=Observer(a.observer) if a.mode=='timeline' else None
    if observer:observer.lib.tc_qwen_profile_scope(-1)
    scope_type=C.CFUNCTYPE(None,C.c_int)
    scope=scope_type(('tc_qwen_profile_scope',observer.lib)) if observer else scope_type()
    lib=C.CDLL(str(a.library));fn=lib.tc_w8_functions_real_probe
    fn.argtypes=[C.c_char_p,C.c_char_p,C.c_char_p,C.c_int,scope_type]
    try:
        if fn(str(a.model).encode(),str(a.capture/'trial/gpu-complement-arrays.safetensors').encode(),str(a.output).encode(),a.repeats,scope):
            raise RuntimeError('real FFN consumer failed; report/stderr preserved')
    finally:
        if observer:
            observer.lib.tc_qwen_profile_scope(-3)
            if observer.lib.tc_qwen_profile_wait(2000):raise RuntimeError('GPU completion observer did not drain')
            write_json(a.output/'observer.json',observer.take('tc_qwen_profile_finish'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('library','native-library','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--case',choices=('generate','edit'),default='generate');p.add_argument('--mode',choices=('stock','timeline'),default='stock')
    p.add_argument('--repeats',type=int,choices=(8,16,32),default=16)
    p.add_argument('--reverse',action='store_true')
    p.add_argument('--model',type=Path,default=ROOT/'models/Comfy-Org-Qwen-Image-2.1')
    p.add_argument('--capture',type=Path);p.add_argument('--observer',type=Path,default=ROOT/'build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib')
    p.add_argument('--child',action='store_true');a=p.parse_args()
    if a.capture is None:a.capture=ROOT/('outputs/qwen-dual-gup-register-c6144-stock-v1-20261019' if a.case=='generate' else 'outputs/qwen-dual-gup-register-c6144-edit-v3-20261019')
    for key in ('library','native_library','model','capture','observer'):setattr(a,key,getattr(a,key).resolve(strict=True))
    a.output=a.output.resolve()
    if a.child:child(a);return
    native=json.loads((a.native_library.parent/'runtime-build/runtime-build-manifest.json').read_text())
    def native_sources():
        if any(sha256_file(ROOT/x['path'])!=x['sha256'] for x in native['inputs']['sources']):raise ValueError('actual native source differs from selected library')
    native_sources()
    if shutil.disk_usage(ROOT).free<(2<<30)+(2<<30):p.error('2GiB reserve plus2GiB bounded outputs required')
    cap=a.capture/'trial/gpu-complement-arrays.safetensors';old=json.loads((a.capture/'summary.json').read_text())
    if old['status']!='complete' or old['case']!=a.case or sha256_file(cap)!=old['arrays_sha256']['gpu-complement-arrays.safetensors']:
        p.error('exact complete original frozen data required, not historical timing')
    source_hashes={x:sha256_file(ROOT/x) for x in SOURCES};assets={k:sha256_file(getattr(a,k)) for k in ('library','native_library','observer')};snapshot=model_snapshot(a.model)
    a.output.mkdir(parents=True,exist_ok=False);trial=a.output/'trial';trial.mkdir()
    s=dict(schema='tc-real-w8-functions-profile-v1',status='running',qualification_passed=False,case=a.case,mode=a.mode,repeats=a.repeats,reverse=a.reverse,
           runtime_build_id=native['runtime_build_id'],source_sha256=source_hashes,assets=assets,model_snapshot=snapshot,
           capture=dict(root=str(a.capture),summary_sha256=sha256_file(a.capture/'summary.json'),arrays_sha256=sha256_file(cap)),model=str(a.model))
    write_json(a.output/'summary.json',s);write_json(a.output/'source.json',{x:(ROOT/x).read_text() for x in SOURCES});load=LoadObservation(a.output/'load.jsonl')
    command=[sys.executable,'-B',__file__,'--child','--library',str(a.library),'--native-library',str(a.native_library),'--model',str(a.model),
             '--capture',str(a.capture),'--observer',str(a.observer),'--case',a.case,'--mode',a.mode,'--repeats',str(a.repeats),'--output',str(trial)]
    if a.reverse:command.append('--reverse')
    env={k:v for k,v in benchmark_environment().items() if not k.startswith('TURBOCIDER_') and k!='MLX_DISABLE_COMPILE'}
    env['TURBOCIDER_PROFILE_REVERSE']=str(int(a.reverse))
    try:
        with (a.output/'stdout.log').open('x') as stdout,(a.output/'stderr.log').open('x') as stderr:
            s['memory']=run_sampled(command,repo=ROOT,output=a.output,stem='trial',env=env,stdout=stdout,stderr=stderr,timeout=600,interval_ms=100,max_gap_ms=500,observer=load)
        r=json.loads((trial/'report.json').read_text())
        if (r['loaded_runtime_id']!=native['runtime_build_id'] or r['compiled_native_id']!=native['runtime_build_id'] or r.get('reverse') is not a.reverse or
            r['cpp_sha256']!=source_hashes['tools/profiling/qwen_base/w8_functions_real_probe.mm']):raise ValueError('actual C++ helper/native binding differs')
        s['analysis']=validate(r,a.repeats);s['arrays_sha256']=sha256_file(trial/'arrays.safetensors')
        if a.mode=='timeline':s['observer']=selectors(json.loads((trial/'observer.json').read_text()),r)
        native_sources();s['status']='complete'
    except Exception as e:s.update(status='failed',error=str(e))
    try:s['load']=load.verify()
    except ValueError as e:s['load_error']=str(e)
    s['performance_load_gate_passed']=s['status']=='complete' and 'load_error' not in s
    if (any(sha256_file(ROOT/x)!=value for x,value in source_hashes.items()) or
        any(sha256_file(getattr(a,k))!=value for k,value in assets.items()) or model_snapshot(a.model)!=snapshot):
        s.update(status='failed',error='source/assets/model changed')
    write_json(a.output/'summary.json',s);print(json.dumps(dict(status=s['status'],output=str(a.output),analysis=s.get('analysis'),load_error=s.get('load_error'))))
    if s['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
