#!/usr/bin/env python3
"""Fixed-shape raw native callback vs explicit dynamic-W compiled complement."""
import argparse
from collections import Counter
import ctypes as C
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

import analyze
from collect import ROOT,Observer,model_snapshot,sha256_file,write_json
from runtime_ane_common import benchmark_environment
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ARMS=('raw','compiled')
SOURCES=tuple('tools/profiling/qwen_base/'+n for n in ('compiled_complement_collect.py','compiled_complement_probe.mm','compiled_complement.hpp','collect.py','analyze.py','observer.mm'))+(
    'tools/native/build_qwen_compiled_complement_probe.sh','tools/native/process_tree_sampler.py','tools/native/verify_process_tree_samples.py',
    'tools/validation/runtime_ane_memory.py','tools/validation/runtime_ane_load.py')


def validate(r,n):
    if (type(n) is not int or n not in (16,32) or r.get('schema')!='tc-compiled-complement-v1' or r.get('status')!='complete' or
        any(r.get(k) is not False for k in ('complete_request','image_export')) or r.get('ane_predictions')!=0 or r.get('extra_owned_weight_bytes')!=0 or
        r.get('repeats')!=n or r.get('warmups')!=2 or r.get('source_unchanged') is not True or r.get('explicit_body_native_bits_exact') is not True or
        r.get('rejected')!=7 or (r.get('m'),r.get('h'),r.get('f'),r.get('channels'))!=(1024,4096,12288,6144)):
        raise ValueError('fixed original BF16 native callback and explicit runtime-W controls required')
    if (r.get('trace_count_after_sources'),r.get('trace_count_final'))!=(1,1):raise ValueError('one shared compiled graph must handle changed X/W without retracing')
    if set(r.get('checks',{}))!={'a','b','w'}:raise ValueError('both inputs and original W1/restoration controls required')
    for v,c in r['checks'].items():
        if any(c.get(k) is not True for k in ('partial_bits_exact','hidden_bits_exact')):raise ValueError('compiled complement must preserve complete BF16/F32 output bits')
        for name,elements in (('partial',4194304),('hidden',6291456)):
            q=c[name]
            if q.get('elements')!=elements or any(type(q.get(k)) not in (int,float) or not math.isfinite(q[k]) or q[k]!=0 for k in ('relative_l2','max_abs')):
                raise ValueError('full actual exact hidden/F32 partial control required')
    expected=[(ARMS[(j+rep+2)%2],rep,rep<0,bool(rep&1),int((rep+2)%4==2)) for rep in range(-2,n) for j in range(2)]
    samples=r['samples']
    if [(s['arm'],s['rep'],s['warmup'],s['input_variant'],s['layer']) for s in samples]!=expected:raise ValueError('rotated original/compiled A/B/W0/W1 warm/measured schedule differs')
    if any(type(s.get('seconds')) not in (int,float) or not math.isfinite(s['seconds']) or s['seconds']<=0 for s in samples):raise ValueError('actual positive completed complement wall required')
    medians={a:statistics.median(s['seconds'] for s in samples if s['arm']==a and not s['warmup']) for a in ARMS}
    per_layer={str(l):{a:statistics.median(s['seconds'] for s in samples if s['arm']==a and not s['warmup'] and s['layer']==l) for a in ARMS} for l in (0,1)}
    return dict(median_seconds=medians,per_layer_seconds=per_layer,reduction=1-medians['compiled']/medians['raw'],measured_complements=2*n,
        scope='complete GPU channel complement on ready frozen input; no ANE, hybrid block, attention or full40 image speed')


def observer_contract(raw,r):
    rows=analyze.observer_evidence(raw)
    if any(x['name'].startswith('ane.') for x in rows):raise ValueError('GPU-only complement cannot invoke ANE')
    counts=Counter(x['request'] for x in rows if x['name']=='metal.command_buffer');expected=set(range(100,100+len(r['samples'])))
    if not all(counts[n]>0 for n in expected):raise ValueError('each sample must complete actual GPU work')
    return dict(ane_calls=0,completed_sample_scopes=len(expected),sample_gpu_cb_count=sum(counts[n] for n in expected),
        scope='actual CB completion evidence; trace count is graph construction, not GPU shader compilation count')


def child(args):
    observer=Observer(args.observer) if args.mode=='timeline' else None
    if observer:observer.lib.tc_qwen_profile_scope(-1)
    C.CDLL(str(args.native_library),mode=C.RTLD_GLOBAL);lib=C.CDLL(str(args.library));scope_type=C.CFUNCTYPE(None,C.c_int)
    scope=scope_type(lambda i:observer.lib.tc_qwen_profile_scope(i)) if observer else scope_type()
    fn=lib.tc_compiled_complement_probe;fn.argtypes=[C.c_char_p]*3+[C.c_int,scope_type];fn.restype=C.c_int
    try:
        if fn(str(args.model).encode(),str(args.capture/'trial/gpu-complement-arrays.safetensors').encode(),str(args.output).encode(),args.repeats,scope):
            raise RuntimeError('compiled complement fixture failed; inspect report/stderr')
    finally:
        if observer:write_json(args.output/'observer.json',observer.take('tc_qwen_profile_finish'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('library','native-library','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--case',choices=('generate','edit'),default='generate');p.add_argument('--mode',choices=('stock','timeline'),default='stock')
    p.add_argument('--repeats',type=int,choices=(16,32),default=16);p.add_argument('--model',type=Path,default=ROOT/'models/Comfy-Org-Qwen-Image-2.1')
    p.add_argument('--observer',type=Path,default=ROOT/'build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib');p.add_argument('--capture',type=Path);p.add_argument('--child',action='store_true');args=p.parse_args()
    if args.capture is None:args.capture=ROOT/('outputs/qwen-dual-gup-register-c6144-stock-v1-20261019' if args.case=='generate' else 'outputs/qwen-dual-gup-register-c6144-edit-v3-20261019')
    for k in ('library','native_library','model','observer','capture'):setattr(args,k,getattr(args,k).resolve(strict=True))
    args.output=args.output.resolve()
    if args.child:child(args);return
    native=json.loads((args.native_library.parent/'runtime-build/runtime-build-manifest.json').read_text())
    def sources():
        if any(sha256_file(ROOT/x['path'])!=x['sha256'] for x in native['inputs']['sources']):raise ValueError('native source differs from bound C++ ABI')
    sources()
    if shutil.disk_usage(ROOT).free<(2<<30)+(200<<20):raise ValueError('2GiB reserve plus bounded200MiB output required')
    old=json.loads((args.capture/'summary.json').read_text());cap=args.capture/'trial/gpu-complement-arrays.safetensors'
    if old['status']!='complete' or old['case']!=args.case or sha256_file(cap)!=old['arrays_sha256']['gpu-complement-arrays.safetensors']:raise ValueError('complete original hidden/source capture required')
    capture=dict(root=str(args.capture),summary_sha256=sha256_file(args.capture/'summary.json'),arrays_sha256=sha256_file(cap),
        scope='data-only frozen actual first-decode base40 input; no old C++ execution or historical timing denominator')
    hashes={x:sha256_file(ROOT/x) for x in SOURCES};assets={k:sha256_file(getattr(args,k)) for k in ('library','native_library','observer')};identity=model_snapshot(args.model)
    args.output.mkdir(parents=True,exist_ok=False);trial=args.output/'trial';trial.mkdir();s=dict(schema='tc-compiled-complement-profile-v1',status='running',qualification_passed=False,
        case=args.case,mode=args.mode,repeats=args.repeats,capture=capture,source_sha256=hashes,assets=assets,model_snapshot=identity,runtime_build_id=native['runtime_build_id'])
    write_json(args.output/'summary.json',s);write_json(args.output/'source.json',{x:(ROOT/x).read_text() for x in SOURCES});load=LoadObservation(args.output/'load.jsonl')
    try:
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--child','--library',str(args.library),'--native-library',str(args.native_library),'--case',args.case,
            '--mode',args.mode,'--repeats',str(args.repeats),'--model',str(args.model),'--observer',str(args.observer),'--capture',str(args.capture),'--output',str(trial)]
        env={k:v for k,v in benchmark_environment().items() if not k.startswith('TURBOCIDER_') and k!='MLX_DISABLE_COMPILE'}
        with (args.output/'stdout.log').open('x') as stdout,(args.output/'stderr.log').open('x') as stderr:
            s['memory']=run_sampled(command,repo=ROOT,output=args.output,stem='trial',env=env,stdout=stdout,stderr=stderr,timeout=300,interval_ms=100,max_gap_ms=500,observer=load)
        r=json.loads((trial/'report.json').read_text())
        if (r['compiled_native_id']!=native['runtime_build_id'] or r['loaded_runtime_id']!=native['runtime_build_id'] or
            r['cpp_sha256']!=hashes['tools/profiling/qwen_base/compiled_complement_probe.mm'] or r['header_sha256']!=hashes['tools/profiling/qwen_base/compiled_complement.hpp']):
            raise ValueError('actual compiled helper/header/native identity differs')
        s['analysis']=validate(r,args.repeats);s['arrays_sha256']=sha256_file(trial/'arrays.safetensors')
        if args.mode=='timeline':
            raw=json.loads((trial/'observer.json').read_text())
            if raw['source_sha256']!=hashes['tools/profiling/qwen_base/observer.mm']:raise ValueError('compiled observer source differs')
            s['observer']=observer_contract(raw,r)
        sources();s['status']='complete'
    except Exception as e:s.update(status='failed',error=str(e))
    try:s['load']=load.verify()
    except ValueError as e:s['load_error']=str(e)
    if (any(sha256_file(ROOT/x)!=v for x,v in hashes.items()) or any(sha256_file(getattr(args,k))!=assets[k] for k in assets) or model_snapshot(args.model)!=identity):
        s.update(status='failed',error='source/assets/model changed')
    write_json(args.output/'summary.json',s);print(json.dumps(s,indent=2))
    if s['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
