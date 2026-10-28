#!/usr/bin/env python3
"""Frozen actual base40 QKV input, native W8 vs Private fixed512 FP16."""
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

ARMS=('gpu','native-w8','fp16-tiled','fp16-single')
def arms(single_only,compare_prepack=False):return ('gpu','native-w8','fp16-single','fp16-prepack') if compare_prepack else ('gpu','native-w8','fp16-single') if single_only else ARMS
SOURCES=tuple('tools/profiling/qwen_base/'+n for n in ('fp16_qkv_collect.py','fp16_qkv_probe.mm','fp16_qkv.hpp','qkv_ane.hpp','collect.py','analyze.py','observer.mm'))+(
    'tools/native/build_qwen_fp16_qkv_probe.sh','tools/native/process_tree_sampler.py','tools/native/verify_process_tree_samples.py','tools/validation/runtime_ane_memory.py','tools/validation/runtime_ane_load.py')


def capture_contract(root,case):
    s=json.loads((root/'summary.json').read_text());r=json.loads((root/'trial/report.json').read_text());request=r['actual_source_request'];path=root/'trial/arrays.safetensors'
    if (s['status']!='complete' or r['status']!='complete' or s['case']!=case or sha256_file(path)!=s['arrays_sha256'] or
        (request.get('model'),request.get('width'),request.get('height'),request.get('steps'),request.get('seed'),request.get('lora_count'))!=('qwen-image-2.1',512,512,40,73,0) or
        request.get('operation')!='image.'+case or request.get('audio') is not False or request.get('execution')!='gpu' or request.get('allow_approximation') is not False or
        request.get('reference_count')!=(case=='edit') or (case=='edit' and request.get('reference_size')!=1024)):
        raise ValueError('complete frozen original base40 GPU input capture provenance required; no historical timing denominator')
    return dict(root=str(root),arrays_sha256=sha256_file(path),summary_sha256=sha256_file(root/'summary.json'),report_sha256=sha256_file(root/'trial/report.json'),
        request=request,source_weight_sha256=r['actual_qkv_weights_sha256'],scope='frozen actual normalized source only, no old helper execution or old timing reuse')


def validate(r,n):
    single_only=r.get('single_only',False);compare_prepack=r.get('compare_prepack',False);active=arms(single_only,compare_prepack)
    if (type(single_only) is not bool or type(compare_prepack) is not bool or (compare_prepack and not single_only) or type(n) is not int or n not in (16,32) or
        r.get('schema') not in ('tc-private-fp16-qkv-screen-v2','tc-private-fp16-qkv-screen-v3','tc-private-fp16-qkv-screen-v4') or r.get('status')!='complete' or
        any(r.get(k) is not False for k in ('complete_request','image_export')) or r.get('frozen_real_input') is not True or
        r.get('repeats')!=n or r.get('warmups')!=2 or r.get('extra_owned_weight_bytes')!=0 or
        r.get('source_unchanged') is not True or r.get('restored_outputs_verified') is not True):raise ValueError('actual fixed512 real-input component, not image request required')
    if (r.get('fp16_slot_bytes')!=117440512 or r.get('w8_slot_bytes')!=65814528 or r.get('w8_predictions')!=n+6 or r.get('fp16_predictions')!=(len(active)-2)*(n+6) or
        r.get('fp16_tiled_predictions')!=(0 if single_only else n+6) or r.get('fp16_single_predictions')!=(2 if compare_prepack else 1)*(n+6) or
        r.get('w8_validation_calls')!=1 or type(r.get('w8_validation_relative_l2')) not in (int,float) or not 0<=r['w8_validation_relative_l2']<.02):
        raise ValueError('original current W8 strong control and fixed FP16 completed calls/slots required')
    keys=('fp16_single_program_key','w8_program_key') if single_only else ('fp16_program_key','fp16_single_program_key','w8_program_key')
    if single_only and r.get('fp16_program_key')!='':raise ValueError('single-only control must not retain a tiled Program')
    for k in keys:
        if not isinstance(r.get(k),str) or len(r[k])!=64:raise ValueError('actual fixed program key required')
    if len({r[k] for k in keys})!=len(keys):raise ValueError('FP16 tiled/single and W8 program identities must differ')
    samples=r['samples'];order=[(active[(j+rep+len(active))%len(active)],rep) for rep in range(-2,n) for j in range(len(active))]
    if [(s['arm'],s['rep']) for s in samples]!=order:raise ValueError('complete four rotated arms with independent GPU baseline required')
    for s in samples:
        if (s.get('warmup') is not (s['rep']<0) or s.get('input_variant') is not bool(s['rep']&1) or
            type(s.get('seconds')) not in (int,float) or not math.isfinite(s['seconds']) or s['seconds']<=0):raise ValueError('actual completed positive raw QKV times required')
    medians={a:statistics.median(s['seconds'] for s in samples if s['arm']==a and not s['warmup']) for a in active}
    return dict(median_seconds=medians,fp16_reduction_vs_native_w8={a:1-medians[a]/medians['native-w8'] for a in active if a.startswith('fp16')},
        measured_projections=len(active)*n,predictions_per_operator=n+6,scope='paired actual frozen normalized base input, complete QKV incl dynamic W; not FFN/lookahead/full40 image qualification')


def observer_contract(raw,n,single_only=False,compare_prepack=False):
    rows=analyze.observer_evidence(raw);hot=[x for x in rows if x['request']>=1];calls=[x for x in hot if x['name']=='ane.evaluate_api'];models={x['model'] for x in calls}
    count=len(arms(single_only,compare_prepack));expected=Counter({1+phase*count+a:1 for phase in range(4) for a in range(1,count)})
    expected.update({100+(rep+2)*count+j:1 for rep in range(-2,n) for j in range(count) if (j+rep+count)%count!=0})
    model_counts=[n+6,2*(n+6)] if compare_prepack else [n+6]*(count-1)
    if Counter(x['request'] for x in calls)!=expected or len(models)!=len(model_counts) or sorted(Counter(x['model'] for x in calls).values())!=model_counts:
        raise ValueError('three actual candidate programs and complete W8/FP16 predict scopes required')
    if any(x['name'] in ('ane.compile_api','ane.load_api') for x in hot):raise ValueError('hot input/W changes compiled or loaded an ANE graph')
    return dict(runtime_models=len(model_counts),workload_evaluations=len(calls),hot_compile_load_zero=True,
        scope='actual selectors/Metal completions for W8 and FP16 candidate owners, not physical precision or overlap')


def child(args):
    observer=Observer(args.observer) if args.mode=='timeline' else None
    if observer:observer.lib.tc_qwen_profile_scope(-1)
    C.CDLL(str(args.native_library),mode=C.RTLD_GLOBAL);lib=C.CDLL(str(args.library));scope_type=C.CFUNCTYPE(None,C.c_int)
    scope=scope_type(lambda i:observer.lib.tc_qwen_profile_scope(i)) if observer else scope_type()
    fn=lib.tc_fp16_qkv_probe_v4;fn.argtypes=[C.c_char_p,C.c_char_p,C.c_char_p,C.c_int,C.c_int,C.c_int,scope_type];fn.restype=C.c_int
    try:
        if fn(str(args.model).encode(),str(args.capture/'trial/arrays.safetensors').encode(),str(args.output).encode(),args.repeats,int(args.single_only),int(args.compare_prepack),scope):raise RuntimeError('Private FP16 QKV screen failed; inspect report/stderr')
    finally:
        if observer:write_json(args.output/'observer.json',observer.take('tc_qwen_profile_finish'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('library','native-library','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--case',choices=('generate','edit'),default='generate');p.add_argument('--mode',choices=('stock','timeline'),default='stock');p.add_argument('--repeats',type=int,choices=(16,32),default=16)
    p.add_argument('--capture',type=Path);p.add_argument('--single-only',action='store_true');p.add_argument('--compare-prepack',action='store_true');p.add_argument('--model',type=Path,default=ROOT/'models/Comfy-Org-Qwen-Image-2.1')
    p.add_argument('--observer',type=Path,default=ROOT/'build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib');p.add_argument('--child',action='store_true');args=p.parse_args()
    if args.capture is None:args.capture=ROOT/('outputs/qwen-base512-private-qkv-c512-single-gen-v3-20261018' if args.case=='generate' else 'outputs/qwen-base512-private-qkv-c512-single-edit-v4-20261018')
    for k in ('library','native_library','model','observer','capture'):setattr(args,k,getattr(args,k).resolve(strict=True))
    args.output=args.output.resolve()
    if args.compare_prepack and not args.single_only:p.error('prepack timing control must use the same single FP16 Program only')
    if args.child:child(args);return
    manifest=json.loads((args.native_library.parent/'runtime-build/runtime-build-manifest.json').read_text())
    def sources():
        if any(sha256_file(ROOT/e['path'])!=e['sha256'] for e in manifest['inputs']['sources']):raise ValueError('linked native source differs')
    sources();capture=capture_contract(args.capture,args.case)
    if shutil.disk_usage(ROOT).free<(2<<30)+((256 if args.single_only and not args.compare_prepack else 320)<<20):raise ValueError('2GiB reserve plus bounded actual raw projection allowance required')
    hashes={x:sha256_file(ROOT/x) for x in SOURCES};assets={k:sha256_file(getattr(args,k)) for k in ('library','native_library','observer')};identity=model_snapshot(args.model)
    args.output.mkdir(parents=True,exist_ok=False);trial=args.output/'trial';trial.mkdir();s=dict(schema='tc-private-fp16-qkv-profile-v1',status='running',qualification_passed=False,
        case=args.case,mode=args.mode,repeats=args.repeats,single_only=args.single_only,compare_prepack=args.compare_prepack,capture=capture,source_sha256=hashes,assets=assets,model_snapshot=identity,runtime_build_id=manifest['runtime_build_id'])
    write_json(args.output/'summary.json',s);write_json(args.output/'source.json',{x:(ROOT/x).read_text() for x in SOURCES});load=LoadObservation(args.output/'load.jsonl')
    try:
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--child','--library',str(args.library),'--native-library',str(args.native_library),'--mode',args.mode,'--repeats',str(args.repeats),
            '--case',args.case,'--capture',str(args.capture),'--model',str(args.model),'--observer',str(args.observer),'--output',str(trial)]
        if args.single_only:command.append('--single-only')
        if args.compare_prepack:command.append('--compare-prepack')
        env={k:v for k,v in benchmark_environment().items() if not k.startswith('TURBOCIDER_') and k!='MLX_DISABLE_COMPILE'}
        with (args.output/'stdout.log').open('x') as stdout,(args.output/'stderr.log').open('x') as stderr:
            s['memory']=run_sampled(command,repo=ROOT,output=args.output,stem='trial',env=env,stdout=stdout,stderr=stderr,timeout=300,interval_ms=100,max_gap_ms=500,observer=load)
        r=json.loads((trial/'report.json').read_text())
        if (r['compiled_native_id']!=manifest['runtime_build_id'] or r['loaded_runtime_id']!=manifest['runtime_build_id'] or r['cpp_sha256']!=hashes['tools/profiling/qwen_base/fp16_qkv_probe.mm'] or
            r['header_sha256']!=hashes['tools/profiling/qwen_base/fp16_qkv.hpp']):raise ValueError('actual compiled helper/header/native C++ ABI differs')
        s['analysis']=validate(r,args.repeats);s['arrays_sha256']=sha256_file(trial/'arrays.safetensors')
        if args.mode=='timeline':
            raw=json.loads((trial/'observer.json').read_text())
            if raw['source_sha256']!=hashes['tools/profiling/qwen_base/observer.mm']:raise ValueError('compiled observer source differs')
            s['observer']=observer_contract(raw,args.repeats,args.single_only,args.compare_prepack)
        sources();s['status']='complete'
    except Exception as e:s.update(status='failed',error=str(e))
    try:s['load']=load.verify()
    except ValueError as e:s['load_error']=str(e)
    if (model_snapshot(args.model)!=identity or capture_contract(args.capture,args.case)!=capture or any(sha256_file(ROOT/x)!=d for x,d in hashes.items()) or
        any(sha256_file(getattr(args,k))!=d for k,d in assets.items())):s.update(status='failed',error='actual capture/source/model/assets changed')
    write_json(args.output/'summary.json',s);print(json.dumps(s,indent=2))
    if s['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
