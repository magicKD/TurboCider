#!/usr/bin/env python3
"""Explicit approximate runtime-cache tier; never a same-full-work benchmark."""
import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import collect as full
import analyze
from collect import ROOT,sha256_file,write_json,model_snapshot
from runtime_ane_common import benchmark_environment,validate_results,validate_fixed_async,validate_fp32_channel_join
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

SOURCES=('tools/profiling/qwen_base/runtime_cache_collect.py',)+full.SOURCES
RECIPE='runtime-conservative-cache-f8w8-onehop-retire-v1'
QUALITY_BUDGET=dict(rgb_ssim_min=.98,rgb_mae_255_max=3.,alpha_max_abs=4)


def environment(route,mode,case,cache):
    if route not in ('gpu','runtime') or mode not in ('stock','timeline') or case not in ('generate','edit') or cache not in ('off','conservative'):
        raise ValueError('explicit base512 route/mode/task/cache tier required')
    runtime=route=='runtime'
    flags=full.environment(route,mode,6144 if runtime else 4096,runtime_prefix=runtime and case=='edit',
        private_qkv=runtime,fixed_program=runtime,ffn_activation='sigmoid-complement' if runtime else 'exp',
        runtime_text_prefix=runtime and case=='generate',text_prefix_qkv_step0='gpu' if runtime and case=='generate' else 'ane',
        private_qkv_prefetch=runtime,gpu_dual_swiglu=False,compiled_complement=False)
    flags.update(TURBOCIDER_QWEN21_RUNTIME_CONSERVATIVE_CACHE='1' if runtime and cache=='conservative' else '0',
                 TURBOCIDER_QWEN21_PREFIX_SNAPSHOT='1')
    return flags


def seed_sequence(seed,count):
    if type(seed) is not int or type(count) is not int or not 0<=seed<(1<<32)-1 or count not in (1,3,4):
        raise ValueError('bounded base seed and one/three/four actual requests required')
    return (seed,seed,seed+1,seed) if count==4 else (seed,seed+1,seed)[:count]


def requests(a):
    values=[]
    seeds=seed_sequence(getattr(a,'seed',73),a.requests)
    for i in range(a.requests):
        q=dict(model='qwen-image-2.1',model_variant='qwen-image-2.1',operation='image.generate' if a.case=='generate' else 'image.edit',
            prompt=a.prompt or full.base.PROMPTS[a.case],width=512,height=512,steps=40,frames=1,audio=False,seed=seeds[i],
            residency='resident',allow_approximation=True,execution='gpu_ane' if a.route=='runtime' else 'gpu',
            qwen21_dit_cache=a.cache,qwen21_reference_size=1024,output=str(a.output/f'{i}.png'))
        if a.route=='runtime':q.update(hybrid_mlp_mode='runtime',ane_manifest=str(a.manifest))
        if a.case=='edit':q['inputs']=[dict(kind='image',role='reference',path=str(a.reference))]
        values.append(q)
    return values


def cache_steps(row,cache):
    value=row.get('qwen21_dbcache')
    if cache=='off':
        if value:raise ValueError('explicit off request performed temporal reuse')
        return 0
    if not value or (value.get('mode'),value.get('front_blocks'),value.get('back_blocks'),value.get('warmup_steps'),value.get('max_consecutive'))!=('conservative',8,0,8,1):
        raise ValueError('actual conservative F8/W8 policy differs')
    steps=value.get('cached_steps')
    threshold=value.get('threshold')
    if (type(steps) is not int or not 0<=steps<=16 or type(threshold) not in (int,float) or not math.isfinite(threshold) or
        abs(threshold-.15)>1e-6 or value.get('saved_middle_blocks')!=24*steps):
        raise ValueError('actual conservative skip budget differs')
    return steps


def validate(metrics,route,case,cache):
    if not metrics:raise ValueError('complete actual request metrics required')
    runtime=route=='runtime'
    validate_results(metrics,route,len(metrics),model_id='qwen-image-2.1',runtime_backend='private' if runtime else 'public',
        expect_device_io=runtime,expected_data_path='w8a8_hadamard' if runtime else None)
    totals=dict(ffn=0,blocks=0,qkv=0,gpu_qkv=0,prefetch=0,discards=0)
    recipe=program=None;coverage=[]
    for i,row in enumerate(metrics):
        if (row.get('model'),row.get('width'),row.get('height'),row.get('steps'),row.get('actual_denoise_steps'),row.get('operation'))!=('qwen-image-2.1',512,512,40,40,'image.generate' if case=='generate' else 'image.edit'):
            raise ValueError('original complete base40 geometry/task required')
        if row.get('lora_applied_projections',0) or row.get('qwen_compiled_complement'):
            raise ValueError('cache tier cannot contain adapter or compiled-complement work')
        skips=cache_steps(row,cache);saved=24*skips;blocks=1280-saved
        coverage.append(dict(cached_steps=skips,saved_middle_blocks=saved,actual_ffn_blocks=blocks if runtime else None))
        special=row.get('qwen_runtime_conservative_cache')
        if not runtime:
            if special or row.get('qwen_private_qkv'):raise ValueError('GPU control retained runtime ANE work')
            continue
        if cache=='off':
            if special:raise ValueError('off control retained runtime cache callback')
        else:
            if not special or special.get('requested') is not True or special.get('recipe')!=RECIPE or special.get('actual_sampler_steps')!=40 or special.get('skipped_middle_blocks')!=saved:
                raise ValueError('actual explicit runtime-cache recipe/steps required')
            for key in ('skip_callbacks_this_request','discarded_qkv_futures_this_request','restaged_step_heads_this_request'):
                if type(special.get(key)) is not int or special[key]!=skips:raise ValueError('all actual skip boundaries must retire/restage QKV: '+key)
            steps=special.get('skipped_steps_this_request')
            if not isinstance(steps,list) or len(steps)!=skips or any(type(n) is not int or not 8<=n<=38 for n in steps) or any(y-x<2 for x,y in zip(steps,steps[1:])):
                raise ValueError('actual skips violate warmup/final/one-consecutive policy')
        prefix=row.get('qwen_runtime_prefix_snapshot') or {};hit=i>0
        if (prefix.get('admitted') is not True or prefix.get('hit') is not hit or prefix.get('published') is not (not hit) or
            prefix.get('decline_reason')!='' or not prefix.get('recipe_sha256') or (recipe is not None and prefix['recipe_sha256']!=recipe)):
            raise ValueError('healthy same-recipe immutable runtime prefix miss/hit required')
        recipe=prefix['recipe_sha256']
        total_rows=row.get('qwen_ffn_phases',{}).get('prefill',{}).get('actual_rows')
        if not hit and (type(total_rows) is not int or total_rows<=1024):raise ValueError('actual complete long prefill geometry required')
        phase=row.get('qwen_ffn_phases') or {};pre=phase.get('prefill') or {};decode=phase.get('decode') or {}
        pre_steps=0 if hit else 1
        if (pre.get('steps_this_request')!=pre_steps or decode.get('steps_this_request')!=40-pre_steps or decode.get('actual_rows')!=1024 or
            pre.get('completed_channel_blocks_this_request')!=32*pre_steps or decode.get('completed_channel_blocks_this_request')!=32*(40-pre_steps)-saved):
            raise ValueError('actual full/skip FFN phase coverage differs')
        calls=blocks if hit else (39*32+32*((total_rows+1023)//1024)-saved)
        totals['ffn']+=calls;totals['blocks']+=blocks
        rw=row['hybrid']['runtime_weight'];h=row['hybrid']
        if (rw.get('hybrid_blocks_session_total')!=totals['blocks'] or h.get('runtime_calls_session_total')!=totals['ffn'] or
            rw.get('device_io_calls_session_total')!=totals['ffn'] or rw.get('ane_channels')!=6144 or rw.get('gpu_channels')!=6144 or
            h.get('bucket')!=1024 or rw.get('function_rows',[]) or rw.get('capability_calls_session_total')!=3):
            raise ValueError('actual fixed model/prediction/complete FFN coverage differs')
        qcalls=(1280 if case=='edit' and hit else 1248)-saved
        totals['qkv']+=qcalls;totals['gpu_qkv']+=0 if case=='edit' and hit else 32
        q=row.get('qwen_private_qkv') or {}
        if (q.get('failed') is not False or q.get('failure_reason')!='' or q.get('validation_passed') is not True or q.get('validation_calls_session_total')!=1 or
            type(q.get('validation_relative_l2')) not in (int,float) or not math.isfinite(q['validation_relative_l2']) or not 0<=q['validation_relative_l2']<.02 or
            any(q.get(k)!=totals['qkv'] for k in ('predictions_session_total','completed_blocks_session_total','weight_stages_session_total')) or
            q.get('fallback_blocks_session_total')!=0 or q.get('gpu_blocks_session_total')!=totals['gpu_qkv'] or not q.get('program_key') or
            (program is not None and q['program_key']!=program)):
            raise ValueError('actual fixed successful QKV coverage/source validation differs')
        program=q['program_key'];claims=qcalls-(1 if case=='edit' and hit else 0)
        totals['prefetch']+=claims;totals['discards']+=skips
        p=row.get('qwen_private_qkv_prefetch') or {}
        if p.get('requested') is not True or p.get('enabled') is not True or p.get('future_pending') is not False:raise ValueError('healthy drained actual one-hop runtime required')
        for key,want in dict(attempts=totals['prefetch']+totals['discards'],submissions=totals['prefetch']+totals['discards'],
            producer_submissions=3*(totals['prefetch']+totals['discards']),claims=totals['prefetch'],weight_completions=totals['prefetch'],
            completed_hits=totals['prefetch'],discards=totals['discards'],retired_claims=0,failures=0).items():
            if type(p.get(key+'_session_total')) is not int or p[key+'_session_total']!=want:raise ValueError('actual cache-aware future balance differs: '+key)
    if runtime:
        validate_fp32_channel_join(metrics,True);validate_fixed_async(metrics,True)
        full.validate_fixed_program(metrics);full.validate_activation_receipt(metrics,'sigmoid-complement')
        full.validate_projection_receipt(metrics,'descriptor')
        full.validate_weight_code_cache(metrics,0,'copy')
    return coverage


def child(a):
    # Reconstruct the controlled environment, never inherit another experiment.
    clean=benchmark_environment();os.environ.clear();os.environ.update(clean);os.environ.update(environment(a.route,a.mode,a.case,a.cache))
    observer=full.Observer(a.observer) if a.mode=='timeline' else None
    if observer:observer.lib.tc_qwen_profile_scope(-1)
    lib=C.CDLL(str(a.library));lib.tc_string_free.argtypes=[C.c_void_p];lib.tc_runtime_build_identity.restype=C.c_void_p
    lib.tc_engine_create_model.argtypes=[C.c_char_p,C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
    callback_type=C.CFUNCTYPE(None,C.c_char_p,C.c_void_p)
    lib.tc_engine_generate.argtypes=[C.c_void_p,C.c_char_p,callback_type,C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
    lib.tc_engine_free.argtypes=[C.c_void_p]
    def take(pointer):
        if not pointer:return ''
        try:return C.string_at(pointer).decode()
        finally:lib.tc_string_free(pointer)
    progress=[];errors=[];current=-1
    @callback_type
    def callback(raw,_):
        if observer:
            try:progress.append(dict(request=current,event=json.loads(raw),at_ns=observer.now()))
            except Exception as error:errors.append(str(error))
    engine=C.c_void_p();error=C.c_void_p()
    report=dict(schema='tc-qwen-runtime-cache-report-v1',route=a.route,cache=a.cache,case=a.case,mode=a.mode,seed=a.seed,
                loaded_runtime_id=take(lib.tc_runtime_build_identity()),runs=[],progress=progress)
    try:
        started=time.perf_counter()
        if lib.tc_engine_create_model(b'qwen-image-2.1',str(a.model).encode(),C.byref(engine),C.byref(error)):raise RuntimeError(take(error.value))
        report['constructor_seconds']=time.perf_counter()-started
        for i,q in enumerate(requests(a)):
            current=i;write_json(a.output/f'request-{i}.json',q)
            if observer:observer.lib.tc_qwen_profile_scope(i)
            begin=observer.now() if observer else None;started=time.perf_counter();result=C.c_void_p();error=C.c_void_p()
            code=lib.tc_engine_generate(engine,json.dumps(q).encode(),callback,None,C.byref(result),C.byref(error))
            elapsed=time.perf_counter()-started;end=observer.now() if observer else None
            raw,failure=take(result.value),take(error.value)
            if code:raise RuntimeError(failure)
            metrics=json.loads(raw)
            report['runs'].append(dict(index=i,request_wall_including_export=elapsed,begin_ns=begin,end_ns=end,metrics=metrics))
            write_json(a.output/'report.json',report)
            if metrics.get('seed')!=q['seed']:raise ValueError('actual native seed differs from the requested source sequence')
            validate([x['metrics'] for x in report['runs']],a.route,a.case,a.cache)
            if errors:raise RuntimeError(errors[0])
            if observer and observer.lib.tc_qwen_profile_wait(2000):raise RuntimeError('observer did not drain')
        report['status']='complete'
    except Exception as error:
        report.update(status='failed',error=str(error));raise
    finally:
        if observer:observer.lib.tc_qwen_profile_scope(-3)
        if engine.value:lib.tc_engine_free(engine)
        if observer:write_json(a.output/'observer.json',observer.take('tc_qwen_profile_finish'))
        write_json(a.output/'report.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,default=ROOT/'models/Comfy-Org-Qwen-Image-2.1')
    p.add_argument('--manifest',type=Path,default=ROOT/'outputs/qwen-base-runtime-c1024-h4096-f12288-20261018.json')
    p.add_argument('--reference',type=Path,default=ROOT/'results/qwen21/teapot-blue-edit-native-512.png')
    p.add_argument('--observer',type=Path,default=ROOT/'build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib')
    p.add_argument('--route',choices=('gpu','runtime'),required=True);p.add_argument('--cache',choices=('off','conservative'),required=True)
    p.add_argument('--case',choices=('generate','edit'),default='generate');p.add_argument('--mode',choices=('stock','timeline'),default='stock')
    p.add_argument('--requests',type=int,choices=(1,3,4),default=3,help='four is same-owner prime/base-hit/changed-seed/restored-base, not a timing denominator')
    p.add_argument('--seed',type=int,default=73,help='base seed; subsequent requests use seed+1 and restore the base seed')
    p.add_argument('--prompt');p.add_argument('--child',action='store_true');a=p.parse_args()
    for name in ('library','model','manifest','reference','observer'):setattr(a,name,getattr(a,name).resolve(strict=True))
    a.output=a.output.absolute();flags=environment(a.route,a.mode,a.case,a.cache)
    seed_sequence(a.seed,a.requests)
    if a.child:child(a);return
    if a.output.exists():p.error('fresh output required')
    native=json.loads((a.library.parent/'runtime-build/runtime-build-manifest.json').read_text())
    def native_sources():
        for entry in native['inputs']['sources']:
            if sha256_file(ROOT/entry['path'])!=entry['sha256']:raise ValueError('linked native source changed')
    native_sources();a.output.mkdir(parents=True);trial=a.output/'trial';trial.mkdir()
    assets={name:sha256_file(getattr(a,name)) for name in ('library','manifest','reference','observer')}
    hashes={name:sha256_file(ROOT/name) for name in SOURCES};model=model_snapshot(a.model)
    s=dict(schema='tc-qwen-runtime-cache-window-v1',status='running',qualification_passed=False,route=a.route,cache=a.cache,case=a.case,mode=a.mode,
        requests=a.requests,prompt=a.prompt,seed=a.seed,environment=flags,assets=assets,model_snapshot=model,source_sha256=hashes,
        runtime_build_id=native['runtime_build_id'],quality_budget=QUALITY_BUDGET,
        scope='complete sampler40; conservative tier skips actual middle blocks, not equal-full-work acceleration')
    write_json(a.output/'source.json',{name:(ROOT/name).read_text() for name in SOURCES});write_json(a.output/'summary.json',s)
    load=LoadObservation(a.output/'load.jsonl')
    try:
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--child','--library',str(a.library),'--model',str(a.model),'--manifest',str(a.manifest),
            '--reference',str(a.reference),'--observer',str(a.observer),'--route',a.route,'--cache',a.cache,'--case',a.case,'--mode',a.mode,
            '--requests',str(a.requests),'--seed',str(a.seed),'--output',str(trial)]
        if a.prompt is not None:command+=['--prompt',a.prompt]
        with (a.output/'stdout.log').open('x') as out,(a.output/'stderr.log').open('x') as err:
            s['memory']=run_sampled(command,repo=ROOT,output=a.output,stem='trial',env=benchmark_environment(),stdout=out,stderr=err,
                                    timeout=1800,interval_ms=100,max_gap_ms=500,observer=load)
        r=json.loads((trial/'report.json').read_text())
        if r.get('status')!='complete' or r['loaded_runtime_id']!=native['runtime_build_id']:raise ValueError('complete bound actual runtime required')
        s['coverage']=validate([x['metrics'] for x in r['runs']],a.route,a.case,a.cache)
        if len(r['runs'])!=a.requests:raise ValueError('all requests must finish')
        walls=[x['request_wall_including_export'] for x in r['runs']]
        if any(not math.isfinite(t) or t<=0 for t in walls):raise ValueError('complete finite positive wall samples required')
        s['analysis']=dict(complete_requests=len(walls),first_request_seconds=walls[0],hot_request_seconds=walls[1:],
            hot_median_seconds=statistics.median(walls[1:]) if len(walls)>1 else None)
        if a.mode=='timeline':
            observed=json.loads((trial/'observer.json').read_text())
            if observed['source_sha256']!=hashes['tools/profiling/qwen_base/observer.mm']:raise ValueError('actual observer source differs')
            s['observer']=analyze.observer_contract(observed,r['runs'],a.route=='runtime',expected_models=2 if a.route=='runtime' else 1)
            if a.requests>1 and (not s['observer']['hot_no_compile_observed'] or not s['observer']['hot_no_model_load_observed']):raise ValueError('actual hot compile/load observed')
        s['png_sha256']={q.name:sha256_file(q) for q in sorted(trial.glob('*.png'))}
        if len(s['png_sha256'])!=a.requests:raise ValueError('all complete exported images required')
        native_sources();s['status']='complete'
    except Exception as error:s.update(status='failed',error=str(error))
    try:s['load']=load.verify()
    except ValueError as error:s['load_error']=str(error)
    if model_snapshot(a.model)!=model or any(sha256_file(ROOT/name)!=digest for name,digest in hashes.items()) or any(sha256_file(getattr(a,name))!=digest for name,digest in assets.items()):
        s.update(status='failed',error='bound original asset or tool changed')
    write_json(a.output/'summary.json',s);print(json.dumps(s,indent=2))
    if s['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
