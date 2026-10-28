#!/usr/bin/env python3
"""Independent complete base40 PNG/source/memory/fixed-function request audit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile
from types import SimpleNamespace
import numpy as np
from PIL import Image

import collect
import analyze
from collect import ROOT,sha256_file,write_json,model_snapshot
from expanded_audit import archive_binding,PINS
from gpu_dual_request_audit import memory_contract
from audit import quality as image_quality


def normalized_request(q):
    return {k:v for k,v in q.items() if k not in ('output','ane_manifest')}


def image_contract(requests,images,edit_step0=None):
    """Check stale outputs and the explicit same-arithmetic restoration policy.

    The legacy edit miss/hit pair intentionally changes QKV placement and may
    differ. The GPU-step0 experiment is intended to remove that difference;
    merely reporting its SSIM would not verify the restoration contract.
    """
    if not requests or len(requests)!=len(images):
        raise ValueError('all complete requests need one independently decoded image')
    for previous,current,left,right in zip(requests,requests[1:],images,images[1:]):
        if normalized_request(previous)!=normalized_request(current) and np.array_equal(left,right):
            raise ValueError('changed request returned stale identical complete image')
    restored=None
    if len(images)>1 and normalized_request(requests[0])==normalized_request(requests[-1]):
        restored=image_quality(images[0],images[-1])
        if edit_step0=='gpu' and not restored['rgba_exact']:
            raise ValueError('GPU-step0 edit policy failed same-seed full RGBA restoration')
    return restored


def replay_key(summary,requests):
    # Only this existing observer toggle may vary. Keep every arithmetic,
    # placement, source and request field in the replay identity.
    environment={k:v for k,v in summary['environment'].items()
                 if k!='TURBOCIDER_QWEN21_PROFILE_STEPS'}
    return json.dumps(dict(runtime=summary['runtime_build_id'],model=summary['model_snapshot'],
        environment=environment,requests=[normalized_request(q) for q in requests]),sort_keys=True)


def replay_contract(control,candidate):
    if len(control)!=len(candidate) or any(not np.array_equal(x,y) for x,y in zip(control,candidate)):
        raise ValueError('same arithmetic stock/timeline or repeated window changed full RGBA')
    return dict(images=len(control),rgba_exact=True)


def complement_pair_contract(control,candidate):
    if control['summary'].get('compiled_complement') not in (None,False) or candidate['summary'].get('compiled_complement') is not True:
        raise ValueError('matched ordinary/compiled complement pair required')
    flag='TURBOCIDER_QWEN21_COMPILED_COMPLEMENT'
    environments=[{k:v for k,v in item['summary']['environment'].items() if k!=flag} for item in (control,candidate)]
    if environments[0]!=environments[1]:raise ValueError('complement pair changed another runtime option')
    replay_contract(control['images'],candidate['images'])
    left,right=control['report']['runs'],candidate['report']['runs']
    if len(left)!=len(right) or len(left)!=len(control['images']):raise ValueError('all complete complement request receipts required')
    for u,v in zip(left,right):
        a=u['metrics'].get('qwen_runtime_prefix_snapshot');b=v['metrics'].get('qwen_runtime_prefix_snapshot')
        if bool(a)!=bool(b):raise ValueError('complement pair changed actual prefix availability')
        if a and (a.get('hit')!=b.get('hit') or not a.get('recipe_sha256') or not b.get('recipe_sha256') or
                  a['recipe_sha256']==b['recipe_sha256']):
            raise ValueError('compiled complement requires matched prefix hits and isolated arithmetic namespace')
    return flag


def speed(control,candidate):
    if any(x['summary']['mode']!='stock' or x['summary']['requests']!=3 or x['summary'].get('load_error') or
           x['summary'].get('load',{}).get('complete') is not True for x in (control,candidate)):
        return None
    a,b=(statistics.median(r['request_wall_including_export'] for r in x['report']['runs'][1:]) for x in (control,candidate))
    return dict(control_seconds=a,candidate_seconds=b,ratio=a/b)


def timing_contract(summary,report):
    walls=[row['request_wall_including_export'] for row in report['runs']]
    if len(walls)!=summary['requests'] or not walls or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in walls):
        raise ValueError('all complete finite positive request timing samples required')
    analysis=summary['analysis'];hot=walls[1:]
    if (analysis.get('complete_requests')!=len(walls) or analysis.get('first_request_seconds')!=walls[0] or
        analysis.get('hot_request_seconds')!=hot or analysis.get('hot_median_seconds')!=(statistics.median(hot) if hot else None)):
        raise ValueError('cold/hot complete-request timing denominator differs from actual requests')
    return walls


def sampler_evidence(report,index,mode):
    steps=[v['event'] for v in report['progress'] if v['request']==index and v['event'].get('phase')=='denoise']
    m=report['runs'][index]['metrics'];previous=report['runs'][index-1]['metrics']['hybrid']['runtime_weight']['hybrid_blocks_session_total'] if index else 0
    if m.get('actual_denoise_steps')!=40 or m['hybrid']['runtime_weight']['hybrid_blocks_session_total']-previous!=1280:
        raise ValueError('forty actual sampler steps and all1280 successful FFN blocks required')
    if mode=='stock' and not report['progress']:
        # The existing stock callback intentionally does not parse/store JSON.
        # Actual counters are incremented by the native sampler/FFN, not echo
        # of requested work. Do not invent a diagnostic progress log here.
        return 'native_actual_steps_and_successful_full_ffn_blocks'
    if [v['completed'] for v in steps]!=list(range(41)) or any(v['total']!=40 for v in steps):
        raise ValueError('all forty actual denoise progress boundaries must complete')
    return 'native_counters_and_complete_progress_sequence'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('native-library','native-source-archive','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--window',type=Path,action='append',required=True)
    p.add_argument('--pair',nargs=3,action='append',default=[],metavar=('LABEL','FIXED','FAMILY'))
    p.add_argument('--policy-pair',nargs=3,action='append',default=[],metavar=('LABEL','ANE_STEP0','GPU_STEP0'))
    p.add_argument('--complement-pair',nargs=3,action='append',default=[],metavar=('LABEL','ORDINARY','COMPILED'))
    p.add_argument('--tool-source-archive',type=Path,action='append',default=[])
    p.add_argument('--observer',type=Path,default=ROOT/'build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib')
    p.add_argument('--model',type=Path,default=ROOT/'models/Comfy-Org-Qwen-Image-2.1');a=p.parse_args()
    if a.output.exists():p.error('fresh independent output required')
    native=json.loads((a.native_library.parent/'runtime-build/runtime-build-manifest.json').read_text());archive_binding(a.native_source_archive,native)
    history={}
    for path in a.tool_source_archive:
        with tarfile.open(path,'r:gz') as archive:
            for m in archive.getmembers():
                if m.isfile():history.setdefault(m.name,set()).add(archive.extractfile(m).read())
    current_model=model_snapshot(a.model);full={};windows={};records=[]
    for name in current_model:
        path=(a.model/name).resolve(strict=True);digest=sha256_file(path)
        if path.name in PINS and digest!=PINS[path.name]:raise ValueError('complete original published source SHA differs')
        full[str(path)]=digest
    for root in a.window:
        root=root.resolve(strict=True);s=json.loads((root/'summary.json').read_text());r=json.loads((root/'trial/report.json').read_text())
        if (s['status']!='complete' or r['status']!='complete' or s['runtime_build_id']!=native['runtime_build_id'] or
            r['loaded_runtime_id']!=native['runtime_build_id'] or s['assets']['library']!=sha256_file(a.native_library) or
            s['assets']['observer']!=sha256_file(a.observer) or s['model_snapshot']!=current_model):
            raise ValueError('complete actual same-model/native/observer binding required')
        source=json.loads((root/'source.json').read_text())
        if set(source)!=set(s['source_sha256']):raise ValueError('complete exact captured profiling source required')
        for name,digest in s['source_sha256'].items():
            raw=source[name].encode()
            if hashlib.sha256(raw).hexdigest()!=digest or not(raw==(ROOT/name).read_bytes() or raw in history.get(name,set())):
                raise ValueError('current-or-archived actual profiling source differs')
        functions=s.get('function_rows')
        if r.get('function_rows')!=functions:raise ValueError('actual requested family configuration differs')
        edit_step0=s.get('edit_prefix_qkv_step0')
        if r.get('edit_prefix_qkv_step0')!=edit_step0:raise ValueError('actual edit first-step policy differs')
        compiled=s.get('compiled_complement')
        if r.get('compiled_complement')!=compiled:raise ValueError('actual compiled complement policy differs')
        options=dict(runtime_text_prefix=s['runtime_text_prefix_snapshot'],text_prefix_qkv_step0=s['text_prefix_qkv_step0'],
                     private_qkv_prefetch=s['private_qkv_prefetch'],gpu_dual_swiglu=s['gpu_dual_swiglu'],function_rows=functions,edit_prefix_qkv_step0=edit_step0,compiled_complement=compiled)
        metrics=[x['metrics'] for x in r['runs']]
        walls=timing_contract(s,r)
        collect.validate(metrics,s['route'],s['mode'],s['channels'],s['runtime_prefix_snapshot'],s['resident_prefix'],s['private_qkv'],s['fixed_program'],s['ffn_projection'],s['ffn_activation'],**options)
        environment=collect.environment(s['route'],s['mode'],s['channels'],s['runtime_prefix_snapshot'],s['resident_prefix'],s['private_qkv'],s['fixed_program'],s['ffn_projection'],s['ffn_activation'],
            runtime_text_prefix=s['runtime_text_prefix_snapshot'],text_prefix_qkv_step0=s['text_prefix_qkv_step0'],
            private_qkv_prefetch=s['private_qkv_prefetch'],gpu_dual_swiglu=s['gpu_dual_swiglu'],function_rows=functions,edit_prefix_qkv_step0=edit_step0,compiled_complement=compiled)
        if environment!=s['environment']:raise ValueError('actual immutable environment/partition differs')
        requests=[json.loads((root/f'trial/request-{i}.json').read_text()) for i in range(s['requests'])]
        manifest=Path(requests[0]['ane_manifest']).resolve(strict=True)
        if sha256_file(manifest)!=s['assets']['manifest']:raise ValueError('original fixed descriptor changed')
        expected=collect.requests(SimpleNamespace(**{**s,'output':root/'trial','manifest':manifest,'reference':
            Path(requests[0]['inputs'][0]['path']) if s['case']=='edit' else ROOT/'results/qwen21/teapot-blue-edit-native-512.png'}))
        if requests!=expected:raise ValueError('actual full prompt/steps/seed/reference/policy requests differ')
        if s['case']=='edit':
            reference=Path(requests[0]['inputs'][0]['path'])
            if sha256_file(reference)!=s['assets']['reference']:raise ValueError('original reference changed')
        images=[]
        step_evidence=[]
        for i,q in enumerate(requests):
            step_evidence.append(sampler_evidence(r,i,s['mode']))
            path=root/f'trial/{i}.png'
            if sha256_file(path)!=s['png_sha256'][path.name]:raise ValueError('original complete output PNG changed')
            with Image.open(path) as im:
                im.load()
                if im.mode!='RGBA' or im.size!=(512,512) or im.format!='PNG':raise ValueError('original complete RGBA512 PNG required')
                images.append(np.asarray(im).copy())
        # A prefix miss and hit are not necessarily the same arithmetic route:
        # e.g. ordinary edit hit admits step0 Private QKV while miss keeps the
        # long prefill QKV on GPU. Compare fixed/family at identical hit states;
        # record seed restoration rather than silently demanding losslessness
        # from a pre-existing approximate cache/placement transition.
        restored=image_contract(requests,images,edit_step0)
        memory_contract(root,s)
        observed=analyze.observer_contract(json.loads((root/'trial/observer.json').read_text()),r['runs'],True,expected_models=2) if s['mode']=='timeline' else None
        if observed and s['requests']>1 and (not observed['hot_no_compile_observed'] or not observed['hot_no_model_load_observed']):
            raise ValueError('actual hot NN compile/load observed')
        if observed and observed!=s['analysis']['observer']:raise ValueError('independent actual selector accounting differs')
        row=dict(root=str(root),case=s['case'],mode=s['mode'],function_rows=functions,edit_prefix_qkv_step0=edit_step0,compiled_complement=compiled,images=len(images),actual_steps=40*len(images),
                 wall_seconds=walls,sampler_evidence=step_evidence,same_seed_miss_hit_quality=restored,
                 load_gate_passed=not s.get('load_error') and s.get('load',{}).get('complete') is True,
                 load_error=s.get('load_error'),observer=observed)
        records.append(row);windows[str(root)]=dict(summary=s,report=r,requests=requests,images=images)
        print(json.dumps(dict(window=str(root),independently_audited=True)),flush=True)
    replays=[];first_replays={}
    for root,window in windows.items():
        key=replay_key(window['summary'],window['requests'])
        if key in first_replays:
            original=first_replays[key]
            replays.append(dict(control=original,candidate=root,
                **replay_contract(windows[original]['images'],window['images'])))
        else:
            first_replays[key]=root
    pairs=[]
    for kind,(label,left,right) in [('family',x) for x in a.pair]+[('edit_step0',x) for x in a.policy_pair]+[('compiled_complement',x) for x in a.complement_pair]:
        x,y=windows[str(Path(left).resolve())],windows[str(Path(right).resolve())]
        if (x['summary']['case']!=y['summary']['case'] or x['summary']['mode']!=y['summary']['mode'] or
            [normalized_request(q) for q in x['requests']]!=[normalized_request(q) for q in y['requests']]):
            raise ValueError('matched fixed/family full requests required')
        if kind=='edit_step0':
            if (x['summary']['case']!='edit' or x['summary'].get('edit_prefix_qkv_step0') not in (None,'ane') or
                y['summary'].get('edit_prefix_qkv_step0')!='gpu' or x['summary'].get('function_rows')!=y['summary'].get('function_rows')):
                raise ValueError('only matched edit first-step route may differ in a policy pair')
            if not np.array_equal(x['images'][0],y['images'][0]):raise ValueError('edit policy unexpectedly changed prefix-miss first image')
            varied='TURBOCIDER_QWEN21_RUNTIME_EDIT_PREFIX_QKV_STEP0'
        elif kind=='compiled_complement':
            varied=complement_pair_contract(x,y)
        else:
            if x['summary'].get('function_rows') is not None or not y['summary'].get('function_rows'):raise ValueError('fixed/family pair required')
            varied='TURBOCIDER_PRIVATE_ANE_FUNCTION_ROWS'
        envs=[{k:v for k,v in z['summary']['environment'].items() if k!=varied} for z in (x,y)]
        if envs[0]!=envs[1]:raise ValueError('comparison changed an unrelated runtime recipe')
        pairs.append(dict(label=label,kind=kind,speed=speed(x,y),quality=[image_quality(u,v) for u,v in zip(x['images'],y['images'])],
                          scope='complete same-native requests; speed is null if either CPU-load gate failed'))
    if model_snapshot(a.model)!=current_model:raise ValueError('source generation changed during independent audit')
    write_json(a.output,dict(status='complete',qualification_passed=False,scope='complete base40 functional/quality evidence; no blanket semantic or physical-engine qualification',
        native_runtime_id=native['runtime_build_id'],native_source_archive_sha256=sha256_file(a.native_source_archive),
        full_source_sha256=full,windows=records,pairs=pairs,replays=replays,
        audit_script_sha256=sha256_file(Path(__file__))))


if __name__=='__main__':main()
