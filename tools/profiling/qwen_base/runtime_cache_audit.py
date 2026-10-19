#!/usr/bin/env python3
"""Independent full-media/source/actual-work audit of the approximate cache tier."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
from types import SimpleNamespace
import numpy as np
from PIL import Image

import runtime_cache_collect as screen
from collect import ROOT,sha256_file,write_json,model_snapshot
from expanded_audit import archive_binding,PINS
from gpu_dual_request_audit import memory_contract
from audit import quality
import analyze


def normalized(q):
    return {k:v for k,v in q.items() if k not in ('output','execution','hybrid_mlp_mode','ane_manifest','qwen21_dit_cache')}


def numerical_budget(q):
    b=screen.QUALITY_BUDGET;r=q['rgb']
    return bool(r['ssim']>=b['rgb_ssim_min'] and r['mae_255']<=b['rgb_mae_255_max'] and q['alpha_max_abs']<=b['alpha_max_abs'])


def restoration(requests,images):
    if len(images)!=len(requests) or not images:raise ValueError('one complete decoded image for every request required')
    for a,b,x,y in zip(requests,requests[1:],images,images[1:]):
        if normalized(a)!=normalized(b) and np.array_equal(x,y):raise ValueError('changed real request returned stale identical image')
    if len(images)==4:
        seed=requests[0]['seed']
        if [q['seed'] for q in requests]!=list(screen.seed_sequence(seed,4)) or normalized(requests[1])!=normalized(requests[3]):
            raise ValueError('exact same-seed prime/hit, changed-seed hit and restored-seed hit sequence required')
        if not np.array_equal(images[1],images[3]):raise ValueError('same warm route failed full RGBA seed restoration')
        return dict(hot_seed_restore_rgba_exact=True,prime_hit_quality=quality(images[0],images[1]))
    return dict(hot_seed_restore_rgba_exact=None,prime_hit_quality=quality(images[0],images[-1]) if len(images)>1 and normalized(requests[0])==normalized(requests[-1]) else None)


def load_contract(root,s):
    raw=(root/'load.jsonl').read_bytes();rows=[json.loads(line) for line in raw.splitlines()]
    if len(rows)<2:raise ValueError('original load evidence is incomplete')
    points=[r['monotonic_ns'] for r in rows]
    if any(type(p) is not int for p in points) or any(y<=x for x,y in zip(points,points[1:])):raise ValueError('original load sample clocks invalid')
    busy=any(r['competing_processes'] for r in rows);gap=max(y-x for x,y in zip(points,points[1:]));digest=hashlib.sha256(raw).hexdigest()
    recorded=s.get('load') or {};passed=not s.get('load_error') and not busy and gap<=2500_000_000 and recorded.get('complete') is True
    if recorded and (recorded.get('evidence_sha256')!=digest or recorded.get('samples')!=len(rows)):raise ValueError('original complete load binding changed')
    return dict(passed=passed,competing_activity_observed=busy,maximum_sample_gap_ns=gap,evidence_sha256=digest,
                scope='original continuous process CPU-load heuristic, not GPU exclusivity or physical device attribution')


def compare(control,candidate,quality_only=False):
    x,y=control['summary'],candidate['summary']
    if (x['case']!=y['case'] or x['requests']!=y['requests'] or x['model_snapshot']!=y['model_snapshot'] or
        x['assets']!=y['assets'] or x['runtime_build_id']!=y['runtime_build_id'] or
        [normalized(q) for q in control['requests']]!=[normalized(q) for q in candidate['requests']]):
        raise ValueError('matched actual native/source/full requests and original reference required')
    if not quality_only and x['mode']!=y['mode']:raise ValueError('cross-mode comparisons are quality-only')
    if len(control['images'])!=len(candidate['images']):raise ValueError('all original images required')
    values=[]
    for a,b in zip(control['images'],candidate['images']):
        q=quality(a,b);values.append(dict(quality=q,provisional_numerical_budget_passed=numerical_budget(q)))
    speed=None
    if not quality_only and x['mode']=='stock' and x['requests']==3 and control['load']['passed'] and candidate['load']['passed']:
        a,b=(statistics.median(v['request_wall_including_export'] for v in z['report']['runs'][1:]) for z in (control,candidate))
        speed=dict(control_seconds=a,candidate_seconds=b,ratio=a/b)
    return dict(speed=speed,images=values,all_provisional_numerical_budgets_passed=all(v['provisional_numerical_budget_passed'] for v in values),
                actual_saved_block_counts_differ=[v['saved_middle_blocks'] for v in control['coverage']]!=[v['saved_middle_blocks'] for v in candidate['coverage']],
                actual_runtime_skip_indices={name:[v['metrics'].get('qwen_runtime_conservative_cache',{}).get('skipped_steps_this_request') for v in item['report']['runs']]
                                             for name,item in (('control',control),('candidate',candidate))},
                scope='configuration-level comparison; conservative reuse changes actual work; no equal-full-work kernel or semantic-quality claim')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('native-library','native-source-archive','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',type=Path,default=ROOT/'models/Comfy-Org-Qwen-Image-2.1')
    p.add_argument('--window',type=Path,action='append',required=True)
    p.add_argument('--tool-source-archive',type=Path,action='append',default=[])
    p.add_argument('--pair',nargs=3,action='append',default=[],metavar=('LABEL','CONTROL','CANDIDATE'))
    p.add_argument('--quality-pair',nargs=3,action='append',default=[],metavar=('LABEL','CONTROL','CANDIDATE'))
    a=p.parse_args()
    if a.output.exists():p.error('fresh independent audit required')
    manifest=json.loads((a.native_library.parent/'runtime-build/runtime-build-manifest.json').read_text());archive_binding(a.native_source_archive,manifest)
    history={}
    for file in a.tool_source_archive:
        with tarfile.open(file,'r:gz') as archive:
            for member in archive.getmembers():
                if member.isfile():history.setdefault(member.name,set()).add(archive.extractfile(member).read())
    snapshot=model_snapshot(a.model);full={}
    for name in snapshot:
        file=(a.model/name).resolve(strict=True);digest=sha256_file(file)
        if file.name in PINS and digest!=PINS[file.name]:raise ValueError('full original model payload SHA differs')
        full[str(file)]=digest
    records=[];windows={};replays=[];first={}
    for root in a.window:
        root=root.resolve(strict=True);s=json.loads((root/'summary.json').read_text());r=json.loads((root/'trial/report.json').read_text())
        if (s.get('status')!='complete' or r.get('status')!='complete' or s['runtime_build_id']!=manifest['runtime_build_id'] or
            r['loaded_runtime_id']!=manifest['runtime_build_id'] or s['assets']['library']!=sha256_file(a.native_library) or
            s['model_snapshot']!=snapshot or s.get('quality_budget')!=screen.QUALITY_BUDGET):
            raise ValueError('complete exact native/source/budget binding required')
        source=json.loads((root/'source.json').read_text())
        if set(source)!=set(s['source_sha256']):raise ValueError('complete exact collected tool bytes required')
        for name,digest in s['source_sha256'].items():
            raw=source[name].encode()
            if hashlib.sha256(raw).hexdigest()!=digest or not(raw==(ROOT/name).read_bytes() or raw in history.get(name,set())):
                raise ValueError('actual source differs from current or explicit archived bytes')
        if screen.environment(s['route'],s['mode'],s['case'],s['cache'])!=s['environment']:raise ValueError('actual controlled environment differs')
        if any(r.get(key)!=s[key] for key in ('route','case','cache','mode')):raise ValueError('actual child policy differs')
        requests=[json.loads((root/f'trial/request-{i}.json').read_text()) for i in range(s['requests'])]
        descriptor=Path(requests[0].get('ane_manifest',ROOT/'outputs/qwen-base-runtime-c1024-h4096-f12288-20261018.json')).resolve(strict=True)
        reference=Path(requests[0]['inputs'][0]['path']) if s['case']=='edit' else ROOT/'results/qwen21/teapot-blue-edit-native-512.png'
        args=SimpleNamespace(**s,output=root/'trial',manifest=descriptor,reference=reference)
        if screen.requests(args)!=requests:raise ValueError('actual steps/seed/prompt/reference/cache requests differ')
        if any(run['metrics'].get('seed')!=q['seed'] for run,q in zip(r['runs'],requests)):
            raise ValueError('actual native seed differs from the exact request')
        if 'seed' in s and r.get('seed')!=s['seed']:raise ValueError('actual child base-seed configuration differs')
        for name,file in (('manifest',descriptor),('reference',reference)):
            if sha256_file(file)!=s['assets'][name]:raise ValueError('original manifest/reference changed')
        coverage=screen.validate([v['metrics'] for v in r['runs']],s['route'],s['case'],s['cache'])
        if coverage!=s['coverage'] or len(r['runs'])!=s['requests']:raise ValueError('actual full/skip work differs')
        walls=[v['request_wall_including_export'] for v in r['runs']];hot=walls[1:]
        expected=dict(complete_requests=len(walls),first_request_seconds=walls[0],hot_request_seconds=hot,hot_median_seconds=statistics.median(hot) if hot else None)
        if expected!=s['analysis'] or any(not np.isfinite(t) or t<=0 for t in walls):raise ValueError('complete actual timing denominators differ')
        memory_contract(root,s);load=load_contract(root,s)
        observed=None
        if s['mode']=='timeline':
            raw=json.loads((root/'trial/observer.json').read_text())
            if raw['source_sha256']!=s['source_sha256']['tools/profiling/qwen_base/observer.mm']:raise ValueError('actual observer source differs')
            observed=analyze.observer_contract(raw,r['runs'],s['route']=='runtime',expected_models=2 if s['route']=='runtime' else 1)
            if observed!=s['observer']:raise ValueError('independent actual model/compile/prediction accounting differs')
            for i in range(s['requests']):
                progress=[v['event'] for v in r['progress'] if v['request']==i and v['event'].get('phase')=='denoise']
                if [v['completed'] for v in progress]!=list(range(41)) or any(v['total']!=40 for v in progress):raise ValueError('all actual forty sampler boundaries required')
        images=[]
        for i in range(s['requests']):
            file=root/f'trial/{i}.png'
            if sha256_file(file)!=s['png_sha256'][file.name]:raise ValueError('complete original PNG bytes changed')
            with Image.open(file) as im:
                im.load()
                if im.format!='PNG' or im.mode!='RGBA' or im.size!=(512,512):raise ValueError('complete original RGBA512 image required')
                images.append(np.asarray(im).copy())
        restored=restoration(requests,images)
        record=dict(root=str(root),route=s['route'],cache=s['cache'],case=s['case'],mode=s['mode'],images=len(images),actual_sampler_steps=40*len(images),
            coverage=coverage,analysis=expected,load=load,observer=observed,restoration=restored)
        records.append(record);value=dict(summary=s,report=r,requests=requests,images=images,coverage=coverage,load=load)
        key=json.dumps(dict(route=s['route'],cache=s['cache'],case=s['case'],requests=requests and [normalized(q) for q in requests]),sort_keys=True)
        if key in first:
            other=windows[first[key]]
            if any(not np.array_equal(x,y) for x,y in zip(other['images'],images)):raise ValueError('same recipe stock/timeline/repeat changed full RGBA')
            replays.append(dict(control=first[key],candidate=str(root),images=len(images),rgba_exact=True))
        else:first[key]=str(root)
        windows[str(root)]=value;print(json.dumps(dict(window=str(root),independently_audited=True)),flush=True)
    pairs=[]
    for only_quality,items in [(False,x) for x in a.pair]+[(True,x) for x in a.quality_pair]:
        label,left,right=items;x=windows[str(Path(left).resolve())];y=windows[str(Path(right).resolve())]
        pairs.append(dict(label=label,**compare(x,y,only_quality)))
    if model_snapshot(a.model)!=snapshot:raise ValueError('original source generation changed during audit')
    write_json(a.output,dict(status='complete',qualification_passed=False,scope='complete actual cache-tier requests and numerical budgets; not semantic acceptance or equal-full-work acceleration',
        native_runtime_id=manifest['runtime_build_id'],native_source_archive_sha256=sha256_file(a.native_source_archive),
        full_source_sha256=full,quality_budget=screen.QUALITY_BUDGET,windows=records,pairs=pairs,replays=replays,
        auditor_sha256=sha256_file(Path(__file__))))


if __name__=='__main__':main()
