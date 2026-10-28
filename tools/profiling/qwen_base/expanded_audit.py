#!/usr/bin/env python3
"""Independent whole-image/request/source/selector audit after all model windows."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile
import numpy as np
from PIL import Image

import expanded_collect as screen
import analyze
from collect import ROOT,sha256_file,write_json
from audit import quality
from gpu_dual_request_audit import memory_contract

PINS={
    'qwen_image_2.1_turbo_bf16.safetensors':'8b74237a7f505773e2c66726fa696444b4c704585c93a347cff9044937f7914a',
    'qwen_image_2.1_bf16.safetensors':'89f4158d066cc33906a199fca85634f766892dd78f49b6698dabf187ac86c4bc',
    'qwen_image_2.1_int8_convrot.safetensors':'cb74113cb03faecd79611b01fd7fd642f0aa60d6f0b95086abee214d75eaa57d',
    'qwen_image_2.1_vae_bf16.safetensors':'bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9',
    'qwen3vl_8b_bf16.safetensors':'68bdc82bc1b66851162ae656225e7e2068166b603db19bd5d5a3b90eb12669a9',
    'qwen-image-2.1-Q4_K_M.gguf':'631d532e7ca71e8d90a87c71d3699761a812039d22e3370e87498d87754660fe',
    'Qwen3-VL-8B-Instruct-Q4_K_M.gguf':'108e7ff92b78eefd3db4741885104acba514255c11b617d3c7b197a5f46efe89',
}


def archive_binding(path,manifest):
    with tarfile.open(path,'r:gz') as tf:
        members={m.name:m for m in tf.getmembers()}
        if set(members)!={x['path'] for x in manifest['inputs']['sources']}:raise ValueError('exact complete bound native archive required')
        for x in manifest['inputs']['sources']:
            if hashlib.sha256(tf.extractfile(members[x['path']]).read()).hexdigest()!=x['sha256']:raise ValueError('native archive differs from library manifest')


def pixels(path,w,h):
    with Image.open(path) as im:
        im.load()
        if im.format!='PNG' or im.size!=(w,h) or im.mode!='RGBA':raise ValueError('complete original-sized RGBA native PNG required')
        return np.asarray(im).copy()


def normalize(q):
    return {k:v for k,v in q.items() if k not in ('output','execution','hybrid_mlp_mode','ane_manifest','encoder_ane_manifest')}


def pair_fields(control,candidate,quality_only=False):
    fields=('variant','format','case','scenario','requests')+(() if quality_only else ('mode',))
    if any(control[k]!=candidate[k] for k in fields):raise ValueError('matched checkpoint/task/scenario/mode required; cross-mode is quality-only')


def pair_speed(control,candidate,quality_only=False):
    if quality_only or control['mode']!='stock' or candidate['mode']!='stock' or control['scenario']!='repeat':return None
    off,on=(s['analysis']['hot_median_seconds'] for s in (control,candidate))
    if off is None or on is None:return None
    return dict(control=off,candidate=on,reduction=1-on/off,ratio=off/on)


def observer(raw,r,a):
    rows=analyze.observer_evidence(raw);models={x['model'] for x in rows if x['request']>=0 and x['name']=='ane.evaluate_api'}
    expected=int(a.route=='runtime')+int(a.encoder_ane)
    if len(models)!=expected:raise ValueError('actual fixed ANE model count differs from admitted owners')
    previous_ffn=previous_encoder=0;details=[]
    for i,run in enumerate(r['runs']):
        owned=[x for x in rows if x['request']==i];counts=__import__('collections').Counter(x['name'] for x in owned)
        if not counts['metal.command_buffer']:raise ValueError('actual completed GPU work missing')
        if i>0 and (counts['ane.compile_api'] or counts['ane.load_api']):raise ValueError('actual hot ANE compilation/model load observed')
        m=run['metrics'];ffn=(m.get('hybrid') or {}).get('runtime_calls_session_total',previous_ffn)
        enc=(m.get('encoder_hybrid') or {}).get('runtime_calls_session_total',previous_encoder)
        if ffn<previous_ffn or enc<previous_encoder:raise ValueError('retained fixed-owner prediction counters regressed')
        workload=ffn-previous_ffn+enc-previous_encoder;capability=3*expected if i==0 else 0
        if counts['ane.evaluate_api']!=workload+capability:raise ValueError('actual successful predictions do not match full workload plus separate capability calls')
        previous_ffn,previous_encoder=ffn,enc
        details.append(dict(index=i,counts=dict(counts),workload_predictions=workload,capability_predictions=capability))
    return dict(runtime_model_count=len(models),hot_compile_load_zero=True,requests=details,
        scope='actual Private selectors and completed GPU CBs; no physical MAC precision, counters or hardware overlap qualification')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-build',nargs=2,type=Path,metavar=('LIBRARY','SOURCE_ARCHIVE'),action='append',required=True)
    p.add_argument('--window',type=Path,action='append',required=True);p.add_argument('--tool-source-archive',type=Path,action='append',default=[])
    p.add_argument('--pair',nargs=3,metavar=('LABEL','CONTROL','CANDIDATE'),action='append',default=[])
    p.add_argument('--quality-pair',nargs=3,metavar=('LABEL','CONTROL','CANDIDATE'),action='append',default=[],help='allow stock/timeline image comparison, never a speed ratio')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('fresh independent audit output required')
    bindings={};historical={};full_hashes={}
    for lib,archive in a.native_build:
        manifest=json.loads((lib.parent/'runtime-build/runtime-build-manifest.json').read_text());archive_binding(archive,manifest)
        bindings[sha256_file(lib)]=dict(library=str(lib),runtime_build_id=manifest['runtime_build_id'],source_archive_sha256=sha256_file(archive),inputs=len(manifest['inputs']['sources']))
    for path in a.tool_source_archive:
        with tarfile.open(path,'r:gz') as tf:
            for m in tf.getmembers():
                if m.isfile():historical.setdefault(m.name,set()).add(tf.extractfile(m).read())
    records=[];images={};raw_requests={};summaries={}
    for root in a.window:
        s=json.loads((root/'summary.json').read_text());r=json.loads((root/'trial/report.json').read_text());binding=bindings.get(s['assets']['library'])
        if not binding or s['status']!='complete' or r['status']!='complete' or s['runtime_build_id']!=binding['runtime_build_id'] or r['loaded_runtime_id']!=binding['runtime_build_id']:
            raise ValueError('complete actual library/native build binding required')
        source=json.loads((root/'source.json').read_text())
        if set(source)!=set(s['source_sha256']):raise ValueError('complete exact source snapshot required')
        for name,digest in s['source_sha256'].items():
            b=source[name].encode()
            if hashlib.sha256(b).hexdigest()!=digest or not (b==(ROOT/name).read_bytes() or b in historical.get(name,set())):raise ValueError('captured/current-or-archived profiling source differs')
        q=[json.loads((root/f'trial/request-{i}.json').read_text()) for i in range(s['requests'])]
        fields={k:s[k] for k in ('variant','format','route','case','width','height','steps','scenario','mode','requests','channels','encoder_ane','retain_encoder_weights','fresh_prompts')}
        fields.update(runtime_prefix_snapshot=s.get('runtime_prefix_snapshot',False),runtime_text_prefix_snapshot=s.get('runtime_text_prefix_snapshot',False),convrot_quant=s.get('convrot_quant','row'),
            lora=Path(s['lora']) if s.get('lora') else None,lora_ranks=s.get('lora_ranks','separate'),
            output=root/'trial',manifest=Path(q[0].get('ane_manifest',ROOT/'outputs/qwen-base-runtime-c1024-h4096-f12288-20261018.json')),
            reference=Path(q[0]['inputs'][0]['path']) if q[0].get('inputs') else ROOT/'results/qwen21/teapot-blue-edit-native-512.png',
            encoder_manifest=Path(q[0]['encoder_ane_manifest']) if q[0].get('encoder_ane_manifest') else None)
        checkpoint=s['model_snapshot']['diffusion_models/'+screen.LEAVES[s['variant'],s['format']]]
        fields['model']=Path(s['model_root']) if s.get('model_root') else Path(checkpoint['canonical']).parent.parent
        args=SimpleNamespace(**fields);expected=screen.requests(args)
        # Output paths may be relative only in the independent audit invocation.
        if any({k:v for k,v in x.items() if k!='output'}!={k:v for k,v in y.items() if k!='output'} for x,y in zip(q,expected)):raise ValueError('actual full request shape/prompt/seed/policy differs')
        if screen.snapshot(args)!=s['model_snapshot']:raise ValueError('actual source file generation/header changed')
        for info in s['model_snapshot'].values():
            path=Path(info['canonical'])
            if str(path) not in full_hashes:
                digest=sha256_file(path)
                if path.name in PINS and digest!=PINS[path.name]:raise ValueError('full original published asset SHA256 differs')
                full_hashes[str(path)]=digest;print(json.dumps(dict(asset=path.name,full_sha256_verified=True)),flush=True)
        analysis=screen.validate(r,args,expected)
        if analysis!=s['analysis']:raise ValueError('independent full request timing/accounting differs')
        env=screen.environment(args);old=s['environment'].copy()
        for key in ('TURBOCIDER_QWEN21_RUNTIME_PREFIX_SNAPSHOT','TURBOCIDER_QWEN21_RUNTIME_TEXT_PREFIX_SNAPSHOT',
            'TURBOCIDER_QWEN21_RUNTIME_SHARE_LORA_RANKS','TURBOCIDER_QWEN21_RUNTIME_SPLIT_DOWN_RANKS'):
            if env.get(key)=='0':env.pop(key)
            if old.get(key)=='0':old.pop(key)
        if old!=env:raise ValueError('actual explicit source/precision environment differs')
        for key in ('manifest','reference'):
            if sha256_file(getattr(args,key))!=s['assets'][key]:raise ValueError('original descriptor/reference changed')
        if args.encoder_manifest and sha256_file(args.encoder_manifest)!=s['assets']['encoder_manifest']:raise ValueError('original encoder descriptor differs')
        if args.lora and (screen.lora_snapshot(args)!=s['lora_snapshot'] or sha256_file(args.lora)!=s['assets']['lora'] or s['assets']['lora']!=screen.lora_pin(args)[1]):
            raise ValueError('original complete trained adapter/source generation differs')
        im=[]
        for i,request in enumerate(q):
            path=root/f'trial/{i}.png'
            if sha256_file(path)!=s['png_sha256'][path.name]:raise ValueError('complete output PNG changed')
            im.append(pixels(path,request['width'],request['height']))
        restored=None
        if len(im)>1 and normalize(q[0])==normalize(q[-1]):
            restored=bool(np.array_equal(im[0],im[-1]))
            if not restored and not (args.runtime_prefix_snapshot or args.runtime_text_prefix_snapshot):raise ValueError('same seed/shape restored stale or changed original output')
        if len(im)>1 and normalize(q[0])!=normalize(q[1]) and im[0].shape==im[1].shape and np.array_equal(im[0],im[1]):raise ValueError('different real input returned stale identical image')
        observed=None
        if s['mode']=='timeline':
            raw=json.loads((root/'trial/observer.json').read_text())
            if raw['source_sha256']!=s['source_sha256']['tools/profiling/qwen_base/observer.mm']:raise ValueError('actual observer source differs')
            observed=observer(raw,r,args)
            if args.lora:
                proof=screen.validate_source_proof((root/'stderr.log').read_text().splitlines(),args.lora.stat().st_size,args.requests)
                if proof!=s.get('native_lora_source_proof'):raise ValueError('native adapter full-SHA/held-source warm proof differs')
        key=str(root.resolve());images[key]=im;raw_requests[key]=q;summaries[key]=s
        records.append(dict(root=key,variant=s['variant'],format=s['format'],route=s['route'],case=s['case'],mode=s['mode'],scenario=s['scenario'],
            analysis=analysis,observer=observed,same_seed_restore_exact=restored,memory=memory_contract(root,s),load_error=s.get('load_error'),summary_sha256=sha256_file(root/'summary.json')))
    pairs=[]
    for quality_only,label,control,candidate in [(False,*p) for p in a.pair]+[(True,*p) for p in a.quality_pair]:
        g,h=str(Path(control).resolve()),str(Path(candidate).resolve());gs,hs=summaries[g],summaries[h]
        pair_fields(gs,hs,quality_only)
        if gs['model_snapshot']!=hs['model_snapshot'] or [normalize(q) for q in raw_requests[g]]!=[normalize(q) for q in raw_requests[h]]:
            raise ValueError('matched pair must preserve checkpoint, source, task, shape, actual steps, prompt, reference and seeds')
        qs=[dict(index=i,quality=quality(x,y)) for i,(x,y) in enumerate(zip(images[g],images[h]))]
        pairs.append(dict(label=label,control=g,candidate=h,hot_seconds=pair_speed(gs,hs,quality_only),images=qs,quality_only=quality_only,
            scope='matched same-checkpoint actual full requests; cross-mode pairs are quality-only; approximate image metrics are not semantic quality qualification'))
    result=dict(schema='tc-qwen-expanded-audit-v1',status='complete',qualification_passed=False,bindings=bindings,full_asset_sha256=full_hashes,windows=records,pairs=pairs,
        script_sha256=sha256_file(Path(__file__)),tool_source_archive_sha256={str(x):sha256_file(x) for x in a.tool_source_archive})
    a.output.mkdir(parents=True);write_json(a.output/'audit.json',result);print(json.dumps(dict(status='complete',windows=len(records),pairs=len(pairs))),flush=True)


if __name__=='__main__':main()
