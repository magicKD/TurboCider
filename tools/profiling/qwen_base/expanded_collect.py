#!/usr/bin/env python3
"""Qwen base/Turbo, shape and encoder complete-request profiler, no promotion.

Separate stock and selector timelines. One owner serves an entire sequence;
every requested denoise step and actual FFN prediction remains observable.
"""
import argparse
from collections import Counter
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
import time

import analyze
import collect
from collect import ROOT,Observer,sha256_file,write_json
from runtime_ane_common import benchmark_environment
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled
from qwen_encoder_residency_screen import validate_request_local_source
from qwen_lora_source_proof import validate_source_proof

SOURCES=tuple('tools/profiling/qwen_base/'+n for n in ('expanded_collect.py','collect.py','analyze.py','observer.mm'))+(
    'tools/native/process_tree_sampler.py','tools/native/verify_process_tree_samples.py','tools/validation/runtime_ane_memory.py',
    'tools/validation/runtime_ane_load.py','tools/validation/qwen_base_runtime_screen.py','tools/validation/runtime_ane_common.py',
    'tools/validation/qwen_encoder_residency_screen.py','tools/validation/qwen_lora_source_proof.py')
PROMPTS=dict(generate='A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.',
    edit='Change the teapot to bright red. Preserve its shape, handle, spout, table and background.')
LEAVES={('base','bf16'):'qwen_image_2.1_bf16.safetensors',('turbo','bf16'):'qwen_image_2.1_turbo_bf16.safetensors',
    ('base','convrot'):'qwen_image_2.1_int8_convrot.safetensors',('turbo','convrot'):'qwen_image_2.1_turbo_int8_convrot.safetensors',
    ('base','gguf'):'qwen-image-2.1-Q4_K_M.gguf'}
CONVROT_QUANT={
    'row':(0,0,'comfy-h256-direct-q8-source-round-a8-rne-norm-f16-v1'),
    'input256':(256,0,'comfy-h256-direct-q8-source-round-input256-hiddenrow-a8-rne-ratio-norm-f16-v2'),
    'hidden256':(0,256,'comfy-h256-direct-q8-source-round-inputrow-hidden256-a8-rne-ratio-norm-f16-v2'),
    'both256':(256,256,'comfy-h256-direct-q8-source-round-group256-a8-rne-ratio-norm-f16-v2'),
    'bf16-row':(0,0,'comfy-h256-direct-q8-source-round-row-a8-rne-bf16-value-guarded-zeros-canonical-v2'),
}
LORA_PINS={
    'Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors':(128,'bafb91d0047df3f9b8a5a850b0c967f051164314d8aad778dfa34d9c24ec345b'),
    'Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors':(256,'2a0148f5c73abbed5f97da5ea356e439318aadb281d01fce4af39cdf43728803'),
}


def lora_pin(a):
    path=getattr(a,'lora',None)
    ranks=getattr(a,'lora_ranks','separate')
    if ranks not in ('separate','shared','split-down','both') or (ranks!='separate' and (path is None or a.route!='runtime')):
        raise ValueError('LoRA rank scheduling requires an explicit adapter/runtime route')
    if path is None:return None
    if (path.name not in LORA_PINS or a.variant!='base' or a.format!='bf16' or a.steps!=6 or
        (a.width,a.height)!=(512,512) or a.encoder_ane or a.scenario!='repeat' or
        a.runtime_prefix_snapshot or a.runtime_text_prefix_snapshot):
        raise ValueError('LoRA screen requires pinned authored six-step BF16 student512, no encoder or unsupported prefix composition')
    return LORA_PINS[path.name]


def lora_snapshot(a):
    if getattr(a,'lora',None) is None:return None
    s=a.lora.stat()
    return dict(canonical=str(a.lora.resolve(strict=True)),device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def validate_lora(m,a,index=0):
    pin=lora_pin(a)
    if pin is None:
        if m.get('lora_applied_projections',0)!=0:raise ValueError('base control unexpectedly applied LoRA')
        return
    if m.get('lora_strategy')!='inference_time' or m.get('lora_applied_projections')!=227:
        raise ValueError('all227 pinned adapter terms must actually bind online, never merge or omit terms')
    if a.route=='runtime':
        shared=getattr(a,'lora_ranks','separate') in ('shared','both');s=m.get('shared_lora_ranks') or {};blocks=32*a.steps if shared else 0
        if (s.get('enabled') is not shared or s.get('prepared_sets_this_request')!=blocks or
            s.get('completed_hybrid_blocks_this_request')!=blocks or s.get('completed_adapter_rank_arrays_this_request')!=2*blocks):
            raise ValueError('actual shared rank producer/consumer coverage required, not an environment-only flag')
    if a.route=='gpu' and pin[0]==128:
        f=m.get('qwen_lora_swiglu') or {};hit=a.case=='edit' and index>0
        if (f.get('enabled') is not True or f.get('completed_sampler_steps')!=6 or
            f.get('completed_decode_blocks')!=32*(6-(0 if hit else 1)) or f.get('complete_down_lora') is not True or
            f.get('online_independent_ranks') is not True or f.get('rank_math')!='f32' or
            f.get('prefix_snapshot_enabled') is not (a.case=='edit') or f.get('prefix_snapshot_hit') is not hit):
            raise ValueError('actual complete strong GPU typed-LoRA consumer required')
        q=m.get('qwen_corrected_qkv') or {}
        if (q.get('enabled') is not True or q.get('recipe')!='decode-only-corrected-qkv-v3-original-buffers' or
            q.get('weight_storage')!='original-buffers' or q.get('bank_bytes')!=0 or q.get('online_lora') is not True or
            q.get('completed_decode_blocks')!=f['completed_decode_blocks'] or q.get('completed_sampler_steps')!=6 or
            q.get('rank_math')!='f32' or q.get('prefill_changed') is not False):
            raise ValueError('strong GPU control requires actual original-buffer direct QKV, not a slower substituted baseline')


def convrot_quant(a):
    mode=getattr(a,'convrot_quant','row')
    if mode not in CONVROT_QUANT:raise ValueError('explicit supported ConvRot quantization recipe required')
    if mode!='row' and (a.format!='convrot' or a.route!='runtime' or a.encoder_ane):
        raise ValueError('alternate quantization requires the separately admitted ConvRot runtime FFN')
    return CONVROT_QUANT[mode]


def validate_convrot_quant(w,a):
    g,h,recipe=convrot_quant(a)
    if (w.get('a8_group_size')!=g or w.get('hidden_a8_group_size')!=h or
        w.get('source_recipe')!=recipe+'+fp32-partial-join-v1' or w.get('fp32_channel_join_enabled') is not True):
        raise ValueError('actual ConvRot input/hidden grouping, value boundaries or source recipe differs')


def timeout_seconds(value):
    if not math.isfinite(value) or not 1<=value<=3600:raise ValueError('owned experiment timeout must be finite and within1...3600 seconds')
    return value


def validate_runtime_geometry(h,a):
    descriptor=json.loads(a.manifest.read_text())
    rows=descriptor.get('rows')
    if (type(rows) is not int or rows<=0 or h.get('bucket')!=rows or
        h.get('hidden')!=4096 or h.get('mlp_width')!=12288 or h.get('output_channels')!=4096):
        raise ValueError('actual fixed row bucket or Qwen FFN geometry differs from the descriptor')


def environment(a):
    g,h,_=convrot_quant(a)
    pin=lora_pin(a)
    activation='sigmoid-complement' if a.route=='runtime' and a.format!='convrot' and pin is None else 'exp'
    flags=collect.environment('runtime' if a.route=='runtime' else 'gpu','stock',a.channels if a.route=='runtime' else 4096,
        fixed_program=a.route=='runtime',ffn_activation=activation)
    if a.encoder_ane:
        flags.update(collect.environment('runtime','stock',a.channels,fixed_program=True,ffn_activation='sigmoid-complement'))
        flags['TURBOCIDER_QWEN21_ENCODER_RETAIN_RUNTIME']='1';flags['TURBOCIDER_QWEN21_ENCODER_ANE_CHANNELS']=str(a.channels)
    flags['TURBOCIDER_QWEN21_ENCODER_RETAIN_WEIGHTS']=str(int(a.retain_encoder_weights))
    if a.format=='gguf':flags['TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN']='0'
    if a.format=='convrot':
        flags['TURBOCIDER_QWEN21_CONVROT_PACKED']='1'
        if a.route=='runtime':flags['TURBOCIDER_PRIVATE_ANE_DATA_PATH']='convrot_w8a8'
        if g or h:
            flags['TURBOCIDER_PRIVATE_ANE_A8_GROUP_SIZE']='256'
            flags['TURBOCIDER_PRIVATE_ANE_A8_GROUP_SCOPE']='both' if g and h else 'input' if g else 'hidden'
        if getattr(a,'convrot_quant','row')=='bf16-row':flags['TURBOCIDER_PRIVATE_ANE_CONVROT_BF16_BOUNDARIES']='1'
    flags['TURBOCIDER_QWEN21_RUNTIME_PREFIX_SNAPSHOT']=str(int(a.runtime_prefix_snapshot))
    flags['TURBOCIDER_QWEN21_RUNTIME_TEXT_PREFIX_SNAPSHOT']=str(int(a.runtime_text_prefix_snapshot))
    if pin:
        flags['TURBOCIDER_QWEN21_LORA_BF16_AB']='1'
        if a.route=='runtime':
            ranks=getattr(a,'lora_ranks','separate')
            flags['TURBOCIDER_QWEN21_RUNTIME_SHARE_LORA_RANKS']=str(int(ranks in ('shared','both')))
            flags['TURBOCIDER_QWEN21_RUNTIME_SPLIT_DOWN_RANKS']=str(int(ranks in ('split-down','both')))
        if a.case=='edit':flags['TURBOCIDER_QWEN21_LORA_REF512_DIAGNOSTIC']='1'
        if a.route=='gpu' and pin[0]==128:
            flags.update(TURBOCIDER_QWEN21_CORRECTED_QKV='1',TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT='1',TURBOCIDER_QWEN21_LORA_SWIGLU='1',
                TURBOCIDER_QWEN21_CORRECTED_QKV_PREFIX=str(int(a.case=='edit')),TURBOCIDER_QWEN21_PREFIX_SNAPSHOT=str(int(a.case=='edit')))
    if a.mode=='timeline':
        flags['TURBOCIDER_QWEN21_PROFILE_STEPS']='1'
        # Diagnostic only: force THIS process to establish its first full
        # payload hash. Its in-process verified generations still serve hot
        # requests. Do not delete/change the user's authenticated store.
        if pin:flags['TURBOCIDER_SOURCE_REGISTRATION_CACHE']=''
    return flags


def snapshot(a):
    paths=[a.model/'diffusion_models'/LEAVES[a.variant,a.format],a.model/'vae/qwen_image_2.1_vae_bf16.safetensors',a.model/'processor/tokenizer.json']
    text=a.model/'text_encoders/qwen3vl_8b_bf16.safetensors'
    if not text.is_file():text=a.model/'text_encoders/Qwen3-VL-8B-Instruct-Q4_K_M.gguf'
    paths.append(text);result={}
    for p in paths:
        before=p.stat()
        with p.open('rb') as f:
            prefix=f.read(8)
            if p.suffix=='.safetensors':
                n=int.from_bytes(prefix,'little')
                if len(prefix)!=8 or not 2<=n<=16<<20:raise ValueError('bounded complete source header required')
                header=f.read(n)
                if len(header)!=n:raise ValueError('truncated source header')
            else:header=prefix+f.read(65536-8)
        after=p.stat();fields=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if fields(before)!=fields(after):raise ValueError('source changed while snapshotting')
        result[str(p.relative_to(a.model))]=dict(canonical=str(p.resolve(strict=True)),device=after.st_dev,inode=after.st_ino,bytes=after.st_size,
            mtime_ns=after.st_mtime_ns,ctime_ns=after.st_ctime_ns,header_or_prefix_sha256=hashlib.sha256(header).hexdigest())
    return result


def requests(a):
    dims=[(512,512),(768,512),(1024,1024),(512,512)] if a.scenario=='shapes' else [(a.width,a.height)]*a.requests
    result=[]
    for i,(w,h) in enumerate(dims):
        prompt=PROMPTS[a.case]
        if a.fresh_prompts:prompt+=' '+('Use neutral lighting.','Use warm lighting.','Use cool lighting.','Use neutral lighting.')[i]
        r=dict(model='qwen-image-2.1',model_variant='qwen-image-2.1-turbo' if a.variant=='turbo' else 'qwen-image-2.1',
            operation='image.'+a.case,prompt=prompt,width=w,height=h,steps=a.steps,seed=(73,74,73,73)[i],frames=1,audio=False,
            execution='gpu_ane' if a.route=='runtime' else 'gpu',residency='resident',allow_approximation=True,qwen21_dit_cache='off',
            qwen21_reference_size=1024,output=str(a.output/f'{i}.png'))
        if a.scenario=='shapes':r['seed']=73
        if getattr(a,'lora',None) is not None:
            r.update(lora_strategy='inference_time',loras=[dict(path=str(a.lora),role='transformer',strength=1.)])
            if a.case=='edit':r['qwen21_reference_size']=512
        if a.route=='runtime':r.update(hybrid_mlp_mode='runtime',ane_manifest=str(a.manifest))
        if a.encoder_ane:r['encoder_ane_manifest']=str(a.encoder_manifest)
        if a.case=='edit':r['inputs']=[dict(kind='image',role='reference',path=str(a.reference))]
        result.append(r)
    return result


def validate(r,a,expected,partial=False):
    runs=r.get('runs',[])
    if not runs or (not partial and len(runs)!=len(expected)) or len(runs)>len(expected):raise ValueError('all actual complete requests required')
    previous_blocks=previous_calls=0
    for i,row in enumerate(runs):
        q=expected[i];m=row['metrics']
        if row.get('index')!=i or any(m.get(k)!=q[k] for k in ('model','model_variant','operation','width','height','steps','seed')) or m.get('actual_denoise_steps')!=q['steps']:
            raise ValueError('actual checkpoint/sampler/shape/request differs')
        if m.get('checkpoint')!=LEAVES[a.variant,a.format] or m.get('qwen21_dbcache'):
            raise ValueError('wrong source leaf or hidden LoRA/temporal work')
        validate_lora(m,a,i)
        if a.variant=='turbo' and m.get('qwen21_sampling_schedule')!='qwen21-turbo-authored-fixed8-cfg1-v1':raise ValueError('actual Turbo sampling recipe missing')
        events=[x['event'] for x in r['progress'] if x['request']==i and x['event'].get('phase')=='denoise']
        if [x['completed'] for x in events]!=list(range(q['steps']+1)) or any(x.get('total')!=q['steps'] for x in events):raise ValueError('actual completed denoise progress differs')
        if any(type(row.get(k)) not in (int,float) or not math.isfinite(row[k]) or row[k]<=0 for k in ('request_wall_including_export',)):
            raise ValueError('positive complete request wall required')
        if a.route=='runtime':
            h=m.get('hybrid') or {};w=h.get('runtime_weight') or {};blocks=w.get('hybrid_blocks_session_total');calls=h.get('runtime_calls_session_total')
            validate_runtime_geometry(h,a)
            if (h.get('runtime_failed') is not False or w.get('executor_backend')!='private_ane' or w.get('data_path')!=('w8a8_convrot' if a.format=='convrot' else 'w8a8_hadamard') or
                w.get('ane_channels')!=a.channels or w.get('gpu_channels')!=12288-a.channels or w.get('partition_axis')!='intermediate_channels' or
                w.get('fallback_blocks_session_total')!=0 or w.get('overflow_retries_session_total')!=0 or w.get('headroom_scale')!=1 or
                type(blocks) is not int or blocks-previous_blocks!=q['steps']*32 or type(calls) is not int or calls<=previous_calls):
                raise ValueError('actual healthy fixed FFN execution missing, declined/retried/rebuilt or source changed')
            collect.validate_fixed_program([m]);collect.validate_activation_receipt([m],'exp' if a.format=='convrot' or lora_pin(a) else 'sigmoid-complement')
            if lora_pin(a) and (w.get('lora_channel_range_calls_session_total')!=blocks or w.get('lora_channel_full_calls_session_total')!=0):
                raise ValueError('actual complete online LoRA correction coverage on every hybrid block required')
            if lora_pin(a) and w.get('split_down_rank_blocks_session_total')!=(blocks if getattr(a,'lora_ranks','separate') in ('split-down','both') else 0):
                raise ValueError('actual complete split-down rank consumer coverage differs')
            if a.format=='convrot':
                if w.get('convrot_stage_submissions_session_total')!=blocks:raise ValueError('actual H256-aware ConvRot source staging required, not reinterpretation as dense/ordinary affine')
                validate_convrot_quant(w,a)
            previous_blocks,previous_calls=blocks,calls
        elif m.get('hybrid'):raise ValueError('GPU DiT control unexpectedly used runtime FFN')
        if a.runtime_prefix_snapshot or a.runtime_text_prefix_snapshot:
            p=m.get('qwen_runtime_prefix_snapshot') or {};hit=i>0 and not a.fresh_prompts
            if p.get('admitted') is not True or p.get('hit') is not hit or p.get('published') is not (not hit) or p.get('decline_reason')!='':
                raise ValueError('actual admitted completed runtime prefix miss/publish/hit differs')
            if a.runtime_text_prefix_snapshot and m.get('qwen_private_qkv'):
                raise ValueError('Turbo text prefix screen keeps ALL QKV on GPU')
        if a.encoder_ane:
            h=m.get('encoder_hybrid') or {};reuse=m.get('encoder_runtime_reuse') or {}
            if not m.get('prompt_cache_hit') and (h.get('runtime_failed') is not False or reuse.get('actual_calls_this_request',0)<=0 or
                (h.get('runtime_weight') or {}).get('executor_backend')!='private_ane'):raise ValueError('fresh conditioning needs actual successful encoder ANE work')
            if not m.get('prompt_cache_hit'):
                collect.validate_fixed_program([dict(hybrid=h)]);collect.validate_activation_receipt([dict(hybrid=h)],'sigmoid-complement')
                if reuse.get('executor_retained') is not True or reuse.get('executor_reused') is not (i>0):raise ValueError('one actually retained encoder executor must serve all fresh requests')
        elif m.get('encoder_hybrid'):raise ValueError('GPU encoder control unexpectedly used ANE')
        if a.retain_encoder_weights:
            w=m.get('encoder_weight_residency') or {}
            if w.get('enabled') is not True or w.get('weights_retained') is not True or w.get('loads_session_total')!=1 or w.get('decline_reason'):
                raise ValueError('matched encoder comparison requires actual admitted retained original source, not an environment-only request')
            if a.fresh_prompts and (m.get('prompt_cache_hit') is not False or w.get('weights_reused') is not (i>0)):
                raise ValueError('fresh encoder requests must really compute new conditioning with the same source generation')
        elif a.fresh_prompts:
            if m.get('prompt_cache_hit') is not False:raise ValueError('request-local fresh encoder must not hit conditioning cache')
            validate_request_local_source(m.get('encoder_weight_residency'),i)
    return dict(first_seconds=runs[0]['request_wall_including_export'],hot_seconds=[x['request_wall_including_export'] for x in runs[1:]],
        hot_median_seconds=statistics.median(x['request_wall_including_export'] for x in runs[1:]) if len(runs)>1 and a.scenario=='repeat' else None,
        completed_images=len(runs),completed_denoise_steps=sum(x['metrics']['actual_denoise_steps'] for x in runs),
        scope='full native requests including encoder/denoise/VAE/export; cold/hot and shape-changing scenarios kept separate')


def child(a):
    observer=Observer(a.observer) if a.mode=='timeline' else None
    if observer:observer.lib.tc_qwen_profile_scope(-1)
    lib=C.CDLL(str(a.library));lib.tc_string_free.argtypes=[C.c_void_p];lib.tc_runtime_build_identity.restype=C.c_void_p
    lib.tc_engine_create_model.argtypes=[C.c_char_p,C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
    cb_type=C.CFUNCTYPE(None,C.c_char_p,C.c_void_p);lib.tc_engine_generate.argtypes=[C.c_void_p,C.c_char_p,cb_type,C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
    lib.tc_engine_free.argtypes=[C.c_void_p]
    def consume(p):
        if not p:return ''
        try:return C.string_at(p).decode()
        finally:lib.tc_string_free(p)
    current=-1;progress=[];callback_errors=[]
    @cb_type
    def callback(raw,_):
        try:progress.append(dict(request=current,at_ns=observer.now() if observer else time.monotonic_ns(),event=json.loads(raw)))
        except Exception as e:callback_errors.append(str(e))
    engine=C.c_void_p();err=C.c_void_p();r=dict(schema='tc-qwen-expanded-request-v1',status='running',loaded_runtime_id=consume(lib.tc_runtime_build_identity()),runs=[],progress=progress)
    try:
        start=time.perf_counter()
        if lib.tc_engine_create_model(b'qwen-image-2.1',str(a.model).encode(),C.byref(engine),C.byref(err)):raise RuntimeError(consume(err.value))
        r['constructor_seconds']=time.perf_counter()-start;expected=requests(a)
        for i,q in enumerate(expected):
            current=i;write_json(a.output/f'request-{i}.json',q)
            if observer:observer.lib.tc_qwen_profile_scope(i)
            begin=observer.now() if observer else None;out=C.c_void_p();err=C.c_void_p();start=time.perf_counter()
            code=lib.tc_engine_generate(engine,json.dumps(q).encode(),callback,None,C.byref(out),C.byref(err));wall=time.perf_counter()-start
            end=observer.now() if observer else None;value,error=consume(out.value),consume(err.value)
            if code:raise RuntimeError(error)
            if callback_errors:raise RuntimeError(callback_errors[0])
            if observer and observer.lib.tc_qwen_profile_wait(2000):raise RuntimeError('real GPU completion observer failed to drain')
            r['runs'].append(dict(index=i,request_wall_including_export=wall,begin_ns=begin,end_ns=end,metrics=json.loads(value)))
            write_json(a.output/'report.json',r);validate(r,a,expected,partial=True)
        r['status']='complete'
    except Exception as e:r.update(status='failed',error=str(e));raise
    finally:
        if observer:observer.lib.tc_qwen_profile_scope(-3)
        if engine.value:lib.tc_engine_free(engine)
        if observer:write_json(a.output/'observer.json',observer.take('tc_qwen_profile_finish'))
        write_json(a.output/'report.json',r)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('library','model','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--variant',choices=('base','turbo'),required=True);p.add_argument('--format',choices=('bf16','convrot','gguf'),default='bf16')
    p.add_argument('--route',choices=('gpu','runtime'),required=True);p.add_argument('--case',choices=('generate','edit'),default='generate')
    p.add_argument('--width',type=int,default=512);p.add_argument('--height',type=int,default=512);p.add_argument('--steps',type=int)
    p.add_argument('--scenario',choices=('repeat','shapes'),default='repeat');p.add_argument('--requests',type=int,choices=(1,3,4),default=3)
    p.add_argument('--channels',type=int,choices=(4096,5120,6144),default=6144);p.add_argument('--mode',choices=('stock','timeline'),default='stock')
    p.add_argument('--convrot-quant',choices=CONVROT_QUANT,default='row',help='one initial fixed ConvRot recipe per process, never hot-switch')
    p.add_argument('--timeout-seconds',type=float,default=3600,help='whole owned experiment deadline, including cold compilation; never stops shared services')
    p.add_argument('--lora',type=Path,help='pinned Viggle six-step adapter; never a Turbo8/base40 speed comparison')
    p.add_argument('--lora-ranks',choices=('separate','shared','split-down','both'),default='separate')
    p.add_argument('--encoder-ane',action='store_true');p.add_argument('--retain-encoder-weights',action='store_true');p.add_argument('--fresh-prompts',action='store_true')
    p.add_argument('--runtime-prefix-snapshot',action='store_true');p.add_argument('--runtime-text-prefix-snapshot',action='store_true')
    p.add_argument('--manifest',type=Path,default=ROOT/'outputs/qwen-base-runtime-c1024-h4096-f12288-20261018.json')
    p.add_argument('--encoder-manifest',type=Path);p.add_argument('--reference',type=Path,default=ROOT/'results/qwen21/teapot-blue-edit-native-512.png')
    p.add_argument('--observer',type=Path,default=ROOT/'build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib');p.add_argument('--child',action='store_true');a=p.parse_args()
    if a.steps is None:a.steps=6 if a.lora else 8 if a.variant=='turbo' else 40
    try:convrot_quant(a);timeout_seconds(a.timeout_seconds);lora_pin(a)
    except ValueError as e:p.error(str(e))
    if (a.variant,a.format) not in LEAVES or (a.variant=='turbo' and a.steps!=8):p.error('explicit supported checkpoint and authored Turbo8 schedule required')
    if a.scenario=='shapes' and (a.requests!=4 or a.mode!='timeline' or a.encoder_ane or a.retain_encoder_weights):p.error('shapes requires four timeline requests without the separately512-only encoder controls')
    if a.requests==4 and a.scenario!='shapes':p.error('four requests reserved for shape/restoration sequence')
    if a.encoder_ane and (not a.encoder_manifest or a.format!='bf16' or (a.width,a.height)!=(512,512)):p.error('encoder ANE requires explicit descriptor and resident512 BF16 route')
    if a.runtime_prefix_snapshot or a.runtime_text_prefix_snapshot:
        if a.runtime_prefix_snapshot and a.runtime_text_prefix_snapshot:p.error('independent edit/text prefix policy required')
        if a.route!='runtime' or a.encoder_ane or a.format!='bf16' or (a.width,a.height)!=(512,512) or a.scenario!='repeat':p.error('prefix requires separate runtime512 BF16 repeat without encoder ANE')
        if a.runtime_prefix_snapshot!=(a.case=='edit'):p.error('edit prefix requires edit; text prefix requires generation')
    for k in ('library','model','manifest','reference','observer'):setattr(a,k,getattr(a,k).resolve(strict=True))
    if a.encoder_manifest:a.encoder_manifest=a.encoder_manifest.resolve(strict=True)
    if a.lora:a.lora=a.lora.resolve(strict=True)
    a.output=a.output.resolve()
    if a.child:child(a);return
    native=json.loads((a.library.parent/'runtime-build/runtime-build-manifest.json').read_text())
    def native_sources():
        if any(sha256_file(ROOT/x['path'])!=x['sha256'] for x in native['inputs']['sources']):raise ValueError('native source differs from current library')
    native_sources();identity=snapshot(a);adapter_identity=lora_snapshot(a)
    if shutil.disk_usage(ROOT).free<(2<<30)+(256<<20):p.error('2GiB reserve plus256MiB bounded outputs required')
    hashes={x:sha256_file(ROOT/x) for x in SOURCES};assets={k:sha256_file(getattr(a,k)) for k in ('library','manifest','reference','observer')}
    if a.encoder_manifest:assets['encoder_manifest']=sha256_file(a.encoder_manifest)
    if a.lora:
        assets['lora']=sha256_file(a.lora)
        if assets['lora']!=lora_pin(a)[1] or lora_snapshot(a)!=adapter_identity:raise ValueError('full original pinned adapter identity differs or changed during verification')
    a.output.mkdir(parents=True,exist_ok=False);trial=a.output/'trial';trial.mkdir();s=dict(schema='tc-qwen-expanded-profile-v1',status='running',qualification_passed=False,
        variant=a.variant,format=a.format,route=a.route,case=a.case,width=a.width,height=a.height,steps=a.steps,scenario=a.scenario,mode=a.mode,requests=a.requests,
        channels=a.channels,encoder_ane=a.encoder_ane,retain_encoder_weights=a.retain_encoder_weights,fresh_prompts=a.fresh_prompts,convrot_quant=a.convrot_quant,timeout_seconds=a.timeout_seconds,
        lora=str(a.lora) if a.lora else None,lora_snapshot=adapter_identity,lora_rank=lora_pin(a)[0] if a.lora else None,lora_ranks=a.lora_ranks,
        runtime_prefix_snapshot=a.runtime_prefix_snapshot,runtime_text_prefix_snapshot=a.runtime_text_prefix_snapshot,model_root=str(a.model),
        model_snapshot=identity,model_identity_scope='actual original filename, file generations and bounded headers/prefixes; not full payload verification',
        environment=environment(a),assets=assets,source_sha256=hashes,runtime_build_id=native['runtime_build_id'])
    write_json(a.output/'summary.json',s);write_json(a.output/'source.json',{x:(ROOT/x).read_text() for x in SOURCES});load=LoadObservation(a.output/'load.jsonl')
    try:
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--child']
        for k in ('library','model','variant','format','route','case','width','height','steps','scenario','requests','channels','mode','manifest','reference','observer','convrot_quant'):
            command+=['--'+k.replace('_','-'),str(getattr(a,k))]
        command+=['--output',str(trial)]
        for k in ('encoder_ane','retain_encoder_weights','fresh_prompts','runtime_prefix_snapshot','runtime_text_prefix_snapshot'):
            if getattr(a,k):command+=['--'+k.replace('_','-')]
        if a.encoder_manifest:command+=['--encoder-manifest',str(a.encoder_manifest)]
        if a.lora:command+=['--lora',str(a.lora),'--lora-ranks',a.lora_ranks]
        env={k:v for k,v in benchmark_environment().items() if not k.startswith('TURBOCIDER_') and k!='MLX_DISABLE_COMPILE'};env.update(environment(a))
        with (a.output/'stdout.log').open('x') as stdout,(a.output/'stderr.log').open('x') as stderr:
            s['memory']=run_sampled(command,repo=ROOT,output=a.output,stem='trial',env=env,stdout=stdout,stderr=stderr,timeout=a.timeout_seconds,interval_ms=100,max_gap_ms=500,observer=load)
        r=json.loads((trial/'report.json').read_text());s['analysis']=validate(r,a,requests(argparse.Namespace(**{**vars(a),'output':trial})))
        if r['status']!='complete' or r['loaded_runtime_id']!=native['runtime_build_id']:raise ValueError('actual completed native request identity differs')
        if a.mode=='timeline' and a.lora:
            s['native_lora_source_proof']=validate_source_proof((a.output/'stderr.log').read_text().splitlines(),a.lora.stat().st_size,a.requests)
        if a.mode=='timeline':
            raw=json.loads((trial/'observer.json').read_text())
            if raw['source_sha256']!=hashes['tools/profiling/qwen_base/observer.mm']:raise ValueError('actual observer source differs')
            if a.encoder_ane:
                rows=analyze.observer_evidence(raw);hot=[x for x in rows if x['request']>0]
                models={x['model'] for x in rows if x['request']>=0 and x['name']=='ane.evaluate_api'}
                if len(models)!=(2 if a.route=='runtime' else 1) or any(x['name'] in ('ane.compile_api','ane.load_api') for x in hot):raise ValueError('fixed retained encoder/DiT actual models/hotcompile differs')
                s['observer']=dict(runtime_model_count=len(models),hot_compile_load_zero=True)
            else:
                s['observer']=analyze.observer_contract(raw,r['runs'],a.route=='runtime')
                if a.requests>1 and a.route=='runtime' and (s['observer']['hot_no_compile_observed'] is not True or s['observer']['hot_no_model_load_observed'] is not True):raise ValueError('actual hot runtime ANE compile/load observed')
        s['png_sha256']={p.name:sha256_file(p) for p in sorted(trial.glob('*.png'))}
        if len(s['png_sha256'])!=a.requests:raise ValueError('every complete native output PNG required')
        native_sources();s['status']='complete'
    except Exception as e:s.update(status='failed',error=str(e))
    try:s['load']=load.verify()
    except ValueError as e:s['load_error']=str(e)
    if snapshot(a)!=identity or lora_snapshot(a)!=adapter_identity or any(sha256_file(ROOT/x)!=v for x,v in hashes.items()) or any(sha256_file(getattr(a,k))!=v for k,v in assets.items()):s.update(status='failed',error='source/assets changed')
    write_json(a.output/'summary.json',s);print(json.dumps(s,indent=2))
    if s['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
