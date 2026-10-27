#!/usr/bin/env python3
"""Ready-domain row-scaled I8 dot: strong ready F32/Q8 controls, no rotation/ANE."""
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
from collect import ROOT,Observer,sha256_file,write_json
from runtime_ane_common import benchmark_environment
from runtime_ane_memory import run_sampled
from runtime_ane_load import LoadObservation

ARMS=('raw-f32-ready','q8-f32-ready','factored')
SOURCES=tuple('tools/profiling/qwen_base/'+n for n in ('convrot_factorized.hpp','convrot_factorized_probe.mm','convrot_factorized_collect.py','collect.py','analyze.py','observer.mm'))+(
    'tools/native/build_qwen_convrot_factorized_probe.sh','tools/native/process_tree_sampler.py','tools/native/verify_process_tree_samples.py',
    'tools/validation/runtime_ane_memory.py','tools/validation/runtime_ane_load.py')


def validate(r,n,real,bm,bn):
    if (type(n) is not int or n not in (16,32) or type(real) is not bool or bm not in (32,64) or bn not in (64,128) or
        r.get('schema')!='tc-convrot-row-scale-dot-v1' or r.get('status')!='complete' or r.get('real_weights') is not real or
        r.get('complete_request') is not False or r.get('image_export') is not False or r.get('ane_predictions')!=0 or
        r.get('candidate_owned_dense_weight_bytes')!=0 or r.get('source_unchanged') is not True or r.get('rejected')!=8 or
        (r.get('bm'),r.get('bn'),r.get('bk'))!=(bm,bn,64) or r.get('repeats')!=n or r.get('warmups')!=2):
        raise ValueError('actual ready-dot/no-W-bank/no-ANE/typed-tile fixture required')
    geometry=tuple(r.get(k) for k in ('m','k','n','physical_pitch','row_begin','column_begin'))
    if geometry!=((1024,6144,4096,12288,0,0) if real else (33,128,257,320,5,32)):raise ValueError('complete physical pitch/offset/tail shape differs')
    if r.get('strong_f32_control_weight_bytes')!=2*r['n']*r['k']*4:raise ValueError('ready F32 reference W cost must be declared, not hidden or per-call charged')
    for c in ('a','b','w'):
        for arm in ('packed','factored'):
            q=r['checks'][c][arm]
            if q.get('elements')!=r['m']*r['n'] or any(type(q.get(k)) not in (float,int) or not math.isfinite(q[k]) or q[k]<0 for k in ('relative_l2','max_abs')) or q['relative_l2']>=1e-4:
                raise ValueError('complete finite F32 dots must meet predeclared1e-4 screening bound')
    samples=r['samples'];expected=[(ARMS[(j+rep+3)%3],rep,rep<0,bool(rep&1),int((rep+2)%4==2)) for rep in range(-2,n) for j in range(3)]
    if [(s['arm'],s['rep'],s['warmup'],s['input_variant'],s['layer']) for s in samples]!=expected:raise ValueError('all rotated ready controls, source/input changes and warm samples required')
    if any(type(s.get('seconds')) not in (float,int) or not math.isfinite(s['seconds']) or s['seconds']<=0 for s in samples):raise ValueError('actual completed positive dot wall required')
    medians={a:statistics.median(s['seconds'] for s in samples if s['arm']==a and not s['warmup']) for a in ARMS}
    return dict(median_seconds=medians,reduction_vs_ready_f32=1-medians['factored']/medians[ARMS[0]],measured_dots=3*n,
        scope='ready BF16 dot-domain input, original signed I8 W/F32 row scale; ready full-F32 W is a strong control; excludes H256 rotation, complete FFN, hybrid/image speed')


def observer_contract(raw,r):
    events=analyze.observer_evidence(raw)
    if any(x['name'].startswith('ane.') for x in events):raise ValueError('row-scaled GPU dot must not invoke ANE')
    c=Counter(x['request'] for x in events if x['name']=='metal.command_buffer');expected=set(range(100,100+len(r['samples'])))
    if not all(c[i]>0 for i in expected):raise ValueError('actual completed GPU work missing for dot sample')
    return dict(ane_calls=0,completed_sample_scopes=len(expected),sample_cb_count=sum(c[i] for i in expected))


def child(a):
    observer=Observer(a.observer) if a.mode=='timeline' else None
    if observer:observer.lib.tc_qwen_profile_scope(-1)
    C.CDLL(str(a.native_library),mode=C.RTLD_GLOBAL);lib=C.CDLL(str(a.library));scope_type=C.CFUNCTYPE(None,C.c_int)
    scope=scope_type(lambda n:observer.lib.tc_qwen_profile_scope(n)) if observer else scope_type()
    fn=lib.tc_convrot_factorized_probe;fn.argtypes=[C.c_char_p]*3+[C.c_int]*3+[scope_type];fn.restype=C.c_int
    try:
        if fn(str(a.model).encode(),str(a.capture/'trial/gpu-complement-arrays.safetensors').encode() if a.real else b'-',str(a.output).encode(),a.repeats,a.bm,a.bn,scope):
            raise RuntimeError('row-scale projection compile/numeric/source fixture failed; inspect report/stderr')
    finally:
        if observer:write_json(a.output/'observer.json',observer.take('tc_qwen_profile_finish'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('library','native-library','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--real',action='store_true');p.add_argument('--bm',type=int,choices=(32,64),default=32);p.add_argument('--bn',type=int,choices=(64,128),default=64)
    p.add_argument('--mode',choices=('stock','timeline'),default='stock');p.add_argument('--repeats',type=int,choices=(16,32),default=16)
    p.add_argument('--model',type=Path,default=ROOT/'models/Comfy-Org-Qwen-Image-2.1-ConvRot')
    p.add_argument('--capture',type=Path,default=ROOT/'outputs/qwen-dual-gup-register-c6144-stock-v1-20261019')
    p.add_argument('--observer',type=Path,default=ROOT/'build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib');p.add_argument('--child',action='store_true');a=p.parse_args()
    for k in ('library','native_library','observer'):setattr(a,k,getattr(a,k).resolve(strict=True))
    for k in ('model','capture'):setattr(a,k,getattr(a,k).resolve(strict=a.real))
    a.output=a.output.resolve()
    if a.child:child(a);return
    native=json.loads((a.native_library.parent/'runtime-build/runtime-build-manifest.json').read_text())
    def native_sources():
        if any(sha256_file(ROOT/x['path'])!=x['sha256'] for x in native['inputs']['sources']):raise ValueError('actual native sources changed from bound C++ ABI')
    native_sources()
    if shutil.disk_usage(ROOT).free<(2<<30)+((184 if a.real else 16)<<20):raise ValueError('2GiB reserve plus bounded full-output allowance required')
    capfile=a.capture/'trial/gpu-complement-arrays.safetensors'
    if a.real:
        old=json.loads((a.capture/'summary.json').read_text())
        if old['status']!='complete' or sha256_file(capfile)!=old['arrays_sha256']['gpu-complement-arrays.safetensors']:raise ValueError('complete frozen real BF16 input capture required')
    capture=dict(root=str(a.capture),summary_sha256=sha256_file(a.capture/'summary.json'),arrays_sha256=sha256_file(capfile)) if a.real else None
    weights=a.model/'diffusion_models/qwen_image_2.1_int8_convrot.safetensors';generation=None
    if a.real:
        st=weights.stat();generation=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    hashes={x:sha256_file(ROOT/x) for x in SOURCES};assets={k:sha256_file(getattr(a,k)) for k in ('library','native_library','observer')}
    a.output.mkdir(parents=True,exist_ok=False);trial=a.output/'trial';trial.mkdir();s=dict(schema='tc-convrot-factorized-dot-profile-v1',status='running',qualification_passed=False,
        real=a.real,mode=a.mode,repeats=a.repeats,bm=a.bm,bn=a.bn,capture=capture,original_weight_file=str(weights),weight_generation=generation,
        source_sha256=hashes,assets=assets,runtime_build_id=native['runtime_build_id'])
    write_json(a.output/'summary.json',s);write_json(a.output/'source.json',{x:(ROOT/x).read_text() for x in SOURCES});load=LoadObservation(a.output/'load.jsonl')
    try:
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--child']
        for k in ('library','native_library','model','capture','observer','bm','bn','mode','repeats'):command+=['--'+k.replace('_','-'),str(getattr(a,k))]
        command+=['--output',str(trial)]
        if a.real:command+=['--real']
        env={k:v for k,v in benchmark_environment().items() if not k.startswith('TURBOCIDER_') and k!='MLX_DISABLE_COMPILE'}
        with (a.output/'stdout.log').open('x') as stdout,(a.output/'stderr.log').open('x') as stderr:
            s['memory']=run_sampled(command,repo=ROOT,output=a.output,stem='trial',env=env,stdout=stdout,stderr=stderr,timeout=300,interval_ms=100,max_gap_ms=500,observer=load)
        r=json.loads((trial/'report.json').read_text())
        if (r['compiled_native_id']!=native['runtime_build_id'] or r['loaded_runtime_id']!=native['runtime_build_id'] or
            r['cpp_sha256']!=hashes['tools/profiling/qwen_base/convrot_factorized_probe.mm'] or r['header_sha256']!=hashes['tools/profiling/qwen_base/convrot_factorized.hpp']):raise ValueError('actual compiled header/probe/native identity differs')
        s['analysis']=validate(r,a.repeats,a.real,a.bm,a.bn);s['arrays_sha256']=sha256_file(trial/'arrays.safetensors')
        if a.mode=='timeline':
            raw=json.loads((trial/'observer.json').read_text())
            if raw['source_sha256']!=hashes['tools/profiling/qwen_base/observer.mm']:raise ValueError('actual observer source differs')
            s['observer']=observer_contract(raw,r)
        native_sources();s['status']='complete'
    except Exception as e:s.update(status='failed',error=str(e))
    try:s['load']=load.verify()
    except ValueError as e:s['load_error']=str(e)
    after=None
    if a.real:
        st=weights.stat();after=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    if any(sha256_file(ROOT/x)!=v for x,v in hashes.items()) or any(sha256_file(getattr(a,k))!=v for k,v in assets.items()) or (a.real and after!=generation):s.update(status='failed',error='bound sources/assets changed')
    write_json(a.output/'summary.json',s);print(json.dumps(s,indent=2))
    if s['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
