"""Real-weight preparation/cache lifecycle; never downloads or modifies sources."""
import argparse,ctypes as C,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
lib=C.CDLL(str(root/'build/native/libturbocider.dylib'));lib.tc_string_free.argtypes=[C.c_void_p]
lib.tc_engine_create.argtypes=[C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)];lib.tc_engine_free.argtypes=[C.c_void_p];lib.tc_engine_cancel.argtypes=[C.c_void_p]
common=[C.c_void_p,C.c_char_p,C.c_void_p,C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
lib.tc_engine_generate.argtypes=common;lib.tc_engine_cache.argtypes=common
lib.tc_engine_prepare.argtypes=common[:2]+[C.c_int]+common[2:]
lib.tc_engine_unload.argtypes=[C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
CB=C.CFUNCTYPE(None,C.c_char_p,C.c_void_p)
def take(p):
 if not p.value:return ''
 text=C.string_at(p).decode();lib.tc_string_free(p);return text
engine,error=C.c_void_p(),C.c_void_p();status=lib.tc_engine_create(a.model.encode(),C.byref(engine),C.byref(error));assert status==0,take(error)
checks=[];reports={};events=[]
def call(name,value,warmup=None,callback=None):
 result,error=C.c_void_p(),C.c_void_p();args=[engine,json.dumps(value).encode()]
 if warmup is not None:args.append(int(warmup))
 args +=[callback,None,C.byref(result),C.byref(error)]
 code=getattr(lib,name)(*args);text,message=take(result),take(error)
 if code:raise RuntimeError((code,message))
 return json.loads(text)
@CB
def event(raw,_):events.append(json.loads(raw))
r={'model':'flux2-klein-4b','prompt':'A red fox sitting in a snowy forest, soft morning light, detailed photography.','width':256,'height':256,'seed':42,'steps':4,'output':str(out/'never-warmup.png')}
try:
 reports['prepare']=call('tc_engine_prepare',r,False,event);assert reports['prepare']['prepared'];checks.append('load_prompt_and_materialized_weights')
 reports['warmup']=call('tc_engine_prepare',r,True,event);assert reports['warmup']['warmup'] and reports['warmup']['output'] is None and not Path(r['output']).exists();checks.append('warmup_no_output')
 r['output']=str(out/'generated.png');reports['generation']=call('tc_engine_generate',r);assert reports['generation']['prompt_cache_hit'];checks.append('prepared_prompt_reused')
 cache=out/'cache';c={'cache':str(cache),'source':a.source,'action':'compile_manifest'}
 reports['compile']=call('tc_engine_cache',c,callback=event);manifest=reports['compile']['manifest'];assert Path(manifest).exists();checks.append('compile_20_partitions')
 reports['compile_hit']=call('tc_engine_cache',c);assert reports['compile_hit']['cache_hits']==20;checks.append('content_cache_reused')
 c['action']='inspect';reports['cache']=call('tc_engine_cache',c);assert len(reports['cache']['entries'])==20 and reports['cache']['bytes']>0
 r.update(execution='gpu_ane',allow_approximation=True,ane_manifest=manifest,output=str(out/'never-hybrid-warmup.png'))
 reports['hybrid_warmup']=call('tc_engine_prepare',r,True,event);assert not Path(r['output']).exists() and reports['hybrid_warmup']['hybrid']['calls_session_total']==80;checks.append('compiled_partitions_predict_and_warmup')
 r['output']=str(out/'hybrid.png');reports['hybrid']=call('tc_engine_generate',r);assert reports['hybrid']['hybrid']['calls_session_total']==160;checks.append('hybrid_session_reused')
 sentinel=cache/'user-file.txt';sentinel.write_text('keep');c['action']='clear';reports['clear']=call('tc_engine_cache',c);assert reports['clear']['removed_entries']==20 and not Path(manifest).exists() and sentinel.read_text()=='keep';checks.append('clear_owned_cache_only')
 result,error=C.c_void_p(),C.c_void_p();assert lib.tc_engine_unload(engine,C.byref(result),C.byref(error))==0,take(error);reports['unload']=json.loads(take(result));assert reports['unload']['mlx_active_bytes']<64*1024*1024;checks.append('unload_releases_weights_and_hybrid')
 @CB
 def cancel(raw,_):
  if json.loads(raw)['phase']=='denoise':lib.tc_engine_cancel(engine)
 r.pop('ane_manifest');r.update(execution='gpu',output=str(out/'cancelled-warmup.png'))
 try:call('tc_engine_prepare',r,True,cancel);raise AssertionError('cancellation ignored')
 except RuntimeError as ex:assert ex.args[0][0]==2,ex
 assert not Path(r['output']).exists();checks.append('warmup_cancellation')
 (out/'report.json').write_text(json.dumps({'passed':True,'checks':checks,'reports':reports},indent=2));print(json.dumps({'passed':True,'checks':checks}))
finally:lib.tc_engine_free(engine)
