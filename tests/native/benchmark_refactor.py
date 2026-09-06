"""Fresh-process C ABI end-to-end benchmark; keeps PNG hashes and phase timings."""
import argparse, ctypes as C, hashlib, json, statistics, time
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('library','model','output','mode'):p.add_argument('--'+name,required=True)
p.add_argument('--manifest');p.add_argument('--runs',type=int,default=6)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
lib=C.CDLL(str(Path(a.library).resolve()))
lib.tc_string_free.argtypes=[C.c_void_p];lib.tc_engine_free.argtypes=[C.c_void_p]
lib.tc_engine_create.argtypes=[C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
lib.tc_engine_generate.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p,C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
def take(p):
 if not p.value:return ''
 s=C.string_at(p).decode();lib.tc_string_free(p);return s
start=time.perf_counter();engine,error=C.c_void_p(),C.c_void_p()
assert lib.tc_engine_create(a.model.encode(),C.byref(engine),C.byref(error))==0,take(error)
create=time.perf_counter()-start;runs=[]
try:
 for i in range(a.runs):
  r=dict(model='flux2-klein-4b',prompt='A red fox sitting in a snowy forest, soft morning light, detailed photography.',width=512,height=512,steps=4,seed=42,execution=a.mode,output=str(out/f'{i}.png'))
  if a.mode=='gpu_ane':r.update(allow_approximation=True,ane_manifest=a.manifest)
  result,error=C.c_void_p(),C.c_void_p();start=time.perf_counter()
  code=lib.tc_engine_generate(engine,json.dumps(r).encode(),None,None,C.byref(result),C.byref(error));wall=time.perf_counter()-start
  text,message=take(result),take(error);assert code==0,message
  runs.append(dict(wall=wall,metrics=json.loads(text),png_sha256=hashlib.sha256(Path(r['output']).read_bytes()).hexdigest()))
finally:lib.tc_engine_free(engine)
report=dict(mode=a.mode,create_seconds=create,cold_e2e_seconds=create+runs[0]['wall'],warm_median_seconds=statistics.median(x['wall'] for x in runs[1:]),runs=runs)
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='runs'}))
