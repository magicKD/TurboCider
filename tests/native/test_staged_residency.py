"""Real image-input parity, resource release, and hybrid metrics across staged unloading."""
import argparse,ctypes as C,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser()
for name in ['model','manifest','output']:p.add_argument('--'+name,required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[2];out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
lib=C.CDLL(str(root/'build/native/libturbocider.dylib'))
lib.tc_string_free.argtypes=[C.c_void_p];lib.tc_engine_free.argtypes=[C.c_void_p]
lib.tc_engine_create.argtypes=[C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
lib.tc_engine_generate.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p,C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
def take(p):
 if not p.value:return ''
 value=C.string_at(p).decode();lib.tc_string_free(p);return value
engine,error=C.c_void_p(),C.c_void_p();assert lib.tc_engine_create(a.model.encode(),C.byref(engine),C.byref(error))==0,take(error)
reports={}
def run(name,request):
 r=dict(request,output=str(out/(name+'.png')));result,error=C.c_void_p(),C.c_void_p()
 assert lib.tc_engine_generate(engine,json.dumps(r).encode(),None,None,C.byref(result),C.byref(error))==0,take(error)
 reports[name]=json.loads(take(result));return hashlib.sha256(Path(r['output']).read_bytes()).hexdigest()
try:
 r=dict(model='flux2-klein-4b',prompt='A red fox in a snowy forest.',width=256,height=256,steps=4,seed=42,execution='gpu')
 run('input',r)
 for mode in ['gpu','gpu_ane']:
  r.update(execution=mode,allow_approximation=True,ane_manifest=a.manifest)
  for operation,role in [('image.transform','init_image'),('image.edit','reference')]:
   r.update(operation=operation,inputs=[dict(kind='image',role=role,path=str(out/'input.png'),strength=0.5)])
   prefix=mode+'-'+operation
   resident=run(prefix+'-resident',dict(r,residency='resident'))
   staged=run(prefix+'-staged',dict(r,residency='component_staged'))
   assert resident==staged,(mode,operation)
   assert reports[prefix+'-staged']['memory']['mlx_active_bytes']<128*1024*1024
   if mode=='gpu_ane':assert reports[prefix+'-staged']['hybrid']['calls_session_total']>0
 (out/'report.json').write_text(json.dumps(dict(passed=True,checks=['resident_staged_png_parity_both_image_operations_both_backends','staged_releases_weights','hybrid_metrics_survive_unload'],reports=reports),indent=2))
 print('PASS: staged image parity, resource release, hybrid metrics')
finally:lib.tc_engine_free(engine)
