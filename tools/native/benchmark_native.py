"""Measure a selected public native model from one persistent C ABI process."""
import argparse,ctypes as c,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--model',required=True);p.add_argument('--model-id',default='flux2-klein-4b');p.add_argument('--request',required=True);p.add_argument('--output',required=True);p.add_argument('--runs',type=int,default=5);p.add_argument('--dump-tensors',action='store_true',help='write one dump directory per run; disabled for performance measurements');a=p.parse_args()
lib=c.CDLL(a.library);lib.tc_string_free.argtypes=[c.c_void_p]
lib.tc_engine_create_model.argtypes=[c.c_char_p,c.c_char_p,c.POINTER(c.c_void_p),c.POINTER(c.c_void_p)]
lib.tc_engine_generate.argtypes=[c.c_void_p,c.c_char_p,c.c_void_p,c.c_void_p,c.POINTER(c.c_void_p),c.POINTER(c.c_void_p)]
lib.tc_engine_free.argtypes=[c.c_void_p]
def consume(ptr):
 if not ptr.value:return None
 value=c.string_at(ptr).decode();lib.tc_string_free(ptr);return value
r=json.loads(Path(a.request).read_text());r.pop('dump_tensors',None);r['model']=a.model_id;out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
e,err=c.c_void_p(),c.c_void_p();start=time.perf_counter();status=lib.tc_engine_create_model(a.model_id.encode(),a.model.encode(),c.byref(e),c.byref(err));message=consume(err)
if status:raise RuntimeError(message)
constructor=time.perf_counter()-start;runs=[]
try:
 for i in range(a.runs):
  if a.dump_tensors:r['dump_tensors']=str((out/f'{i}-tensors').resolve())
  r['output']=str((out/f'{i}.png').resolve());payload=json.dumps(r).encode();result,err=c.c_void_p(),c.c_void_p()
  start=time.perf_counter();status=lib.tc_engine_generate(e,payload,None,None,c.byref(result),c.byref(err));wall=time.perf_counter()-start
  text,error=consume(result),consume(err)
  if status:raise RuntimeError(error)
  runs.append({'run':i,'request_wall_including_export':wall,'metrics':json.loads(text)})
  (out/'report.json').write_text(json.dumps({'engine':'TurboCider native C ABI','model_id':a.model_id,'constructor_seconds':constructor,'request':r,'runs':runs},indent=2))
finally:lib.tc_engine_free(e)
