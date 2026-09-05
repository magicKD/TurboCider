"""Measure the public C ABI from a persistent process, including JSON/export."""
import argparse,ctypes as c,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--model',required=True);p.add_argument('--request',required=True);p.add_argument('--output',required=True);p.add_argument('--runs',type=int,default=5);a=p.parse_args()
lib=c.CDLL(a.library);lib.tc_string_free.argtypes=[c.c_void_p]
lib.tc_engine_create.argtypes=[c.c_char_p,c.POINTER(c.c_void_p),c.POINTER(c.c_void_p)]
lib.tc_engine_generate.argtypes=[c.c_void_p,c.c_char_p,c.c_void_p,c.c_void_p,c.POINTER(c.c_void_p),c.POINTER(c.c_void_p)]
lib.tc_engine_free.argtypes=[c.c_void_p]
def consume(ptr):
 if not ptr.value:return None
 value=c.string_at(ptr).decode();lib.tc_string_free(ptr);return value
r=json.loads(Path(a.request).read_text());r.pop('dump_tensors',None);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
e,err=c.c_void_p(),c.c_void_p();start=time.perf_counter();status=lib.tc_engine_create(a.model.encode(),c.byref(e),c.byref(err));message=consume(err)
if status:raise RuntimeError(message)
constructor=time.perf_counter()-start;runs=[]
try:
 for i in range(a.runs):
  r['output']=str((out/f'{i}.png').resolve());payload=json.dumps(r).encode();result,err=c.c_void_p(),c.c_void_p()
  start=time.perf_counter();status=lib.tc_engine_generate(e,payload,None,None,c.byref(result),c.byref(err));wall=time.perf_counter()-start
  text,error=consume(result),consume(err)
  if status:raise RuntimeError(error)
  runs.append({'run':i,'request_wall_including_export':wall,'metrics':json.loads(text)})
  (out/'report.json').write_text(json.dumps({'engine':'TurboCider native C ABI','constructor_seconds':constructor,'request':r,'runs':runs},indent=2))
finally:lib.tc_engine_free(e)
