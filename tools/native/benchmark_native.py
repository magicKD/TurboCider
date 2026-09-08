"""Measure a selected public native model from one persistent C ABI process."""
import argparse,ctypes as c,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--model',required=True);p.add_argument('--model-id',default='flux2-klein-4b');p.add_argument('--request',required=True);p.add_argument('--output',required=True);p.add_argument('--runs',type=int,default=5);p.add_argument('--width',type=int);p.add_argument('--height',type=int);p.add_argument('--steps',type=int);p.add_argument('--seed',type=int);p.add_argument('--prompt',action='append',help='override the text prompt; repeat to benchmark prompt changes in one resident session');p.add_argument('--execution',choices=('gpu','auto','gpu_ane'));p.add_argument('--ane-manifest');p.add_argument('--prepare',choices=('none','load','warmup'),default='none',help='run tc_engine_prepare before measured generations; load is resource-only, warmup runs a full no-output request');p.add_argument('--coreml-warmup-iterations',type=int,choices=range(0,9),default=None,help='override per-branch zero-input Core ML warmup iterations');p.add_argument('--lora');p.add_argument('--lora-strength',type=float,default=1.0);p.add_argument('--dump-tensors',action='store_true',help='write one dump directory per run; disabled for performance measurements');a=p.parse_args()
lib=c.CDLL(a.library);lib.tc_string_free.argtypes=[c.c_void_p]
lib.tc_engine_create_model.argtypes=[c.c_char_p,c.c_char_p,c.POINTER(c.c_void_p),c.POINTER(c.c_void_p)]
lib.tc_engine_generate.argtypes=[c.c_void_p,c.c_char_p,c.c_void_p,c.c_void_p,c.POINTER(c.c_void_p),c.POINTER(c.c_void_p)]
lib.tc_engine_prepare.argtypes=[c.c_void_p,c.c_char_p,c.c_int,c.c_void_p,c.c_void_p,c.POINTER(c.c_void_p),c.POINTER(c.c_void_p)]
lib.tc_engine_free.argtypes=[c.c_void_p]
def consume(ptr):
 if not ptr.value:return None
 value=c.string_at(ptr).decode();lib.tc_string_free(ptr);return value
r=json.loads(Path(a.request).read_text());r.pop('dump_tensors',None);r['model']=a.model_id;out=Path(a.output);out.mkdir(parents=True,exist_ok=True)

if r.get('schema_version')==2:
 image=next((value for value in r.get('outputs',[]) if value.get('kind')=='image'),None)
 sampling=r.setdefault('sampling',{})
 if image is None:raise RuntimeError('schema v2 benchmark request has no image output')
 if a.width is not None:image['width']=a.width
 if a.height is not None:image['height']=a.height
 if a.steps is not None:sampling['steps']=a.steps
 if a.seed is not None:sampling['seed']=a.seed
 execution=r.setdefault('execution',{})
 if a.execution is not None:execution['policy']=a.execution
 if a.ane_manifest:
  execution['ane_manifest']=str(Path(a.ane_manifest).resolve())
  execution['allow_approximation']=True
 if a.coreml_warmup_iterations is not None:execution['warmup_iterations']=a.coreml_warmup_iterations
else:
 if a.width is not None:r['width']=a.width
 if a.height is not None:r['height']=a.height
 if a.steps is not None:r['steps']=a.steps
 if a.seed is not None:r['seed']=a.seed
 if a.execution is not None:r['execution']=a.execution
 if a.ane_manifest:
  r['ane_manifest']=str(Path(a.ane_manifest).resolve())
  r['allow_approximation']=True
 if a.coreml_warmup_iterations is not None:r['warmup_iterations']=a.coreml_warmup_iterations
if a.lora:
 r['loras']=[{'path':str(Path(a.lora).resolve()),'strength':a.lora_strength,'role':'transformer'}]

def set_output(request,path):
 if request.get('schema_version')==2:
  outputs=request.get('outputs',[])
  image=next((value for value in outputs if value.get('kind')=='image'),None)
  if image is None:raise RuntimeError('schema v2 benchmark request has no image output')
  image['path']=str(path)
 else:request['output']=str(path)
def set_prompt(request,value):
 if request.get('schema_version')==2:
  prompt=next((item for item in request.get('inputs',[]) if item.get('kind')=='text' and item.get('role')=='prompt'),None)
  if prompt is None:raise RuntimeError('schema v2 benchmark request has no prompt input')
  prompt['text']=value
 else:request['prompt']=value
e,err=c.c_void_p(),c.c_void_p();start=time.perf_counter();status=lib.tc_engine_create_model(a.model_id.encode(),a.model.encode(),c.byref(e),c.byref(err));message=consume(err)
if status:raise RuntimeError(message)
constructor=time.perf_counter()-start;runs=[];preparation=None
try:
 if a.prepare!='none':
  result,err=c.c_void_p(),c.c_void_p();payload=json.dumps(r).encode();start=time.perf_counter()
  status=lib.tc_engine_prepare(e,payload,1 if a.prepare=='warmup' else 0,None,None,c.byref(result),c.byref(err));wall=time.perf_counter()-start
  text,error=consume(result),consume(err)
  if status:raise RuntimeError(error)
  preparation={'mode':a.prepare,'wall_seconds':wall,'metrics':json.loads(text)}
 for i in range(a.runs):
  if a.prompt:set_prompt(r,a.prompt[i%len(a.prompt)])
  if a.dump_tensors:r['dump_tensors']=str((out/f'{i}-tensors').resolve())
  set_output(r,(out/f'{i}.png').resolve());payload=json.dumps(r).encode();result,err=c.c_void_p(),c.c_void_p()
  start=time.perf_counter();status=lib.tc_engine_generate(e,payload,None,None,c.byref(result),c.byref(err));wall=time.perf_counter()-start
  text,error=consume(result),consume(err)
  if status:raise RuntimeError(error)
  runs.append({'run':i,'request_wall_including_export':wall,'metrics':json.loads(text)})
  (out/'report.json').write_text(json.dumps({'engine':'TurboCider native C ABI','model_id':a.model_id,'constructor_seconds':constructor,'preparation':preparation,'request':r,'runs':runs},indent=2))
finally:lib.tc_engine_free(e)
