"""Persistent-engine Z-Image shape benchmark with exact native tokenizer counts."""
import argparse, ctypes as c, json, time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--model',required=True)
p.add_argument('--library',default='build/native/libturbocider.dylib')
p.add_argument('--modes',default='gpu,fixed1056,fixed,enumerated,range')
p.add_argument('--lora')
p.add_argument('--cases',help='optional comma-separated short,long,medium schedule')
a=p.parse_args();a.root=a.root.resolve()
lib=c.CDLL(a.library)
lib.tc_string_free.argtypes=[c.c_void_p]
lib.tc_engine_create_model.argtypes=[c.c_char_p,c.c_char_p,c.POINTER(c.c_void_p),c.POINTER(c.c_void_p)]
lib.tc_engine_generate.argtypes=[c.c_void_p,c.c_char_p,c.c_void_p,c.c_void_p,c.POINTER(c.c_void_p),c.POINTER(c.c_void_p)]
lib.tc_engine_free.argtypes=[c.c_void_p]
lib.tc_z_image_tokenize_json.argtypes=[c.c_char_p,c.c_char_p,c.POINTER(c.c_void_p),c.POINTER(c.c_void_p)]
def take(ptr):
 if not ptr.value:return None
 value=c.string_at(ptr).decode();lib.tc_string_free(ptr);return value
short='A cinematic red fox walking through fresh snow, soft morning light.'
# A controlled length stressor, not a prompt-quality comparison.
prompts={'short':short,'long':short+' snow'*491,'medium':short+' snow'*34}
counts={}
for name,prompt in prompts.items():
 out,err=c.c_void_p(),c.c_void_p();status=lib.tc_z_image_tokenize_json(a.model.encode(),prompt.encode(),c.byref(out),c.byref(err))
 text,error=take(out),take(err)
 if status:raise RuntimeError(error)
 counts[name]=json.loads(text)
assert counts['long']['valid']==512
for mode in a.modes.split(','):
 manifest=None
 if mode=='fixed1056':
  manifest=json.loads(Path('outputs/z-image-m4pro-512-20260907/ane-a8192-b1056.compile.json').read_text())['manifest']
 elif mode!='gpu':
  compiled=a.root/f'compile-{mode}.json'
  if not compiled.exists():raise RuntimeError(f'compile first: {compiled}')
  manifest=json.loads(compiled.read_text())['manifest']
 engine,err=c.c_void_p(),c.c_void_p();start=time.perf_counter()
 status=lib.tc_engine_create_model(b'z-image-turbo',a.model.encode(),c.byref(engine),c.byref(err));error=take(err)
 if status:raise RuntimeError(error)
 report={'mode':mode,'constructor_seconds':time.perf_counter()-start,'token_counts':counts,'runs':[]}
 schedule=a.cases.split(',') if a.cases else (['short']*4 if mode=='fixed1056' else ['short']*3+['long']*3+['medium','short'])
 try:
  for i,case in enumerate(schedule):
   request={'schema_version':2,'model':'z-image-turbo','operation':'image.generate',
    'inputs':[{'kind':'text','role':'prompt','text':prompts[case]}],
    'outputs':[{'kind':'image','path':str(a.root/f'{mode}-{i}-{case}.png'),'width':512,'height':512}],
    'sampling':{'steps':9,'seed':42},
    'execution':{'policy':'gpu_ane' if manifest else 'gpu','residency':'resident','allow_approximation':bool(manifest)},
    'parameters':{'dynamic_text':True}}
   if manifest:request['execution']['ane_manifest']=manifest
   if a.lora:request['loras']=[{'path':a.lora,'strength':1.0,'role':'transformer'}]
   out,err=c.c_void_p(),c.c_void_p();start=time.perf_counter()
   status=lib.tc_engine_generate(engine,json.dumps(request).encode(),None,None,c.byref(out),c.byref(err));wall=time.perf_counter()-start
   text,error=take(out),take(err)
   row={'index':i,'case':case,'wall_seconds':wall,'request':request,'status':status,'error':error,'metrics':json.loads(text) if text else None}
   report['runs'].append(row)
   (a.root/f'bench-{mode}.json').write_text(json.dumps(report,indent=2))
   print(mode,i,case,round(wall,3),error or 'ok',flush=True)
   if status:raise RuntimeError(error)
 finally:lib.tc_engine_free(engine)
