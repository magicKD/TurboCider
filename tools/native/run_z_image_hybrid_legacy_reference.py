#!/usr/bin/env python3
"""One legacy hybrid reference for a frozen Z prompt and partition; no release qualification."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import mlx.core as mx
from run_streaming_campaign import load_native_library, create_native_engine, consume

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--library',type=Path,required=True)
p.add_argument('--model',type=Path,required=True)
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--residency',choices=['resident','streamed'],required=True)
p.add_argument('--fixture',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
fixture=a.fixture.resolve();prompt=(fixture/'prompt.txt').read_text()
noise=mx.load(str(fixture/'inputs/initial.npy'))
mx.save_safetensors(str(a.output/'noise.safetensors'),{'noise':noise})
manifest=a.manifest.resolve()
metadata=json.loads(manifest.read_text())
assert metadata['shape']['buckets']==[1088] and metadata['shape']['ane_mlp_end']==5120
assert set(metadata['artifacts'])=={str(i) for i in range(32)}
request={'schema_version':2,'model':'z-image-turbo','operation':'image.generate',
 'inputs':[{'kind':'text','role':'prompt','text':prompt}],
 'outputs':[{'kind':'image','path':str(a.output/'image.png'),'width':512,'height':512}],
 'sampling':{'seed':42,'steps':9},'parameters':{'dynamic_text':True,'compile_gpu':False,'noise_path':str(a.output/'noise.safetensors')},
 'dump_tensors':str(a.output/'tensors'),
 'execution':{'policy':'gpu_ane','residency':a.residency,'memory_budget_bytes':8<<30,
 'warmup_iterations':0,'ane_manifest':str(manifest),'allow_approximation':True}}
config={'library':str(a.library.resolve()),'model_path':str(a.model.resolve()),'model_id':'z-image-turbo','constructor':'public','verify_streaming_sources':True}
plan={'scope':__doc__,'config':config,'request':request,'library_sha256':hashlib.sha256(a.library.read_bytes()).hexdigest(),
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
 'export_identity':metadata.get('export_identity'),
 'migration_rule':'Require byte-identical repeated legacy tensors before assessing byte-identical typed stage final latent; preserve failures without loosening tolerance',
 'initial_sha256':hashlib.sha256((fixture/'inputs/initial.npy').read_bytes()).hexdigest(),
 'comparison':'Same partition migration diagnostic; two separately invoked fresh processes; no release or performance qualification'}
(a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
lib=load_native_library(config);engine=None
callback_type=C.CFUNCTYPE(None,C.c_char_p,C.c_void_p)
with (a.output/'events.jsonl').open('w') as events:
 @callback_type
 def event(raw,unused):
  events.write(raw.decode()+'\n');events.flush()
 try:
  engine=create_native_engine(lib,config)
  (a.output/'source-verification.json').write_text(json.dumps(lib._tc_source_verification,indent=2)+'\n')
  result,error=C.c_void_p(),C.c_void_p()
  status=lib.tc_engine_generate(engine,json.dumps(request).encode(),event,None,C.byref(result),C.byref(error))
  value,failure=consume(lib,result),consume(lib,error)
  report={'status':status,'error':failure,'result':json.loads(value) if value else None}
  (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
  if status:raise RuntimeError(failure)
 finally:
  if engine is not None:lib.tc_engine_free(engine)
print('Legacy hybrid reference complete')
