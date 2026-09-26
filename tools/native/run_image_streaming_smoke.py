#!/usr/bin/env python3
"""One verified-source private streaming image request, with preserved raw evidence.

Use collect_streaming_memory.py to observe the complete process lifetime.
This smoke does not qualify a public record or establish performance benefits.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
from run_streaming_campaign import load_native_library, create_native_engine, consume
p=argparse.ArgumentParser(description=__doc__)
for name in ['library','model','request','output']:p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
request=json.loads(a.request.read_text())
assert request['operation']=='image.generate' and len(request['outputs'])==1
assert request['execution']['streaming']['enabled'] is True
request['outputs'][0]['path']=str(a.output/'image.png')
config={'library':str(a.library.resolve()),'model_path':str(a.model.resolve()),'model_id':request['model'],'constructor':'candidate','verify_streaming_sources':True}
plan={'scope':__doc__,'config':config,'request':request,'library_sha256':hashlib.sha256(a.library.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
lib=load_native_library(config);engine=None
callback_type=C.CFUNCTYPE(None,C.c_char_p,C.c_void_p)
with (a.output/'events.jsonl').open('w') as events:
 @callback_type
 def event(raw,unused):events.write(raw.decode()+'\n');events.flush()
 try:
  engine=create_native_engine(lib,config)
  (a.output/'source-verification.json').write_text(json.dumps(lib._tc_source_verification,indent=2)+'\n')
  result,error=C.c_void_p(),C.c_void_p()
  status=lib.tc_engine_generate(engine,json.dumps(request).encode(),event,None,C.byref(result),C.byref(error))
  value,failure=consume(lib,result),consume(lib,error)
  (a.output/'result.json').write_text(json.dumps({'status':status,'error':failure,'result':json.loads(value) if value else None},indent=2)+'\n')
  if status:raise RuntimeError(failure)
 finally:
  if engine is not None:lib.tc_engine_free(engine)
print('streaming image complete')
