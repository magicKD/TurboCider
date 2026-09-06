"""Real-weight automatic selection, fallback and explicit-route parity. No downloads."""
import argparse, ctypes as C, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
lib=C.CDLL(str(root/'build/native/libturbocider.dylib'))
lib.tc_string_free.argtypes=[C.c_void_p];lib.tc_engine_free.argtypes=[C.c_void_p]
lib.tc_engine_create.argtypes=[C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
common=[C.c_void_p,C.c_char_p,C.c_void_p,C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
lib.tc_engine_generate.argtypes=common;lib.tc_engine_prepare.argtypes=common[:2]+[C.c_int]+common[2:]
def take(p):
 if not p.value:return ''
 s=C.string_at(p).decode();lib.tc_string_free(p);return s
engine,error=C.c_void_p(),C.c_void_p();assert lib.tc_engine_create(a.model.encode(),C.byref(engine),C.byref(error))==0,take(error)
def call(request,prepare=False):
 result,error=C.c_void_p(),C.c_void_p();args=[engine,json.dumps(request).encode()]
 if prepare:args.append(0)
 args.extend([None,None,C.byref(result),C.byref(error)])
 code=(lib.tc_engine_prepare if prepare else lib.tc_engine_generate)(*args);s,e=take(result),take(error)
 if code:raise RuntimeError(e)
 return json.loads(s)
r={'model':'flux2-klein-4b','prompt':'A red fox in a snowy forest.','width':256,'height':256,'steps':4,'seed':42,'execution':'auto','allow_approximation':True,'ane_manifest':a.manifest}
reports={};checks=[]
try:
 reports['prepare']=call(r,True);assert reports['prepare']['execution']=='gpu_ane';checks.append('hardware_checkpoint_and_bucket_match')
 r['output']=str(out/'auto.png');reports['auto']=call(r);assert reports['auto']['hybrid']['calls_session_total']==80
 r.update(execution='gpu_ane',output=str(out/'explicit.png'));reports['explicit']=call(r);assert reports['explicit']['hybrid']['calls_session_total']==160
 sha=lambda name:hashlib.sha256((out/name).read_bytes()).hexdigest()
 assert sha('auto.png')==sha('explicit.png');checks.extend(['hybrid_session_reused','auto_explicit_png_identical'])
 r.update(execution='auto',width=768,height=768);reports['oversize']=call(r,True);assert reports['oversize']['execution']=='gpu' and 'bucket' in reports['oversize']['acceleration_selection'];checks.append('oversize_bucket_falls_back')
 r.update(width=256,height=256,ane_manifest=str(out/'missing.json'));reports['missing']=call(r,True);assert reports['missing']['execution']=='gpu';checks.append('missing_manifest_falls_back')
 malformed=out/'malformed.json';malformed.write_text('{"schema_version":2,"shape":null}')
 r['ane_manifest']=str(malformed);reports['malformed']=call(r,True);assert reports['malformed']['execution']=='gpu';checks.append('malformed_manifest_falls_back')
 r.update(ane_manifest=a.manifest,allow_approximation=False);reports['exact_only']=call(r,True);assert reports['exact_only']['execution']=='gpu';checks.append('approximation_opt_in_required')
 r.update(ane_manifest=str(out/'missing.json'),execution='gpu_ane',allow_approximation=True)
 try:call(r,True);raise AssertionError('explicit hybrid silently fell back')
 except RuntimeError:checks.append('explicit_hybrid_still_strict')
 reports['png_sha256']=sha('auto.png')
 (out/'report.json').write_text(json.dumps({'passed':True,'checks':checks,'reports':reports},indent=2))
 print(json.dumps({'passed':True,'checks':checks}))
finally:lib.tc_engine_free(engine)
