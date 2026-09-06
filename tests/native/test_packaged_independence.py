"""Run after copying this file + dist/cli outside the checkout, under read-denial sandbox."""
import argparse,ctypes as C,hashlib,json,statistics,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--binary',required=True);p.add_argument('--model',required=True);p.add_argument('--manifest',required=True);p.add_argument('--profile',required=True);p.add_argument('--output',required=True);p.add_argument('--blocked',action='append',required=True);a=p.parse_args()
binary=Path(a.binary).resolve();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);checks=[];reports={}
for file in a.blocked:
 try:Path(file).read_bytes();raise AssertionError('sandbox did not deny '+file)
 except PermissionError:pass
checks.append('checkout_and_sibling_reads_denied')
def command(value):
 f=out/'request.json';f.write_text(json.dumps(value));r=subprocess.run([str(binary),'coreml',str(f)],capture_output=True,text=True);assert r.returncode==0,r.stderr;return json.loads(r.stdout)
reports['export_reuse']=command({'action':'export','model_root':a.model,'profile':a.profile});checks.append('packaged_exporter_uses_managed_toolchain_without_checkout')
reports['compile']=command({'action':'compile','profile':a.profile,'source_manifest':reports['export_reuse']['source_manifest']});assert reports['compile']['cache_hits']==20;checks.append('owned_compiled_cache_reused')
lib=C.CDLL(str(binary.parent/'libturbocider.dylib'));lib.tc_string_free.argtypes=[C.c_void_p];lib.tc_engine_free.argtypes=[C.c_void_p]
lib.tc_engine_create.argtypes=[C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
lib.tc_engine_generate.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p,C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
def take(p):
 if not p.value:return ''
 s=C.string_at(p).decode();lib.tc_string_free(p);return s
engine,error=C.c_void_p(),C.c_void_p();assert lib.tc_engine_create(a.model.encode(),C.byref(engine),C.byref(error))==0,take(error)
def generate(r):
 result,error=C.c_void_p(),C.c_void_p();start=time.monotonic();code=lib.tc_engine_generate(engine,json.dumps(r).encode(),None,None,C.byref(result),C.byref(error));text,message=take(result),take(error);assert code==0,message;return {'wall':time.monotonic()-start,'metrics':json.loads(text)}
try:
 r={'model':'flux2-klein-4b','prompt':'A red fox in a snowy forest.','width':256,'height':256,'steps':4,'seed':42,'execution':'gpu_ane','allow_approximation':True,'ane_manifest':a.manifest,'output':str(out/'golden.png')}
 reports['golden']=generate(r);sha=hashlib.sha256((out/'golden.png').read_bytes()).hexdigest();assert sha=='26eed24a400cf453183e0850d46d24e987dce1a3ed3fb3110fd7c898d3d055d7',sha;checks.append('managed_toolchain_golden_png_identical')
 reports['png_sha256']=sha
 r.update(width=512,height=512,prompt='A red fox sitting in a snowy forest, soft morning light, detailed photography.')
 for mode in ['gpu','gpu_ane']:
  runs=[]
  for i in range(4):r.update(execution=mode,output=str(out/f'{mode}-{i}.png'));runs.append(generate(r))
  reports[mode]={'runs':runs,'warm_median_seconds':statistics.median(x['wall'] for x in runs[1:])}
 checks.extend(['gpu_inference_without_external_code','hybrid_inference_without_external_artifacts'])
finally:lib.tc_engine_free(engine)
reports['inventory']=command({'action':'inventory','manifest':a.manifest});checks.append('owned_resource_inventory')
(out/'report.json').write_text(json.dumps({'passed':True,'checks':checks,'reports':reports},indent=2));print(json.dumps({'passed':True,'checks':checks}))
