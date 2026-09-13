"""Confirm tiled GPU layout bridge and dimension/placement ablations."""
import json,os,time,subprocess,shutil
from pathlib import Path
from prepare import prepare
from private_mil import private_mil
from reference import reference
from inspect_plan import inspect
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'notes/raw/bridge';out.mkdir(parents=True,exist_ok=True)
for m,h in [(256,1024),(1024,1024),(256,2048),(1024,2048)]:
    f=4*h;af=f//2;cache=ROOT/'cache'/f'bridge_{m}_{h}'
    prep=prepare(cache,m,h,f,af,0,3,'supergraph',m//2);private_mil(cache,m,h,f,af,3)
    try:(out/f'{m}_{h}_plan.json').write_text(json.dumps(inspect(cache/'layer_00/up.mlmodelc'),indent=2))
    except Exception as e:(out/f'{m}_{h}_plan_error.json').write_text(json.dumps({'error':str(e)}))
    for dh in ([32,64,128] if (m,h)==(256,1024) else [64]):
        reference(cache,m,h,f,3,h//dh)
        for l in [1,2,3]:
            for r in range(3):
                variants=[('public',False),('private_gpu_bridge',True),('public_serial',False)]
                if r%2:variants.reverse()
                for name,private in variants:
                    cmd=[str(ROOT/'build/transformer-private-gpu'),'--stack-root',str(cache),'--layers',str(l),'--m',str(m),'--h',str(h),'--f',str(f),'--heads',str(h//dh),'--kv-heads',str(h//dh),'--qkv-gpu-n',str(3*h),'--qkv-cpu-n','0','--qkv-ane-n','0','--up-gpu-n',str(2*(f-af)),'--up-cpu-n','0','--up-ane-n',str(2*af),'--ffn-layout','supergraph','--coreml-output','backing','--attention','steel' if dh==64 else 'sdpa','--reference-output',str(cache/f'reference_L{l}.fp16'),'--warmup','10','--iterations','40']
                    env=os.environ.copy()
                    if private:env['TC_PRIVATE_FFN']='1'
                    if name=='public_serial':env['TC_SERIAL_BRANCHES']='1'
                    t=time.perf_counter();run=subprocess.run(cmd,env=env,cwd='/Users/kd/Documents/project/mac_local_ai',capture_output=True,text=True,timeout=240)
                    (out/f'{m}_{h}_d{dh}_L{l}_{name}_r{r}.json').write_text(json.dumps({'command':cmd,'private':private,'process_s':time.perf_counter()-t,'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr},indent=2))
                    lines=[json.loads(x) for x in run.stdout.splitlines() if x.startswith('{')]
                    print(m,h,dh,l,r,name,run.returncode,lines[-1].get('speedup') if lines else run.stderr[-300:],flush=True)
                    if run.returncode:raise RuntimeError('bridge failure; inspect before continuing')
    size=sum(p.stat().st_size for p in cache.rglob('*') if p.is_file());shutil.rmtree(cache)
    with (out/'cleanup.jsonl').open('a') as log:log.write(json.dumps({'owned_path':str(cache),'logical_bytes_removed':size,'exists_after':cache.exists()})+'\n')
