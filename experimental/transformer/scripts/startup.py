"""Explicit compile/load/first-prediction phase accounting; not a cache-flushed cold-start claim."""
import json,os,time,subprocess,shutil
from pathlib import Path
from prepare import prepare
from private_mil import private_mil
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'notes/raw/startup';out.mkdir(parents=True,exist_ok=True)
for m,h in [(256,1024),(1024,2048)]:
    f=4*h;af=f//2;cache=ROOT/'cache'/f'startup_{m}_{h}';prep=prepare(cache,m,h,f,af,0,1,'supergraph',m//2);private_mil(cache,m,h,f,af,1)
    (out/f'{m}_{h}_export.json').write_text(json.dumps(prep,indent=2))
    for r in range(3):
        for name in (['public','private'] if r%2==0 else ['private','public']):
            cmd=[str(ROOT/'build/transformer-startup'),'--stack-root',str(cache),'--layers','1','--m',str(m),'--h',str(h),'--f',str(f),'--heads',str(h//64),'--kv-heads',str(h//64),'--qkv-gpu-n',str(3*h),'--qkv-cpu-n','0','--qkv-ane-n','0','--up-gpu-n',str(4*h),'--up-cpu-n','0','--up-ane-n',str(4*h),'--ffn-layout','supergraph','--coreml-output','backing','--attention','steel','--warmup','10','--iterations','40']
            env=os.environ.copy()
            if name=='private':env['TC_PRIVATE_FFN']='1'
            t=time.perf_counter();run=subprocess.run(cmd,cwd='/Users/kd/Documents/project/mac_local_ai',env=env,capture_output=True,text=True,timeout=240)
            (out/f'{m}_{h}_{name}_r{r}.json').write_text(json.dumps({'command':cmd,'process_s':time.perf_counter()-t,'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr},indent=2));print(m,h,name,r,run.returncode,flush=True)
    size=sum(p.stat().st_size for p in cache.rglob('*') if p.is_file());shutil.rmtree(cache)
    with (out/'cleanup.jsonl').open('a') as log:log.write(json.dumps({'owned_path':str(cache),'logical_bytes_removed':size,'exists_after':cache.exists()})+'\n')
