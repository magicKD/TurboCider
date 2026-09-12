"""Matched three-device CPU-tail search: same materialized FFN schedule, varying only CPU width."""
import json,time,subprocess,shutil
from pathlib import Path
from prepare import prepare
from reference import reference
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'notes/raw/triple_audit';out.mkdir(parents=True,exist_ok=True)
for m,h in [(256,2048),(1024,2048)]:
    f=4*h;af=f//2
    for cf in [0,128,256,512]:
        cache=ROOT/'cache'/f'triple_audit_{m}_{h}_{cf}';prepare(cache,m,h,f,af,cf,3,'materialized',m//2);reference(cache,m,h,f,3,h//64)
        for r in range(3):
            cmd=[str(ROOT/'build/transformer'),'--stack-root',str(cache),'--layers','3','--m',str(m),'--h',str(h),'--f',str(f),'--heads',str(h//64),'--kv-heads',str(h//64),'--qkv-gpu-n',str(3*h),'--qkv-cpu-n','0','--qkv-ane-n','0','--up-gpu-n',str(2*(f-af-cf)),'--up-cpu-n',str(2*cf),'--up-ane-n',str(2*af),'--ffn-layout','materialized','--coreml-output','backing','--attention','steel','--reference-output',str(cache/'reference_L3.fp16'),'--warmup','10','--iterations','40']
            t=time.perf_counter();run=subprocess.run(cmd,cwd='/Users/kd/Documents/project/mac_local_ai',capture_output=True,text=True,timeout=240)
            (out/f'{m}_{h}_cpu{cf}_r{r}.json').write_text(json.dumps({'command':cmd,'process_s':time.perf_counter()-t,'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr},indent=2));print(m,h,cf,r,run.returncode,flush=True)
        size=sum(p.stat().st_size for p in cache.rglob('*') if p.is_file());shutil.rmtree(cache)
        with (out/'cleanup.jsonl').open('a') as log:log.write(json.dumps({'owned_path':str(cache),'logical_bytes_removed':size,'exists_after':cache.exists()})+'\n')
