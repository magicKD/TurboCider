"""Recheck sequence-row outlier, compiler placement, and repeated hot timings."""
import json,time,subprocess,shutil
from pathlib import Path
from prepare import prepare
from reference import reference
from inspect_plan import inspect
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'notes/raw/sp_audit';out.mkdir(parents=True,exist_ok=True)
for m,h in [(256,2048),(1024,2048)]:
    cache=ROOT/'cache'/f'sp_audit_{m}_{h}';prepare(cache,m,h,4*h,0,0,3,'row-supergraph',m//2);reference(cache,m,h,4*h,3,h//64)
    try:(out/f'{m}_{h}_plan.json').write_text(json.dumps(inspect(cache/'layer_00/row.mlmodelc'),indent=2))
    except Exception as e:(out/f'{m}_{h}_plan_error.json').write_text(json.dumps({'error':str(e)}))
    for r in range(3):
        cmd=[str(ROOT/'build/transformer'),'--stack-root',str(cache),'--layers','3','--m',str(m),'--h',str(h),'--f',str(4*h),'--heads',str(h//64),'--kv-heads',str(h//64),'--qkv-gpu-n',str(3*h),'--qkv-cpu-n','0','--qkv-ane-n','0','--up-gpu-n',str(8*h),'--up-cpu-n','0','--up-ane-n','0','--ffn-layout','row-supergraph','--row-gpu-m',str(m//2),'--row-ffn-model',str(cache/'layer_00/row.mlmodelc'),'--coreml-output','backing','--attention','steel','--reference-output',str(cache/'reference_L3.fp16'),'--warmup','10','--iterations','40']
        t=time.perf_counter();run=subprocess.run(cmd,cwd='/Users/kd/Documents/project/mac_local_ai',capture_output=True,text=True,timeout=240)
        (out/f'{m}_{h}_r{r}.json').write_text(json.dumps({'command':cmd,'process_s':time.perf_counter()-t,'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr},indent=2));print(m,h,r,run.returncode,flush=True)
    size=sum(p.stat().st_size for p in cache.rglob('*') if p.is_file());shutil.rmtree(cache)
    with (out/'cleanup.jsonl').open('a') as log:log.write(json.dumps({'owned_path':str(cache),'logical_bytes_removed':size,'exists_after':cache.exists()})+'\n')
