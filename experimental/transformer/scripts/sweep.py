"""Serial shape/plan screening with fresh-process paired samples and owned-cache cleanup."""
import argparse,json,os,subprocess,time,shutil,datetime
from pathlib import Path
from prepare import prepare
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--shapes',default='64x256,256x1024,1024x1024,256x2048,1024x2048');p.add_argument('--iterations',type=int,default=30);p.add_argument('--rounds',type=int,default=1);p.add_argument('--tag',default='screen');a=p.parse_args()
out=ROOT/'notes/raw'/a.tag;out.mkdir(parents=True,exist_ok=True)
plans=[('tp25','supergraph',.25,0),('tp50','supergraph',.5,0),('tp75','supergraph',.75,0),('sp50','row-supergraph',0,0),('triple','materialized',.5,.0625),('cpu','materialized',0,.0625),('projection50','segmented-pipeline',.5,0)]
for shape in a.shapes.split(','):
    m,h=map(int,shape.split('x'));f=4*h
    for name,layout,ar,cr in plans:
        job=f'{shape}_{name}';cache=ROOT/'cache'/a.tag/job
        af=int(f*ar)//64*64;cf=int(f*cr)//64*64
        t=time.perf_counter()
        try:
            rec=prepare(cache,m,h,f,af,cf,3,layout,m//2)
            (out/(job+'_prepare.json')).write_text(json.dumps(rec,indent=2))
            for l in [1,2,3]:
                for r in range(a.rounds):
                    cmd=[str(ROOT/'build/transformer'),'--stack-root',str(cache),'--layers',str(l),'--m',str(m),'--h',str(h),'--f',str(f),'--heads',str(h//64),'--kv-heads',str(h//64),'--qkv-gpu-n',str(3*h),'--qkv-cpu-n','0','--qkv-ane-n','0','--up-gpu-n',str(2*(f-af-cf)),'--up-cpu-n',str(2*cf),'--up-ane-n',str(2*af),'--ffn-layout',layout,'--coreml-output','backing','--attention','sdpa','--mode','both','--warmup','8','--iterations',str(a.iterations)]
                    if layout=='row-supergraph':cmd+=['--row-gpu-m',str(m//2),'--row-ffn-model',str(cache/'layer_00/row.mlmodelc')]
                    start=time.perf_counter();run=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
                    record=dict(job=job,layers=l,round=r,command=cmd,returncode=run.returncode,process_s=time.perf_counter()-start,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stdout=run.stdout,stderr=run.stderr)
                    dest=out/f'{job}_L{l}_r{r}.json';dest.write_text(json.dumps(record,indent=2))
                    lines=[json.loads(x) for x in run.stdout.splitlines() if x.startswith('{')]
                    print(job,l,r,'rc',run.returncode,'speedup',lines[-1].get('speedup') if lines else None,flush=True)
        except Exception as e:
            (out/(job+'_error.json')).write_text(json.dumps({'error':str(e)}));print(job,repr(e),flush=True)
        finally:
            size=sum(x.stat().st_size for x in cache.rglob('*') if x.is_file()) if cache.exists() else 0
            shutil.rmtree(cache,ignore_errors=True)
            with (out/'cleanup.jsonl').open('a') as log:log.write(json.dumps({'owned_path':str(cache),'logical_bytes_removed':size,'exists_after':cache.exists(),'elapsed_s':time.perf_counter()-t})+'\n')
