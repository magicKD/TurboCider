"""Complete attention-head partition; includes global causal softmax per head."""
import json,time,subprocess,shutil
from pathlib import Path
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from prepare import prepare
from reference import reference
ROOT=Path(__file__).resolve().parents[1]
def head_model(m,h,heads,count):
    dh=h//heads;offset=h-count*dh
    @mb.program(input_specs=[mb.TensorSpec(shape=(m,3*h),dtype=types.fp16)],opset_version=ct.target.macOS15)
    def att(x):
        parts=[]
        for i in range(3):
            y=mb.slice_by_index(x=x,begin=[0,i*h+offset],end=[m,(i+1)*h])
            y=mb.reshape(x=y,shape=[m,count,dh]);parts.append(mb.transpose(x=y,perm=[1,0,2]))
        q,k,v=parts
        s=mb.matmul(x=q,y=k,transpose_y=True);s=mb.mul(x=s,y=np.float16(1/np.sqrt(dh)))
        mask=np.where(np.triu(np.ones((m,m),dtype=bool),1),-np.inf,0).astype('float16')
        s=mb.add(x=s,y=mask);prob=mb.softmax(x=s,axis=-1)
        c=mb.matmul(x=prob,y=v);c=mb.transpose(x=c,perm=[1,0,2])
        return mb.reshape(x=c,shape=[m,count*dh],name='y')
    return ct.convert(att,convert_to='mlprogram',minimum_deployment_target=ct.target.macOS15,compute_precision=ct.precision.FLOAT16,skip_model_load=True)
if __name__=='__main__':
    out=ROOT/'notes/raw/heads';out.mkdir(parents=True,exist_ok=True)
    for m,h in [(64,256),(256,1024),(1024,1024)]:
        cache=ROOT/'cache'/f'heads_{m}_{h}';prepare(cache,m,h,4*h,0,0,3,'materialized',m//2)
        reference(cache,m,h,4*h,3,h//64)
        for fraction in [.25,.5]:
            count=max(1,int(h//64*fraction));package=cache/f'head{count}.mlpackage'
            t=time.perf_counter();head_model(m,h,h//64,count).save(str(package));comp=Path(ct.models.utils.compile_model(str(package)));target=cache/f'head{count}.mlmodelc';shutil.move(str(comp),target)
            (out/f'{m}_{h}_{count}_compile.json').write_text(json.dumps({'convert_compile_s':time.perf_counter()-t}))
            for backend in ['cpu','ane']:
                for l in [1,2,3]:
                    cmd=[str(ROOT/'build/transformer-heads'),'--stack-root',str(cache),'--layers',str(l),'--m',str(m),'--h',str(h),'--f',str(4*h),'--heads',str(h//64),'--kv-heads',str(h//64),'--qkv-gpu-n',str(3*h),'--qkv-cpu-n','0','--qkv-ane-n','0','--up-gpu-n',str(8*h),'--up-cpu-n','0','--up-ane-n','0','--attention','materialized','--head-suffix',str(count),'--head-backend',backend,'--head-model',str(target),'--reference-output',str(cache/f'reference_L{l}.fp16'),'--warmup','8','--iterations','30']
                    t=time.perf_counter();r=subprocess.run(cmd,text=True,capture_output=True,timeout=240)
                    (out/f'{m}_{h}_{count}_{backend}_L{l}.json').write_text(json.dumps({'command':cmd,'process_s':time.perf_counter()-t,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2))
                    print(m,h,count,backend,l,r.returncode,flush=True)
        size=sum(p.stat().st_size for p in cache.rglob('*') if p.is_file());shutil.rmtree(cache)
        with (out/'cleanup.jsonl').open('a') as log:log.write(json.dumps({'owned_path':str(cache),'logical_bytes_removed':size,'exists_after':cache.exists()})+'\n')
