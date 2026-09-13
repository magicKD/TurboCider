"""Generate identical full weights across plans; compile only task-owned artifacts."""
import argparse,json,time,shutil
from pathlib import Path
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

def model(m,w,down=None):
    @mb.program(input_specs=[mb.TensorSpec(shape=(m,w.shape[0]),dtype=types.fp16)],opset_version=ct.target.macOS15)
    def graph(x):
        if down is None:
            return mb.matmul(x=x,y=w,name='y')
        f=w.shape[1]//2
        u=mb.matmul(x=x,y=w[:,:f].copy())
        g=mb.matmul(x=x,y=w[:,f:].copy())
        a=mb.mul(x=u,y=mb.mul(x=g,y=mb.sigmoid(x=g)))
        return mb.matmul(x=a,y=down,name='y')
    return ct.convert(graph,convert_to='mlprogram',minimum_deployment_target=ct.target.macOS15,compute_precision=ct.precision.FLOAT16,skip_model_load=True)

def prepare(root,m,h,f,af,cf,layers,layout,row_gpu):
    root.mkdir(parents=True,exist_ok=True)
    records=[]
    for l in range(layers):
        d=root/f'layer_{l:02d}';d.mkdir(exist_ok=True)
        rng=np.random.default_rng(1000+l)
        # Variance-scaled weights keep each residual branch nontrivial.
        qkv=(rng.standard_normal((h,3*h))/np.sqrt(h)).astype('float16')
        up=(rng.standard_normal((h,2*f))/np.sqrt(h)).astype('float16')
        down=(rng.standard_normal((f,h))/np.sqrt(f)).astype('float16')
        out=(rng.standard_normal((h,h))/np.sqrt(h)).astype('float16')
        def save(name,a):np.ascontiguousarray(a).tofile(d/name)
        save('qkv_gpu.weights.fp16',qkv);save('qkv.bias.fp16',np.zeros(3*h,dtype='float16'))
        save('out.weights.fp16',out);save('down.weights.fp16',down)
        for name in ['norm1','norm2']:save(name+'.weights.fp16',np.ones(h,dtype='float16'))
        gf=f-af-cf
        for device,start,width in [('gpu',0,gf),('cpu',gf,cf),('ane',gf+cf,af)]:
            shard=np.concatenate([up[:,start:start+width],up[:,f+start:f+start+width]],axis=1)
            save('upgate_'+device+'.weights.fp16',shard)
            if device=='ane' and width:
                t=time.perf_counter();mo=model(m,shard,down[start:start+width].copy() if layout=='supergraph' else None)
                package=d/'up.mlpackage';mo.save(str(package));converted=time.perf_counter()-t
                t=time.perf_counter();tmp=Path(ct.models.utils.compile_model(str(package)));shutil.move(str(tmp),d/'up.mlmodelc')
                records.append(dict(layer=l,kind='up',convert_save_s=converted,compile_s=time.perf_counter()-t))
        if layout=='row-supergraph':
            t=time.perf_counter();mo=model(m-row_gpu,up,down);package=d/'row.mlpackage';mo.save(str(package));converted=time.perf_counter()-t
            t=time.perf_counter();tmp=Path(ct.models.utils.compile_model(str(package)));shutil.move(str(tmp),d/'row.mlmodelc')
            records.append(dict(layer=l,kind='row',convert_save_s=converted,compile_s=time.perf_counter()-t))
    (root/'preparation.json').write_text(json.dumps(records,indent=2))
    return records
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    for k,v in [('m',64),('h',256),('f',1024),('af',512),('cf',0),('layers',3),('row-gpu',32)]:p.add_argument('--'+k,type=int,default=v)
    p.add_argument('--layout',default='supergraph');a=p.parse_args()
    prepare(a.root,a.m,a.h,a.f,a.af,a.cf,a.layers,a.layout,a.row_gpu)
