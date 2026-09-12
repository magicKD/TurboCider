"""Check FP32 reference against FP64 direct einsum, independent of BLAS matmul."""
from pathlib import Path
import numpy as np,json,tempfile
from reference import reference
ROOT=Path(__file__).resolve().parents[1]
m,h,f,heads=16,64,256,4
with tempfile.TemporaryDirectory(prefix='tc-reference-') as tmp:
    reference(tmp,m,h,f,3,heads)
    state=7;vals=[]
    for _ in range(m*h):state=(state*1664525+1013904223)&0xffffffff;vals.append((((state>>8)&65535)-32768)/32768*.1)
    x=np.array(vals,dtype=np.float16).reshape(m,h).astype(np.float64)
    def mm(a,b):return np.einsum('...ik,...kj->...ij',a,b,optimize=False)
    def norm(a):return a/np.sqrt((a*a).mean(-1,keepdims=True)+1e-6)
    metrics=[]
    for l in range(3):
        rng=np.random.default_rng(1000+l)
        qkv=(rng.standard_normal((h,3*h))/np.sqrt(h)).astype('float16').astype('float64');up=(rng.standard_normal((h,2*f))/np.sqrt(h)).astype('float16').astype('float64');down=(rng.standard_normal((f,h))/np.sqrt(f)).astype('float16').astype('float64');out=(rng.standard_normal((h,h))/np.sqrt(h)).astype('float16').astype('float64')
        q,k,v=[a.reshape(m,heads,h//heads).transpose(1,0,2) for a in np.split(mm(norm(x),qkv),3,axis=1)]
        score=mm(q,k.transpose(0,2,1))/np.sqrt(h//heads);score[:,np.triu(np.ones((m,m),bool),1)]=-np.inf;p=np.exp(score-score.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
        x=x+mm(mm(p,v).transpose(1,0,2).reshape(m,h),out);u,g=np.split(mm(norm(x),up),2,axis=1);x=x+mm(u*g/(1+np.exp(-g)),down)
        stored=np.fromfile(Path(tmp)/f'reference_L{l+1}.fp16',dtype='float16').reshape(m,h).astype('float64')
        assert np.isfinite(stored).all() and np.isfinite(x).all()
        err=np.linalg.norm(stored-x)/np.linalg.norm(x);assert err<.001
        metrics.append({'layer':l+1,'reference_fp32_saved_fp16_vs_direct_fp64_relative_l2':err,'finite':True})
(ROOT/'notes/raw/reference_validation.json').write_text(json.dumps({'shape':[m,h,f],'heads':heads,'independent_algorithm':'np.einsum optimize=False FP64; comparison includes FP16 reference serialization','metrics':metrics},indent=2))
print(metrics)
