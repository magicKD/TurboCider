"""Independent NumPy FP32 full causal Transformer reference (all heads, all channels)."""
import numpy as np
from pathlib import Path

def reference(root,m,h,f,layers,heads):
    root=Path(root);state=7;values=[]
    for _ in range(m*h):
        state=(state*1664525+1013904223)&0xffffffff
        values.append((((state>>8)&65535)-32768)/32768*.1)
    x=np.array(values,dtype=np.float16).reshape(m,h).astype(np.float32)
    mask=np.triu(np.ones((m,m),dtype=bool),1);dh=h//heads
    def norm(x):return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)
    for l in range(layers):
        rng=np.random.default_rng(1000+l)
        qkv=(rng.standard_normal((h,3*h))/np.sqrt(h)).astype('float16').astype('float32')
        up=(rng.standard_normal((h,2*f))/np.sqrt(h)).astype('float16').astype('float32')
        down=(rng.standard_normal((f,h))/np.sqrt(f)).astype('float16').astype('float32')
        out=(rng.standard_normal((h,h))/np.sqrt(h)).astype('float16').astype('float32')
        q,k,v=np.split(norm(x)@qkv,3,axis=1)
        q,k,v=[a.reshape(m,heads,dh).transpose(1,0,2) for a in [q,k,v]]
        score=(q@k.transpose(0,2,1))/np.sqrt(dh);score[:,mask]=-np.inf
        prob=np.exp(score-np.max(score,axis=-1,keepdims=True));prob/=prob.sum(axis=-1,keepdims=True)
        x=x+(prob@v).transpose(1,0,2).reshape(m,h)@out
        u,g=np.split(norm(x)@up,2,axis=1);x=x+(u*g/(1+np.exp(-g)))@down
        assert np.isfinite(x).all(), f'nonfinite FP32 reference layer {l}'
        saved=x.astype('float16')
        assert np.isfinite(saved).all(), f'nonfinite serialized reference layer {l}'
        saved.tofile(root/f'reference_L{l+1}.fp16')
