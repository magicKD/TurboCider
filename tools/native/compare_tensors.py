"""Compare saved validation tensors on CPU; never changes reference files."""
import argparse
import json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('candidate');p.add_argument('--report',required=True);p.add_argument('--require-exact',action='store_true');a=p.parse_args()
    # safetensors numpy BF16 is not supported in every version; use raw header conversion.
    import struct
    def read(path):
        with open(path,'rb') as f:
            n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));s=h['tensor'];raw=f.read()
        b=raw[s['data_offsets'][0]:s['data_offsets'][1]]
        if s['dtype']=='BF16': x=(np.frombuffer(b,dtype='<u2').astype(np.uint32)<<16).view(np.float32)
        else: x=np.frombuffer(b,dtype={'F32':'<f4','F16':'<f2'}[s['dtype']]).astype(np.float32)
        return x.reshape(s['shape'])
    report={}
    for f in sorted(Path(a.reference).glob('*.safetensors')):
        q=Path(a.candidate)/f.name
        if not q.exists(): report[f.stem]={'missing':True};continue
        x,y=read(f),read(q)
        if x.shape!=y.shape:report[f.stem]={'shape_mismatch':[list(x.shape),list(y.shape)]};continue
        diff=x.astype(np.float64)-y.astype(np.float64)
        report[f.stem]={'shape':list(x.shape),'finite':bool(np.isfinite(x).all() and np.isfinite(y).all()),'identical':bool(np.array_equal(x,y)),'max_abs':float(np.max(np.abs(diff))),'mean_abs':float(np.mean(np.abs(diff))),'rmse':float(np.sqrt(np.mean(diff**2))),'reference_rms':float(np.sqrt(np.mean(x.astype(np.float64)**2)))}
    Path(a.report).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if a.require_exact and (not report or not all(v.get('identical') and v.get('finite') for v in report.values())): raise SystemExit(1)
if __name__=='__main__':main()
