#!/usr/bin/env python3
"""Compare diagnostic safetensors without importing MLX or a model runtime.

N1 numbers are a local numerical gate only, not full media/quality qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np


def load(path):
    with path.open("rb") as source:
        prefix=source.read(8)
        if len(prefix)!=8:raise ValueError("truncated tensor file")
        size=struct.unpack("<Q",prefix)[0]
        if not 0<size<=16<<20:raise ValueError("header exceeds limit")
        header=json.loads(source.read(size)); tensor=header["tensor"]
        begin,end=tensor["data_offsets"]
        if not 0<=begin<=end<=path.stat().st_size-8-size:raise ValueError("invalid tensor range")
        source.seek(8+size+begin);raw=source.read(end-begin)
    if tensor["dtype"]=="F32":values=np.frombuffer(raw,dtype="<f4").astype(np.float64)
    elif tensor["dtype"]=="F16":values=np.frombuffer(raw,dtype="<f2").astype(np.float64)
    elif tensor["dtype"]=="BF16":values=(np.frombuffer(raw,dtype="<u2").astype(np.uint32)<<16).view(np.float32).astype(np.float64)
    else:raise ValueError("unsupported diagnostic dtype")
    if values.size!=int(np.prod(tensor["shape"])) or not np.all(np.isfinite(values)):raise ValueError("invalid shape/nonfinite tensor")
    return values,tensor["shape"],hashlib.sha256(raw).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error("output already exists")
    report={"scope":"diagnostic tensor comparison; not complete model/media qualification","tensors":[]}
    for path in sorted(args.reference.glob("*.safetensors")):
        other=args.candidate/path.name
        a,ashape,ahash=load(other);b,bshape,bhash=load(path)
        if ashape!=bshape:raise ValueError("tensor shapes differ")
        na,nb=np.linalg.norm(a),np.linalg.norm(b)
        report["tensors"].append({"name":path.name,"exact":ahash==bhash,"reference_sha256":bhash,"candidate_sha256":ahash,
            "rel_l2":float(np.linalg.norm(a-b)/max(nb,1e-12)),
            "cosine":float(np.dot(a,b)/max(na*nb,1e-12)) if na or nb else 1.,
            "max_abs":float(np.max(np.abs(a-b)))})
    if not report["tensors"]:raise ValueError("no tensor fixtures")
    final=next(item for item in report["tensors"] if item["name"]=="z_latent_final.safetensors")
    report["n1_final_latent_pass"]=final["rel_l2"]<=.03 and final["cosine"]>=.999
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as out:json.dump(report,out,indent=2,allow_nan=False);out.write("\n")
    print("N1 final latent:",report["n1_final_latent_pass"],final["rel_l2"],final["cosine"])

if __name__=="__main__":main()
