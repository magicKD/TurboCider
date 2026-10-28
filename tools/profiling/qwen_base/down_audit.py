#!/usr/bin/env python3
"""Full-array/payload and sampled F64 oracle audit for original-pitch GPU down."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import tarfile

import numpy as np

import handoff_audit
import ffn_projection_audit
import joint_audit
from collect import ROOT,model_snapshot,sha256_file,write_json
from qkv_audit import quality


def read_components(path,specifications):
    """Keep BF16 sources and F32 partials typed; never relax the BF16 reader."""
    with path.open("rb") as stream:
        size=struct.unpack("<Q",stream.read(8))[0]
        if not 2<=size<=16<<20:raise ValueError("bounded component tensor header required")
        header=json.loads(stream.read(size))
    entries={k:v for k,v in header.items() if k!="__metadata__"}
    if set(entries)!=set(specifications):raise ValueError("exact complete component tensor set required")
    result={};extents=[];data_bytes=path.stat().st_size-8-size
    for name,(dtype,shape) in specifications.items():
        info=entries[name];item=2 if dtype=="BF16" else 4 if dtype=="F32" else 0
        begin,end=info["data_offsets"]
        if (not item or info["dtype"]!=dtype or info["shape"]!=list(shape) or type(begin) is not int or type(end) is not int or
                not 0<=begin<=end<=data_bytes or begin%item or end-begin!=math.prod(shape)*item):
            raise ValueError("actual BF16-hidden/F32-output type, shape or physical extent differs")
        value=np.memmap(path,mode="r",dtype="<u2" if dtype=="BF16" else "<f4",offset=8+size+begin,shape=shape)
        result[name]=(value.astype(np.uint32)<<16).view(np.float32) if dtype=="BF16" else value
        extents.append((begin,end))
    cursor=0
    for begin,end in sorted(extents):
        if begin!=cursor:raise ValueError("overlapping or incomplete component payload")
        cursor=end
    if cursor!=data_bytes:raise ValueError("extra component payload bytes")
    return result


def sampled_dot(weights,hidden,result,coordinates):
    """Independent F64 dot at fixed complete-K witnesses; not a full CPU GEMM."""
    if (weights.dtype!=np.dtype("uint16") or hidden.dtype!=np.dtype("float32") or result.dtype!=np.dtype("float32") or
            weights.ndim!=2 or hidden.ndim!=3 or hidden.shape[0]!=1 or result.ndim!=3 or result.shape[0]!=1 or
            weights.shape[1]<hidden.shape[2] or result.shape[1:]!=(hidden.shape[1],weights.shape[0])):
        raise ValueError("original physical weight extent and complete logical-K arrays required")
    rows=[];energy=error=0.
    for m,n in coordinates:
        if not (type(m) is int and type(n) is int and 0<=m<hidden.shape[1] and 0<=n<weights.shape[0]):
            raise ValueError("bounded explicit oracle coordinates required")
        bits=np.asarray(weights[n,:hidden.shape[2]],dtype=np.uint32)<<16
        w=bits.view(np.float32).astype(np.float64)
        x=hidden[0,m].astype(np.float64)
        expected=float(np.dot(x,w));actual=float(result[0,m,n]);delta=actual-expected
        if not math.isfinite(expected) or not math.isfinite(actual) or abs(delta)>5e-4*max(1.,abs(expected)):
            raise ValueError("sampled full-K F64 oracle failed, including original physical row pitch")
        rows.append(dict(row=m,column=n,expected_f64=expected,actual=actual,absolute_error=abs(delta)))
        energy+=expected*expected;error+=delta*delta
    if not rows:raise ValueError("actual sampled full-K oracle witnesses required")
    relative=math.sqrt(error/max(energy,1e-30))
    if relative>=1e-4:raise ValueError("sampled F64 relative error exceeds fixed bound")
    return dict(points=len(rows),relative_l2=relative,max_abs=max(r["absolute_error"] for r in rows),witnesses=rows,
        scope="deterministic sampled output coordinates, complete K per dot; not a full CPU dense FFN oracle")


def original_down(model):
    path=model/"diffusion_models/qwen_image_2.1_bf16.safetensors"
    with path.open("rb") as stream:
        size=struct.unpack("<Q",stream.read(8))[0]
        if not 2<=size<=16<<20:raise ValueError("bounded original checkpoint header required")
        header=json.loads(stream.read(size));info=header["transformer_blocks.0.img_mlp.out.weight"]
    if info["dtype"]!="BF16" or info["shape"]!=[4096,12288] or info["data_offsets"][1]-info["data_offsets"][0]!=4096*12288*2:
        raise ValueError("original full physical BF16 down matrix required")
    return np.memmap(path,dtype="<u2",mode="r",offset=8+size+info["data_offsets"][0],shape=(4096,12288))


def component_arrays(root,record,weights):
    summary=json.loads((root/"summary.json").read_text());report=json.loads((root/"trial/report.json").read_text())
    path=root/"trial/down-arrays.safetensors"
    if sha256_file(path)!=summary.get("down_arrays_sha256"):raise ValueError("complete down component arrays changed")
    expected={"hidden_a","hidden_b","down_a","down_b","control_a","control_b"}
    half=record["gate_up"]=="fp16"
    if half:expected|={"ordinary_hidden_a","ordinary_hidden_b"}
    channels=record["channels"]
    specifications={name:("BF16",(1,1024,12288-channels)) if "hidden" in name else ("F32",(1,1024,4096)) for name in expected}
    values=read_components(path,specifications)
    if any(not np.isfinite(v).all() for v in values.values()):
        raise ValueError("all complete finite component arrays required")
    checks={};oracles={}
    coordinates=[(0,0),(1023,4095),(0,4095),(1023,0)]+[((i*137+11)%1024,(i*977+31)%4096) for i in range(28)]
    for variant in ("a","b"):
        hidden=values["hidden_"+variant];down=values["down_"+variant];control=values["control_"+variant]
        if hidden.shape!=(1,1024,12288-channels) or down.shape!=(1,1024,4096) or control.shape!=down.shape:
            raise ValueError("actual full captured component geometry differs")
        q=quality(control,down);checks["down_"+variant]=q
        if any(not math.isclose(q[k],v,rel_tol=1e-8,abs_tol=1e-10) for k,v in report["down_quality_"+variant].items()):
            raise ValueError("independently recomputed full down quality differs")
        oracles["down_"+variant]=sampled_dot(weights,hidden,down,coordinates)
        oracles["control_"+variant]=sampled_dot(weights,hidden,control,coordinates)
        if half:
            ordinary=values["ordinary_hidden_"+variant]
            if ordinary.shape!=hidden.shape:raise ValueError("complete ordinary BF16 GPU gate/up control required")
            q=quality(ordinary,hidden);checks["gate_up_"+variant]=q
            if any(not math.isclose(q[k],v,rel_tol=1e-8,abs_tol=1e-10) for k,v in report["gate_up_quality_"+variant].items()):
                raise ValueError("independently recomputed full GPU gate/up quality differs")
    hashes={name:hashlib.sha256(value.astype("<f4",copy=False).tobytes(order="C")).hexdigest() for name,value in values.items()}
    return dict(quality=checks,sampled_f64_oracles=oracles,array_sha256=hashes)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("native-library","native-source-archive","output"):p.add_argument("--"+name,type=Path,required=True)
    for name in ("probe-dir","cohort"):p.add_argument("--"+name,type=Path,action="append",required=True)
    p.add_argument("--tool-source-archive",type=Path,action="append",default=[])
    p.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1");args=p.parse_args()
    if args.output.exists():p.error("fresh audit directory required")
    manifest=json.loads((args.native_library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    joint_audit.archive_contract(args.native_source_archive,manifest)
    historical={}
    for path in args.tool_source_archive:
        with tarfile.open(path,"r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():historical.setdefault(member.name,set()).add(archive.extractfile(member).read())
    identity=model_snapshot(args.model);payloads=ffn_projection_audit.original_hashes(args.model);weights=original_down(args.model)
    records=[]
    for root in args.cohort:
        record=handoff_audit.window(root,args.native_library,args.probe_dir,manifest,identity,payloads,historical)
        if record["down_policy"]=="none":raise ValueError("explicit real F32 down screen required")
        record["down_components"]=component_arrays(root,record,weights);records.append(record)
    controls=handoff_audit.matched(records)
    seen={}
    for record in records:
        key=(record["channels"],record["down_policy"],record["gate_up"])
        if key in seen and record["down_components"]["array_sha256"]!=seen[key]:raise ValueError("same recipe did not restore all component arrays")
        seen.setdefault(key,record["down_components"]["array_sha256"])
    args.output.mkdir(parents=True)
    result=dict(schema="tc-qwen-f32-down-audit-v1",status="complete",qualification_passed=False,cohorts=records,matched=controls,
        native_library_sha256=sha256_file(args.native_library),native_source_archive_sha256=sha256_file(args.native_source_archive),
        script_sha256=sha256_file(Path(__file__)),original_ffn_payload_sha256=payloads,
        tool_source_archive_sha256={str(x):sha256_file(x) for x in args.tool_source_archive},
        scope="Real first-decode block and complete component arrays; sampled full-K F64 dots are not a full CPU oracle. No full forty-step image throughput/semantic qualification.")
    write_json(args.output/"audit.json",result)
    print(json.dumps(dict(status="complete",cohorts=len(records),matched=controls),indent=2))


if __name__=="__main__":main()
