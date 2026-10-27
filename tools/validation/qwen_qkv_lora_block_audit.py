#!/usr/bin/env python3
"""Read-only CPU audit of corrected-QKV whole-block replay artifacts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

from flux9_bf16_framework_screen import write_json
import qwen_qkv_lora_block_screen as screen
from runtime_ane_common import sha256_file

ROOT=Path(__file__).resolve().parents[2]


def tensors(path):
    with path.open("rb") as stream:
        length=struct.unpack("<Q",stream.read(8))[0]
        if not 2<=length<=1<<20:raise ValueError("bounded header required")
        header=json.loads(stream.read(length));result={}
        for name,value in header.items():
            if name=="__metadata__":continue
            shape=value["shape"];a,b=value["data_offsets"]
            if value["dtype"]!="BF16" or shape!=[1,1024,4096] or a<0 or b-a!=1024*4096*2:
                raise ValueError("complete actual BF16 block matrix required")
            stream.seek(8+length+a);payload=stream.read(b-a)
            if len(payload)!=b-a:raise ValueError("truncated raw block")
            result[name]=np.frombuffer(payload,dtype="<u2").reshape(shape).copy()
        return result


def difference(a,b):
    if a.shape!=b.shape:raise ValueError("full output geometry differs")
    x=(a.astype(np.uint32)<<16).view(np.float32).astype(np.float64)
    y=(b.astype(np.uint32)<<16).view(np.float32).astype(np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError("nonfinite complete block")
    delta=x-y;energy=float(np.sum(x*x));other=float(np.sum(y*y))
    return dict(relative_l2=math.sqrt(float(np.sum(delta*delta))/max(energy,1e-30)),
        cosine=1. if energy==0 and other==0 else float(np.sum(x*y))/math.sqrt(max(energy*other,1e-60)),
        max_abs=float(np.max(np.abs(delta))),changed=int(np.count_nonzero(a!=b)),elements=int(a.size))


def same_quality(actual,recorded):
    for key in ("changed","elements"):
        if actual[key]!=recorded[key]:raise ValueError("raw integer quality differs")
    for key in ("relative_l2","cosine","max_abs"):
        if not math.isclose(actual[key],recorded[key],rel_tol=1e-9,abs_tol=1e-10):
            raise ValueError("raw FP64 quality differs: "+key)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cohort",type=Path,action="append",required=True);p.add_argument("--failure",type=Path)
    args=p.parse_args();args.library=args.library.resolve(strict=True)
    if args.output.exists():p.error("fresh audit output required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    for item in manifest["inputs"]["sources"]:
        if sha256_file(ROOT/item["path"])!=item["sha256"]:raise ValueError("current native source differs from linked library")
    rows=[];by_case={};source_hashes=None
    for folder in args.cohort:
        folder=folder.resolve(strict=True);summary=json.loads((folder/"summary.json").read_text());native=json.loads((folder/"trial/native.json").read_text())
        if summary["status"]!="complete" or summary["library_sha256"]!=sha256_file(args.library) or\
                native["loaded_runtime_id"]!=manifest["runtime_build_id"]:raise ValueError("completed same-library cohort required")
        snapshots=json.loads((folder/"source.json").read_text())
        for name,digest in summary["source_sha256"].items():
            if hashlib.sha256(snapshots[name].encode()).hexdigest()!=digest:raise ValueError("historical source snapshot changed")
        actual=screen.analyze(native)
        if actual!=summary["analysis"]:raise ValueError("timing or quality summary could not be reproduced")
        memory=summary["memory"]
        for name in ("evidence","report"):
            if sha256_file(folder/memory[name])!=memory[name+"_sha256"]:raise ValueError("memory artifact changed")
        if not memory["verified"]["complete"] or memory["verified"]["command_exit_code"]!=0:raise ValueError("incomplete process-tree evidence")
        values=tensors(folder/"trial/inputs-reference.safetensors");outputs=tensors(folder/"trial/complete-block-outputs.safetensors")
        input_hash=hashlib.sha256(values["input"].tobytes()).hexdigest();reference=values["block_reference"]
        reference_hash=hashlib.sha256(reference.tobytes()).hexdigest()
        if input_hash!=native["input_sha256"] or reference_hash!=native["ordinary_block_sha256"]:raise ValueError("actual input/reference bytes differ")
        quality=[]
        names=screen.JOINT_NAMES if native.get("qkv_recipe_version") else screen.NAMES
        if set(outputs)!=set(names):raise ValueError("complete per-recipe block outputs required")
        for name in names:
            q=difference(reference,outputs[name]);last=[r for r in native["samples"] if r["arm"]==name][-1]
            same_quality(q,last["quality"]);screen.quality(q,name in screen.NAMES[:2]);quality.append(dict(arm=name,**q))
        for initial in native["initial_quality"]:
            value=tensors(folder/"trial"/(initial["arm"]+"-initial.safetensors"))["block"]
            same_quality(difference(reference,value),initial["quality"])
        if source_hashes is None:source_hashes=native["actual_layer_source_sha256"]
        elif source_hashes!=native["actual_layer_source_sha256"]:raise ValueError("base/factor source changed across tasks or versions")
        key=summary["case"]
        if key in by_case and by_case[key]!=(input_hash,reference_hash):raise ValueError("real task input/reference changed between recipes")
        by_case[key]=(input_hash,reference_hash)
        rows.append(dict(cohort=folder.name,case=key,summary_sha256=sha256_file(folder/"summary.json"),analysis=actual,
            raw_quality=quality,input_sha256=input_hash,reference_sha256=reference_hash,
            memory_peak=memory["verified"]["tree_peak_phys_footprint_bytes"],swap_in=memory["verified"]["swap_in_bytes"],
            swap_out=memory["verified"]["swap_out_bytes"],load_error=summary.get("load_error")))
    failure=None
    if args.failure:
        folder=args.failure.resolve(strict=True);native=json.loads((folder/"trial/native.json").read_text())
        if native["status"]!="failed":raise ValueError("preserved numerical failure required")
        reference=tensors(folder/"trial/gpu-replay-initial.safetensors")["block"]
        candidate=tensors(folder/"trial/fused-b32-initial.safetensors")["block"]
        q=difference(reference,candidate);same_quality(q,native["initial_quality"][-1]["quality"])
        if q["relative_l2"]<=.001:raise ValueError("historical numerical failure no longer reproduced")
        failure=dict(cohort=folder.name,raw_quality=q,speed_claim_allowed=False)
    result=dict(schema="tc-qwen-corrected-qkv-audit-v1",status="complete",qualification_passed=False,
        library_sha256=sha256_file(args.library),runtime_build_id=manifest["runtime_build_id"],
        native_source_hashes_verified=len(manifest["inputs"]["sources"]),actual_layer_hash_count=len(source_hashes or {}),
        original_layer_sources_unchanged_between_tasks=True,cohorts=rows,preserved_failure=failure,
        script_sha256=sha256_file(Path(__file__)),scope="CPU/raw/source audit only; one layer, not full image or physical engine qualification")
    write_json(args.output,result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
