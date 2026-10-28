#!/usr/bin/env python3
"""Independent complete-array, original-payload and fixed-model frontier audit."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import tarfile

import numpy as np
import frontier_collect as screen
import joint_audit
from collect import ROOT,model_snapshot,sha256_file,write_json
from down_audit import read_components
from gpu_dual_request_audit import memory_contract
from qkv_audit import quality
from vpipe_audit import trace


def original_payloads(model):
    path=model/"diffusion_models/qwen_image_2.1_bf16.safetensors"
    with path.open("rb") as stream:
        size=struct.unpack("<Q",stream.read(8))[0]
        if not 2<=size<=16<<20:raise ValueError("bounded original header required")
        header=json.loads(stream.read(size));result={}
        expected={f"transformer_blocks.{l}.attn.to_{p}.weight":(4096,4096) for l in (0,1) for p in ("q","k","v")}
        expected.update({"transformer_blocks.0.img_mlp.gate_up.weight":(24576,4096),"transformer_blocks.0.img_mlp.out.weight":(4096,12288)})
        for name,shape in expected.items():
            info=header[name];begin,end=info["data_offsets"]
            if info["dtype"]!="BF16" or info["shape"]!=list(shape) or end-begin!=int(np.prod(shape))*2:raise ValueError("original dense BF16 payload required")
            stream.seek(8+size+begin);digest=hashlib.sha256();remaining=end-begin
            while remaining:
                data=stream.read(min(4<<20,remaining))
                if not data:raise ValueError("truncated original checkpoint payload")
                digest.update(data);remaining-=len(data)
            result[name]=digest.hexdigest()
    return result


def array_contract(values):
    names={"gpu_a","gpu_b","ffn_input_a","ffn_input_b"}|{f"{arm}_{v}_{part}" for arm in screen.ARMS for v in ("a","b","w") for part in ("block","qkv")}
    if set(values)!=names or any(not np.isfinite(x).all() for x in values.values()):raise ValueError("all forty complete finite arrays required")
    hashes={n:hashlib.sha256(x.astype("<f4",copy=False).tobytes()).hexdigest() for n,x in values.items()}
    anchor="original-post"
    for v in ("a","b","w"):
        for part in ("block","qkv"):
            if any(hashes[f"{a}_{v}_{part}"]!=hashes[f"{anchor}_{v}_{part}"] for a in screen.ARMS):
                raise ValueError("frontier/fusion changed complete incremental output bits")
    if any(hashes[n+"_a"]==hashes[n+"_b"] for n in ("gpu","ffn_input")):
        raise ValueError("changed source activation returned stale full GPU/input")
    if (hashes[anchor+"_a_qkv"] in (hashes[anchor+"_b_qkv"],hashes[anchor+"_w_qkv"]) or
        hashes[anchor+"_a_block"]!=hashes[anchor+"_w_block"]):raise ValueError("changed source weights/input or fixed upstream block failed")
    checks={v:quality(values["gpu_"+v],values[anchor+"_"+v+"_block"]) for v in ("a","b")}
    if any(not 0<=q["relative_l2"]<.03 for q in checks.values()):raise ValueError("real current block exceeds explicit original BF16 screening bound")
    return dict(full_array_sha256=hashes,incremental_bits_exact=True,current_block_quality=checks,
        scope="full BF16 array comparisons and current block vs GPU screen, not independent CPU quantization oracle or natural next-block semantics")


def matched(records):
    anchors={}
    for r in records:
        old=anchors.setdefault(r["case"],r)
        if old["arrays"]["full_array_sha256"]!=r["arrays"]["full_array_sha256"]:raise ValueError("same source repeat/timeline complete outputs changed")
    return dict(same_case_complete_arrays_exact=True,scope="no pooling generation/editing or diagnostic/stock times")


def window(root,probe,native,manifest,payloads,identity,historical):
    s=json.loads((root/"summary.json").read_text());r=json.loads((root/"trial/report.json").read_text())
    if (s["status"]!="complete" or s["assets"]["native_library"]!=sha256_file(native) or s["assets"]["library"]!=sha256_file(probe) or
        s["runtime_build_id"]!=manifest["runtime_build_id"] or r["compiled_native_id"]!=manifest["runtime_build_id"] or r["loaded_runtime_id"]!=manifest["runtime_build_id"] or
        s["model_snapshot"]!=identity or r["source_arrays_sha256"]!=payloads):raise ValueError("complete bound actual native/helper/original source required")
    snapshot=json.loads((root/"source.json").read_text())
    if set(snapshot)!=set(s["source_sha256"]):raise ValueError("complete source snapshot required")
    for name,digest in s["source_sha256"].items():
        raw=snapshot[name].encode()
        if hashlib.sha256(raw).hexdigest()!=digest or not (raw==(ROOT/name).read_bytes() or raw in historical.get(name,set())):raise ValueError("current/archive tool source differs")
    if r["cpp_sha256"]!=s["source_sha256"]["tools/profiling/qwen_base/frontier_probe.mm"]:raise ValueError("compiled helper source differs")
    analysis=screen.validate(r,s["repeats"])
    if analysis!=s["analysis"] or sha256_file(root/"trial/arrays.safetensors")!=s["arrays_sha256"]:raise ValueError("actual timing or complete arrays changed")
    specifications={n:("BF16",(1,1024,12288 if n.endswith("_qkv") else 4096)) for n in
        {"gpu_a","gpu_b","ffn_input_a","ffn_input_b"}|{f"{a}_{v}_{p}" for a in screen.ARMS for v in ("a","b","w") for p in ("block","qkv")}}
    arrays=array_contract(read_components(root/"trial/arrays.safetensors",specifications));observed=events=None
    if s["mode"]=="timeline":
        raw=json.loads((root/"trial/observer.json").read_text())
        if raw["source_sha256"]!=s["source_sha256"]["tools/profiling/qwen_base/observer.mm"]:raise ValueError("compiled observer source differs")
        observed=screen.observer_contract(raw,s["repeats"])
        if observed!=s["observer"]:raise ValueError("actual fixed-model selector evidence differs")
        events=trace(raw)
    return dict(cohort=root.name,case=s["case"],mode=s["mode"],repeats=s["repeats"],analysis=analysis,arrays=arrays,
        memory=memory_contract(root,s),observer=observed,load_error=s.get("load_error"),summary_sha256=sha256_file(root/"summary.json")),events


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("native-library","probe-library","native-source-archive","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--cohort",type=Path,action="append",required=True);p.add_argument("--tool-source-archive",type=Path,action="append",default=[])
    p.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1");args=p.parse_args()
    if args.output.exists():p.error("fresh independent audit required")
    manifest=json.loads((args.native_library.parent/"runtime-build/runtime-build-manifest.json").read_text());joint_audit.archive_contract(args.native_source_archive,manifest)
    historical={}
    for path in args.tool_source_archive:
        with tarfile.open(path,"r:gz") as arc:
            for m in arc.getmembers():
                if m.isfile():historical.setdefault(m.name,set()).add(arc.extractfile(m).read())
    payloads=original_payloads(args.model);identity=model_snapshot(args.model);records=[];traces=[]
    for root in args.cohort:
        r,events=window(root,args.probe_library,args.native_library,manifest,payloads,identity,historical);records.append(r)
        if events:traces.append((root.name,events))
    result=dict(schema="tc-qwen-frontier-audit-v1",status="complete",qualification_passed=False,native_changed=False,
        runtime_build_id=manifest["runtime_build_id"],native_sha256=sha256_file(args.native_library),probe_sha256=sha256_file(args.probe_library),
        native_source_archive_sha256=sha256_file(args.native_source_archive),script_sha256=sha256_file(Path(__file__)),original_payloads=payloads,
        cohorts=records,matched=matched(records),scope="real current block plus dependent synthetic normalization/projection edge, not complete40 image speed or physical INT8/overlap qualification")
    args.output.mkdir(parents=True);write_json(args.output/"audit.json",result)
    for name,events in traces:write_json(args.output/(name+"-trace.json"),events)
    print(json.dumps(dict(status="complete",cohorts=len(records)),indent=2))


if __name__=="__main__":main()
