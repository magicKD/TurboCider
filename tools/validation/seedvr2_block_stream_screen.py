#!/usr/bin/env python3
"""Serial complete-context DiT weight-residency screen, not App qualification.

Uses the original checkpoint and held-fd source verification. Every tensor/read
range and every complete velocity/Euler word is checked, without temporal crops.
"""
import argparse
import bisect
import collections
import json
import math
from pathlib import Path
import subprocess
import sys

from seedvr2_framework_screen import sha256,write_json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"tools/coreml"))
from export_ltx_gemma_mlp import read_header

CHECKPOINT_SHA="98669fd2c06df5eca88baf68cd5c478775c8e61fc110e598c52b350145ea2660"
SHARED_ROPE="blocks.0.attn.rope.rope.freqs"


def weight_group(name):
    if name==SHARED_ROPE or not name.startswith("blocks."):return -1
    pieces=name.split(".",2)
    if len(pieces)!=3 or not pieces[1].isdigit() or str(int(pieces[1]))!=pieces[1] or not 0<=int(pieces[1])<32:
        raise ValueError("noncanonical SeedVR2 block group")
    return int(pieces[1])


def validate_weight_reads(value,offset,header):
    records=[(offset+v["data_offsets"][0],offset+v["data_offsets"][1],weight_group(k),k,v)
        for k,v in header.items() if k!="__metadata__"]
    records.sort(key=lambda row:row[0]);starts=[row[0] for row in records]
    expected=collections.Counter();actual=collections.Counter();compute_bytes=0
    for lo,hi,group,name,record in records:
        expected[group]+=hi-lo
        compute_bytes+=(hi-lo)*2 if name.endswith(".rope.freqs") else hi-lo
    if (value.get("schema")!="seedvr2-block-weight-parity-v1" or value.get("status")!="complete" or
        value.get("qualified") is not False or value.get("all_words_equal") is not True or value.get("zero_active_teardown") is not True or
        value.get("tensors")!=len(records) or value.get("compared_compute_bytes")!=compute_bytes or
        value.get("original_source_payload_bytes")!=sum(expected.values()) or set(expected)!=set(range(-1,32))):
        raise ValueError("incomplete original coefficient/group coverage")
    for read in value.get("reads",[]):
        if any(type(read.get(k)) is not int for k in ("offset","bytes","group")) or read["bytes"]<=0:
            raise ValueError("invalid positional read")
        index=bisect.bisect_right(starts,read["offset"])-1
        if index<0:raise ValueError("payload read entered checkpoint header")
        lo,hi,group,_,_=records[index]
        if read["offset"]<lo or read["offset"]+read["bytes"]>hi or group!=read["group"]:
            raise ValueError("read outside the actual selected tensor/group")
        actual[group]+=read["bytes"]
    # The native test explicitly repeats block31 after pre-cancel/recovery.
    if any(actual[g]!=size*(2 if g==31 else 1) for g,size in expected.items()) or set(actual)!=set(expected):
        raise ValueError("selected group read coverage differs")
    return dict(tensors=len(records),source_payload_bytes=sum(expected.values()),compute_bytes=compute_bytes,
        global_source_bytes=expected[-1],forward_source_bytes=sum(expected[g] for g in range(32)),
        read_ranges_checked=len(value["reads"]),all_reads_in_selected_groups=True)


def validate_forward(value,route,shape,repeats,forward_bytes):
    if (value.get("schema")!="seedvr2-block-stream-forward-v1" or value.get("status")!="complete" or value.get("route")!=route or
        value.get("qualified") is not False or value.get("zero_active_teardown") is not True or
        (value.get("frames_latent"),value.get("height_latent"),value.get("width_latent"))!=tuple(shape[2:]) or
        len(value.get("iterations",[]))!=repeats or value.get("sampled_process_footprint_peak_bytes",0)<=0):
        raise ValueError("incomplete full-context block forward")
    for index,row in enumerate(value["iterations"]):
        if (row.get("iteration")!=index or row.get("warm")!=(index>0) or row.get("blocks_completed")!=32 or
            row.get("streamed_blocks")!=(32 if route=="stream" else 0) or
            row.get("source_payload_bytes")!=(forward_bytes if route=="stream" else 0) or
            row.get("mlx_active_peak_bytes",0)<=0 or not math.isfinite(row.get("forward_seconds",float("nan"))) or row["forward_seconds"]<=0):
            raise ValueError("missing block execution/source IO coverage")
    if value.get("total_streamed_blocks")!=(32*repeats if route=="stream" else 0):raise ValueError("stream block count differs")
    if route=="stream" and value.get("peak_parameter_bytes",float("inf"))>=500<<20:
        raise ValueError("block stream retained more than global/current-block parameters")


def compare_words(reference,candidate):
    import numpy as np
    if reference.shape!=candidate.shape or reference.dtype!=candidate.dtype or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("nonfinite or changed complete DiT output geometry/precision")
    if not np.array_equal(reference.view(np.uint8),candidate.view(np.uint8)):
        raise ValueError("block-stream velocity/Euler words differ from original whole-bank route")
    return dict(shape=list(reference.shape),dtype=str(reference.dtype),bytes=reference.nbytes,all_words_equal=True)


def validate_lifecycle(value):
    required=("pre_cancel_checked","mid_read_cancel_typed","after_block_cancel_checked","same_model_recovery_words_equal",
        "non_cancel_read_error_latched","duplicate_group_atomic","wrong_group_rejected","source_mutation_rejected",
        "source_replacement_rejected","unverified_source_rejected","fresh_source_recovery","zero_active_teardown")
    if (value.get("schema")!="seedvr2-block-stream-lifecycle-v1" or value.get("status")!="complete" or
        value.get("qualified") is not False or any(value.get(key) is not True for key in required) or
        type(value.get("mid_read_cancelled_payload_bytes")) is not int or value["mid_read_cancelled_payload_bytes"]<=0 or
        value.get("completed_blocks")!=66):
        raise ValueError("incomplete actual-read/block cancellation, error latch or source recovery gate")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("probe","model","input","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--condition-fixture",type=Path)
    parser.add_argument("--frozen-reference",type=Path)
    parser.add_argument("--lifecycle-probe",type=Path)
    parser.add_argument("--repeats",type=int,default=2)
    args=parser.parse_args()
    if args.output.exists() or not 1<=args.repeats<=5:parser.error("fresh output and one...five fresh iterations required")
    args.lifecycle_probe=args.lifecycle_probe or args.probe.with_name("seedvr2-block-stream-test")
    for key in ("probe","model","input","condition_fixture","frozen_reference","lifecycle_probe"):
        if getattr(args,key):setattr(args,key,getattr(args,key).resolve(strict=True))
    import os
    diagnostics=[key for key in os.environ if key.startswith("TURBOCIDER_SEEDVR2_DIT_")]
    if diagnostics:parser.error("unrecorded inherited DiT diagnostics: "+",".join(diagnostics))
    checkpoint=args.model/"diffusion_models/seedvr2_3b_fp16.safetensors"
    if sha256(checkpoint)!=CHECKPOINT_SHA:parser.error("original 3B checkpoint required")
    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import save_file,load_file
    with safe_open(args.input,framework="numpy") as source:noise=source.get_tensor("noise")
    with safe_open(args.condition_fixture or args.input,framework="numpy") as source:condition=source.get_tensor("condition")
    if (noise.dtype!=np.float32 or len(noise.shape)!=5 or noise.shape[:2]!=(1,16) or condition.shape!=noise.shape or
        not 1<=1+4*(noise.shape[2]-1)<=121):parser.error("complete original noise/conditioning within current source-frame scope required")
    args.output.mkdir(parents=True);prepared=args.output/"input.safetensors"
    save_file(dict(noise=noise,condition=condition),str(prepared))
    paths=dict(probe=args.probe,lifecycle_probe=args.lifecycle_probe,checkpoint=checkpoint,input=args.input,prepared_input=prepared)
    if args.condition_fixture:paths["condition_fixture"]=args.condition_fixture
    if args.frozen_reference:paths["frozen_reference"]=args.frozen_reference
    identities={key:sha256(path) for key,path in paths.items()}
    report=dict(schema="seedvr2-block-stream-screen-v2",status="running",qualified=False,identities=identities,
        shape=list(noise.shape),source_frames=1+4*(noise.shape[2]-1),repeats=args.repeats,cases=[],word_gates=[],
        scope="all original coefficients, selected actual read ranges and complete32-block velocity/Euler; not balanced speed/full request/App/real low-RAM/>121 qualification")
    try:
        for route in ("lifecycle","weights","whole","stream"):
            destination=args.output/route
            command=[str(args.lifecycle_probe),str(args.model),str(destination)] if route=="lifecycle" else [
                str(args.probe),"weights",str(args.model),str(destination)] if route=="weights" else [
                str(args.probe),"forward",str(args.model),str(prepared),str(destination),route,str(args.repeats)]
            print(f"block stream screen route={route} source_frames={report['source_frames']}",flush=True)
            result=subprocess.run(command,capture_output=True,text=True,timeout=600)
            row=dict(route=route,command=command,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr);report["cases"].append(row)
            if result.returncode:raise ValueError("block stream probe failed: "+result.stderr)
            value=json.loads((destination/"report.json").read_text());row["observation"]=value
            if route=="lifecycle":validate_lifecycle(value)
            elif route=="weights":
                offset,header=read_header(checkpoint);report["weight_read_gate"]=validate_weight_reads(value,offset,header)
            else:validate_forward(value,route,noise.shape,args.repeats,report["weight_read_gate"]["forward_source_bytes"])
            write_json(args.output/"report.json",report)
        frozen=load_file(str(args.frozen_reference)) if args.frozen_reference else None
        if frozen is not None:compare_words(condition,frozen["condition"])
        for iteration in range(args.repeats):
            original=load_file(str(args.output/"whole"/f"result-{iteration}.safetensors"))
            candidate=load_file(str(args.output/"stream"/f"result-{iteration}.safetensors"))
            gate=dict(iteration=iteration)
            for name in ("velocity","latent"):
                gate[name]=compare_words(original[name],candidate[name])
                if frozen is not None:compare_words(original[name],frozen[name])
            report["word_gates"].append(gate)
        if identities!={key:sha256(path) for key,path in paths.items()}:raise ValueError("block stream source identity changed")
        report["status"]="complete";print(json.dumps(dict(status="complete",source_frames=report["source_frames"],all_velocity_latent_words_equal=True)),flush=True)
    except BaseException as error:report.update(status="failed",failure=str(error));raise
    finally:write_json(args.output/"report.json",report)


if __name__=="__main__":main()
