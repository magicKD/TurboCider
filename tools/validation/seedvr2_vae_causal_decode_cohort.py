#!/usr/bin/env python3
"""Balanced full-decoder GPU/virtual-halo CPU+NE ABBA component cohort.

Each process retains only weights/model/cache between fresh decoder iterations.
This does not qualify a full request, App integration, physical ANE occupancy or
low-RAM hardware; initial GPU-FP32 folding and every output frame are mandatory.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
from seedvr2_framework_screen import sha256,write_json
from seedvr2_spatial_decode_screen import validate_causal_bank,CHECKPOINT_SHA
from export_seedvr2_ffn import tree_identity


def validate_observation(value,route,frames,height,width,repeats,bank):
    if (value.get("schema")!="seedvr2-vae-causal-decode-bench-v1" or value.get("status")!="complete" or
        value.get("route")!=route or (value.get("frames"),value.get("height"),value.get("width"))!=(frames,height,width) or
        value.get("share_upscale_history") is not True or value.get("qualified") is not False or
        value.get("physical_ane_occupancy_measured") is not False or len(value.get("iterations",[]))!=repeats):
        raise ValueError("incomplete or mislabelled fresh decoder benchmark")
    expected=0 if route=="gpu" else (frames-1)*math.ceil(height/bank["rows"])*math.ceil(width/bank["cols"])
    for index,row in enumerate(value["iterations"]):
        if (row.get("iteration")!=index or row.get("warm")!=(index>0) or row.get("emitted_frames")!=frames or
            row.get("compared_values")!=frames*height*width*3 or row.get("coreml_calls")!=expected or
            row.get("causal_frames")!=(0 if route=="gpu" else frames-1) or row.get("initial_gpu_frame_checked") is not True or
            row.get("quality_passed") is not True or (route=="gpu" and row.get("unequal_values")!=0) or
            type(row.get("mlx_active_peak_bytes")) is not int or row["mlx_active_peak_bytes"]<=0):
            raise ValueError("warm benchmark frame/phase/actual execution coverage differs")
        for field in ("seconds_with_host_quality","relative_l2","cosine","causal_boundary_seconds"):
            if not math.isfinite(row.get(field,float("nan"))):raise ValueError("nonfinite benchmark observation")
        if row["seconds_with_host_quality"]<=0 or not 0<=row["relative_l2"]<=.05 or row["cosine"]<.999:
            raise ValueError("unchanged warm decoder pixel gate failed")
    if (route=="gpu" and value.get("external_slot_bytes")!=0) or (route!="gpu" and value.get("external_slot_bytes",0)<=0):
        raise ValueError("declared external slot coverage differs")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("probe","model","fixture","manifest","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--repeats",type=int,default=3)
    args=parser.parse_args()
    if args.output.exists() or not 2<=args.repeats<=7:parser.error("fresh output and bounded repetitions required")
    for name in ("probe","model","fixture","manifest"):setattr(args,name,getattr(args,name).resolve(strict=True))
    bank=json.loads(args.manifest.read_text());validate_causal_bank(bank)
    compiled=args.manifest.parent/bank["compiled"]["path"]
    if tree_identity(compiled)!=bank["compiled"]["identity"]:parser.error("compiled causal tree differs")
    checkpoint=args.model/"vae/seedvr2_ema_vae_fp16.safetensors"
    if sha256(checkpoint)!=CHECKPOINT_SHA:parser.error("original checkpoint required")
    from safetensors import safe_open
    with safe_open(args.fixture,framework="numpy") as fixture:
        key="decoded" if "decoded" in fixture.keys() else "pixels";shape=fixture.get_slice(key).get_shape()
        latent=fixture.get_slice("latent").get_shape()
        if fixture.get_slice(key).get_dtype()!="F16":parser.error("original FP16 pixel witness required")
    if len(shape)!=5 or shape[:2]!=[1,3] or shape[2]!=1+4*(latent[2]-1):parser.error("complete untrimmed decoder fixture required")
    frames,height,width=shape[2:];args.output.mkdir(parents=True)
    identities=dict(probe=sha256(args.probe),input=sha256(args.fixture),checkpoint=CHECKPOINT_SHA,manifest=sha256(args.manifest))
    report=dict(schema="seedvr2-vae-causal-decode-abba-v1",status="running",qualified=False,
        identities=identities,order=["gpu","ne","ne","gpu"],repeats=args.repeats,frames=frames,height=height,width=width,cohorts=[],
        scope="complete freshly decoded component per warm iteration including host pixel gate, excluding model load; not full request/App/physical RAM qualification")
    try:
        for index,route in enumerate(report["order"]):
            destination=args.output/f"{index}-{route}"
            command=[str(args.probe),str(args.model),str(args.fixture),str(args.manifest),str(destination),route,str(args.repeats)]
            print(f"causal ABBA cohort={index} route={route} complete_frames={frames} fresh_iterations={args.repeats}",flush=True)
            receipt=dict(index=index,route=route,command=command);report["cohorts"].append(receipt);write_json(args.output/"report.json",report)
            with (args.output/f"stdout-{index}.txt").open("w") as stdout,(args.output/f"stderr-{index}.txt").open("w") as stderr:
                process=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=1800)
            receipt["exit_code"]=process.returncode
            if process.returncode:raise ValueError("causal decoder cohort failed; preserved terminal receipt")
            value=json.loads((destination/"report.json").read_text());validate_observation(value,route,frames,height,width,args.repeats,bank)
            receipt["observation"]=value;write_json(args.output/"report.json",report)
        for route in ("gpu","ne"):
            rows=[row for cohort in report["cohorts"] if cohort["route"]==route for row in cohort["observation"]["iterations"] if row["warm"]]
            report[route]=dict(warm_median_seconds=statistics.median(row["seconds_with_host_quality"] for row in rows),
                active_peak_bytes=max(row["mlx_active_peak_bytes"] for row in rows),warm_observations=len(rows))
        report["speed_gate_passed"]=report["ne"]["warm_median_seconds"]<=report["gpu"]["warm_median_seconds"]
        report["active_peak_reduction_bytes"]=report["gpu"]["active_peak_bytes"]-report["ne"]["active_peak_bytes"]
        if identities!=dict(probe=sha256(args.probe),input=sha256(args.fixture),checkpoint=sha256(checkpoint),manifest=sha256(args.manifest)) or tree_identity(compiled)!=bank["compiled"]["identity"]:
            raise ValueError("cohort input/executable/compiled identities changed")
        report["status"]="complete";print(json.dumps({key:report[key] for key in ("status","gpu","ne","speed_gate_passed","active_peak_reduction_bytes")}),flush=True)
    except BaseException as error:report.update(status="failed",failure=str(error));raise
    finally:write_json(args.output/"report.json",report)


if __name__=="__main__":main()
