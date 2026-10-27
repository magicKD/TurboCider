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
from seedvr2_spatial_decode_screen import validate_causal_bank,validate_upscale_causal_bank,CHECKPOINT_SHA
from export_seedvr2_ffn import tree_identity


def validate_geometry(value,bank):
    if not isinstance(value,dict) or (value.get("rows"),value.get("cols"))!=(bank["rows"],bank["cols"]):
        raise ValueError("executed tile geometry differs from selected bank")
    for name,rows,cols,channels in (("input_strides",bank["rows"]+2,bank["cols"]+2,768),
                                    ("output_strides",bank["rows"],bank["cols"],bank["output_channels"])):
        strides=value.get(name)
        if (not isinstance(strides,list) or len(strides)!=4 or any(type(item) is not int or item<=0 for item in strides) or
            strides[3]!=1 or strides[2]<cols or strides[1]<strides[2]*rows or strides[0]<strides[1]*channels):
            raise ValueError("invalid executed surface pitch/extent")


def comparison_specs(reference_route,candidate_route,manifest,upscale_manifest=None,candidate_manifest=None,candidate_upscale_manifest=None):
    """Keep same-route, different-bank experiments distinct in all ABBA rows."""
    if reference_route==candidate_route and not (candidate_manifest or candidate_upscale_manifest):
        raise ValueError("same-route comparisons require an explicit alternative bank")
    up_routes=("upscale","dual","dual_norm")
    hot_routes=("ne","cpu","dual","norm","dual_norm")
    if candidate_manifest and candidate_route not in hot_routes:
        raise ValueError("candidate hot bank is unused by this route")
    if bool(upscale_manifest)!=(reference_route in up_routes or (candidate_route in up_routes and not candidate_upscale_manifest)):
        raise ValueError("reference/shared upscale bank must be explicit and used")
    if candidate_upscale_manifest and candidate_route not in up_routes:
        raise ValueError("candidate upscale bank is unused by this route")
    labels=(reference_route,candidate_route) if reference_route!=candidate_route else ("reference","candidate")
    return {labels[0]:dict(route=reference_route,manifest=manifest,upscale_manifest=upscale_manifest if reference_route in up_routes else None),
            labels[1]:dict(route=candidate_route,manifest=candidate_manifest or manifest,
                upscale_manifest=(candidate_upscale_manifest or upscale_manifest) if candidate_route in up_routes else None)}


def validate_observation(value,route,frames,height,width,repeats,bank,upscale_bank=None,require_geometry=False):
    hot=route in ("ne","cpu","dual","norm","dual_norm");upscale=route in ("upscale","dual","dual_norm")
    if upscale and upscale_bank is None:raise ValueError("upscale benchmark needs its independent full-output bank")
    if (value.get("schema")!="seedvr2-vae-causal-decode-bench-v1" or value.get("status")!="complete" or
        value.get("route")!=route or (value.get("frames"),value.get("height"),value.get("width"))!=(frames,height,width) or
        value.get("share_upscale_history") is not True or value.get("qualified") is not False or
        value.get("physical_ane_occupancy_measured") is not False or len(value.get("iterations",[]))!=repeats):
        raise ValueError("incomplete or mislabelled fresh decoder benchmark")
    expected=(frames-1)*math.ceil(height/bank["rows"])*math.ceil(width/bank["cols"]) if hot else 0
    expected_up=(frames-1)*math.ceil(height/upscale_bank["rows"])*math.ceil(width/upscale_bank["cols"]) if upscale else 0
    for index,row in enumerate(value["iterations"]):
        if (row.get("iteration")!=index or row.get("warm")!=(index>0) or row.get("emitted_frames")!=frames or
            row.get("compared_values")!=frames*height*width*3 or row.get("coreml_calls")!=expected or
            row.get("causal_frames")!=(frames-1 if hot else 0) or row.get("initial_gpu_frame_checked") is not True or
            row.get("per_frame_quality_checked") is not True or
            row.get("upscale_coreml_calls",0)!=expected_up or row.get("upscale_causal_frames",0)!=(frames-1 if upscale else 0) or
            row.get("norm_pack_frames",0)!=(frames-1 if route in ("norm","dual_norm") else 0) or
            row.get("quality_passed") is not True or (route=="gpu" and row.get("unequal_values")!=0) or
            type(row.get("mlx_active_peak_bytes")) is not int or row["mlx_active_peak_bytes"]<=0):
            raise ValueError("warm benchmark frame/phase/actual execution coverage differs")
        if upscale and (row.get("total_coreml_calls")!=expected+expected_up or
            not math.isfinite(row.get("upscale_causal_boundary_seconds",float("nan")))):
            raise ValueError("complete multi-layer call/boundary accounting differs")
        for field in ("seconds_with_host_quality","relative_l2","cosine","causal_boundary_seconds"):
            if not math.isfinite(row.get(field,float("nan"))):raise ValueError("nonfinite benchmark observation")
        if row["seconds_with_host_quality"]<=0 or not 0<=row["relative_l2"]<=.05 or row["cosine"]<.999:
            raise ValueError("unchanged warm decoder pixel gate failed")
    if (route=="gpu" and value.get("external_slot_bytes")!=0) or (route!="gpu" and value.get("external_slot_bytes",0)<=0):
        raise ValueError("declared external slot coverage differs")
    if upscale and (value.get("upscale_layer")!=upscale_bank["layer"] or value.get("upscale_external_slot_bytes",0)<=0 or
        value["external_slot_bytes"]<value["upscale_external_slot_bytes"]):
        raise ValueError("upscale layer/external slot accounting differs")
    if require_geometry:
        for name,used,selected in (("hot_geometry",hot,bank),("upscale_geometry",upscale,upscale_bank)):
            if used:validate_geometry(value.get(name),selected)
            elif value.get(name) is not None:raise ValueError("unused route declares an executed tile geometry")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("probe","model","fixture","manifest","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--upscale-manifest",type=Path)
    parser.add_argument("--candidate-manifest",type=Path)
    parser.add_argument("--candidate-upscale-manifest",type=Path)
    parser.add_argument("--reference-route",choices=("gpu","ne","dual","dual_norm"),default="gpu")
    parser.add_argument("--candidate-route",choices=("ne","upscale","dual","norm","dual_norm"),default="ne")
    args=parser.parse_args()
    if args.output.exists() or not 2<=args.repeats<=7:parser.error("fresh output and bounded repetitions required")
    for name in ("probe","model","fixture","manifest"):setattr(args,name,getattr(args,name).resolve(strict=True))
    for name in ("upscale_manifest","candidate_manifest","candidate_upscale_manifest"):
        if getattr(args,name):setattr(args,name,getattr(args,name).resolve(strict=True))
    try:
        specs=comparison_specs(args.reference_route,args.candidate_route,args.manifest,args.upscale_manifest,
            args.candidate_manifest,args.candidate_upscale_manifest)
    except ValueError as error:parser.error(str(error))
    compiled_trees={}
    for spec in specs.values():
        for key,validator in (("manifest",validate_causal_bank),("upscale_manifest",validate_upscale_causal_bank)):
            path=spec[key]
            if not path:continue
            bank=json.loads(path.read_text());validator(bank);spec[key+"_bank"]=bank
            compiled=path.parent/bank["compiled"]["path"]
            if tree_identity(compiled)!=bank["compiled"]["identity"]:parser.error("compiled causal tree differs")
            compiled_trees[compiled]=bank["compiled"]["identity"]
    checkpoint=args.model/"vae/seedvr2_ema_vae_fp16.safetensors"
    if sha256(checkpoint)!=CHECKPOINT_SHA:parser.error("original checkpoint required")
    from safetensors import safe_open
    with safe_open(args.fixture,framework="numpy") as fixture:
        key="decoded" if "decoded" in fixture.keys() else "pixels";shape=fixture.get_slice(key).get_shape()
        latent=fixture.get_slice("latent").get_shape()
        if fixture.get_slice(key).get_dtype()!="F16":parser.error("original FP16 pixel witness required")
    if len(shape)!=5 or shape[:2]!=[1,3] or shape[2]!=1+4*(latent[2]-1):parser.error("complete untrimmed decoder fixture required")
    frames,height,width=shape[2:]
    sources=dict(probe=args.probe,input=args.fixture,checkpoint=checkpoint)
    for label,spec in specs.items():
        for key in ("manifest","upscale_manifest"):
            if spec[key]:sources[label+"_"+key]=spec[key]
    identities={key:sha256(path) for key,path in sources.items()}
    reference_label,candidate_label=specs
    if args.reference_route==args.candidate_route and all(
        identities.get(reference_label+"_"+key)==identities.get(candidate_label+"_"+key) for key in ("manifest","upscale_manifest")):
        parser.error("same-route comparison has identical banks")
    args.output.mkdir(parents=True)
    report=dict(schema="seedvr2-vae-causal-decode-abba-v2",status="running",qualified=False,
        identities=identities,order=[reference_label,candidate_label,candidate_label,reference_label],
        reference_label=reference_label,candidate_label=candidate_label,
        selections={label:{key:str(spec[key]) if spec[key] else None for key in ("route","manifest","upscale_manifest")} for label,spec in specs.items()},
        reference_route=args.reference_route,candidate_route=args.candidate_route,repeats=args.repeats,frames=frames,height=height,width=width,cohorts=[],
        scope="complete freshly decoded component per warm iteration including host pixel gate, excluding model load; not full request/App/physical RAM qualification")
    try:
        for index,label in enumerate(report["order"]):
            spec=specs[label];route=spec["route"];bank=spec["manifest_bank"];upscale_bank=spec.get("upscale_manifest_bank")
            destination=args.output/f"{index}-{label}"
            command=[str(args.probe),str(args.model),str(args.fixture),str(spec["manifest"]),str(destination),route,str(args.repeats)]
            if spec["upscale_manifest"]:command.append(str(spec["upscale_manifest"]))
            print(f"causal ABBA cohort={index} selection={label} route={route} complete_frames={frames} fresh_iterations={args.repeats}",flush=True)
            receipt=dict(index=index,label=label,route=route,command=command);report["cohorts"].append(receipt);write_json(args.output/"report.json",report)
            with (args.output/f"stdout-{index}.txt").open("w") as stdout,(args.output/f"stderr-{index}.txt").open("w") as stderr:
                process=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=1800)
            receipt["exit_code"]=process.returncode
            if process.returncode:raise ValueError("causal decoder cohort failed; preserved terminal receipt")
            value=json.loads((destination/"report.json").read_text());validate_observation(value,route,frames,height,width,args.repeats,bank,upscale_bank,require_geometry=True)
            receipt["observation"]=value;write_json(args.output/"report.json",report)
        for label in (reference_label,candidate_label):
            rows=[row for cohort in report["cohorts"] if cohort["label"]==label for row in cohort["observation"]["iterations"] if row["warm"]]
            report[label]=dict(warm_median_seconds=statistics.median(row["seconds_with_host_quality"] for row in rows),
                active_peak_bytes=max(row["mlx_active_peak_bytes"] for row in rows),warm_observations=len(rows),
                sampled_process_footprint_peak_bytes=max(cohort["observation"]["sampled_process_footprint_peak_bytes"] for cohort in report["cohorts"] if cohort["label"]==label),
                external_slot_bytes=max(cohort["observation"]["external_slot_bytes"] for cohort in report["cohorts"] if cohort["label"]==label))
        report["speed_gate_passed"]=report[candidate_label]["warm_median_seconds"]<=report[reference_label]["warm_median_seconds"]
        report["active_peak_reduction_bytes"]=report[reference_label]["active_peak_bytes"]-report[candidate_label]["active_peak_bytes"]
        after={key:sha256(path) for key,path in sources.items()}
        if identities!=after or any(tree_identity(path)!=identity for path,identity in compiled_trees.items()):
            raise ValueError("cohort input/executable/compiled identities changed")
        report["status"]="complete";print(json.dumps({key:report[key] for key in ("status",reference_label,candidate_label,"speed_gate_passed","active_peak_reduction_bytes")}),flush=True)
    except BaseException as error:report.update(status="failed",failure=str(error));raise
    finally:write_json(args.output/"report.json",report)


if __name__=="__main__":main()
