#!/usr/bin/env python3
"""Corrected QKV fusion on an ordinary r128/joint whole GPU block.

Strong denominator is the original compiled block, not staged GPU, packed
projection alone or prepared-rank B math. No image/request extrapolation.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

from flux9_bf16_framework_screen import write_json
from qwen_student_prefix_screen import adapter_pin
from runtime_ane_common import benchmark_environment,sha256_file
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ROOT=Path(__file__).resolve().parents[2]
NAMES=("gpu-original","gpu-replay","mlx-packed-base","fused-b32","fused-b16","fused-b64")
JOINT_NAMES=NAMES+("joint-base-a-mlx-b32","joint-base-a-mpp-b32","fused-b48")


def quality(value,exact=False):
    for key in ("relative_l2","cosine","max_abs"):
        if type(value[key]) not in (int,float) or not math.isfinite(value[key]):raise ValueError("finite full-block quality required")
    if (not 0<=value["relative_l2"]<=.001 or not .999<=value["cosine"]<=1.000001 or value["max_abs"]<0 or
            type(value["elements"]) is not int or value["elements"]!=1024*4096 or
            type(value["changed"]) is not int or not 0<=value["changed"]<=value["elements"] or
            (exact and value["changed"]!=0)):
        raise ValueError("full-block numerical/strong-control gate differs")


def analyze(native):
    version=native.get("qkv_recipe_version")
    if version not in (None,"joint-base-a-v1"):raise ValueError("unknown QKV recipe version")
    joint=version is not None
    names=JOINT_NAMES if joint else NAMES
    count=18 if joint else 12
    if joint:
        if native.get("packed_joint_base_a_bytes")!=103809024 or native.get("measured_samples_per_arm")!=18:
            raise ValueError("complete joint base/A geometry and samples required")
        if len(native["rank_boundary_quality"])!=3:raise ValueError("all independent rank boundaries required")
        for i,row in enumerate(native["rank_boundary_quality"]):
            if type(row["projection"]) is not int or row["projection"]!=i:raise ValueError("QKV rank order differs")
            for key in ("mlx","mpp"):
                q=row[key]
                if (any(type(q[k]) not in (int,float) or not math.isfinite(q[k]) for k in ("relative_l2","cosine","max_abs")) or
                        not 0<=q["relative_l2"]<=.005 or q["max_abs"]<0 or
                        type(q["elements"]) is not int or q["elements"]!=1024*128 or not .999<=q["cosine"]<=1.000001 or
                        type(q["changed"]) is not int or not 0<=q["changed"]<=q["elements"]):
                    raise ValueError("typed sole-B rank boundary gate differs")
    if (native.get("status")!="measured" or native.get("runtime_lora_applied_terms")!=227 or native.get("image_export") is not False or
            native.get("borrowed_values_unchanged") is not True or native.get("fresh_arguments_verified") is not True or
            native.get("all_outputs_lazy_before_eval") is not True or native.get("completed_full_dit_steps")!=1 or
            native.get("stopped_step")!=1 or native.get("stopped_layer")!=0 or native.get("packed_base_bytes")!=100663296):
        raise ValueError("complete bounded original student-block evidence required")
    if len(native["initial_quality"])!=len(names)-1:raise ValueError("all initial recipes required")
    for row,name in zip(native["initial_quality"],names[1:]):
        if row["arm"]!=name:raise ValueError("initial recipe order differs")
        quality(row["quality"],name=="gpu-replay")
        if type(row["first_replay_seconds"]) not in (int,float) or not math.isfinite(row["first_replay_seconds"]) or row["first_replay_seconds"]<=0:
            raise ValueError("actual first replay cost required")
    samples=native["samples"]
    if len(samples)!=(count+3)*len(names):raise ValueError("all complete first/warm/measured arms required")
    for iteration in range(-3,count):
        group=[row for row in samples if row["iteration"]==iteration]
        if [row["arm"] for row in group]!=[names[(visit+iteration+len(names))%len(names)] for visit in range(len(names))]:
            raise ValueError("balanced arm order differs")
        for row in group:
            wall=row["seconds"]
            if type(wall) not in (int,float) or not math.isfinite(wall) or wall<=0:raise ValueError("finite complete block wall required")
            quality(row["quality"],row["arm"] in NAMES[:2])
    values={name:[row["seconds"] for row in samples if row["arm"]==name and row["iteration"]>=0] for name in names}
    medians={name:statistics.median(rows) for name,rows in values.items()}
    reductions={name:100*(1-value/medians["gpu-original"]) for name,value in medians.items()}
    primary="joint-base-a-mpp-b32" if joint else "fused-b32"
    return dict(whole_block_seconds=values,medians=medians,reduction_vs_original_gpu_percent=reductions,
        primary_recipe=primary,primary_local_gate_passed=reductions[primary]>=3.,
        scope="one actual ordinary decode block with online ranks and all consumers; no full-image speed/quality claim")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary",type=Path,required=True);p.add_argument("--library",type=Path,required=True)
    p.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--lora",type=Path,default=ROOT/"models/Viggle-Qwen-Image-2.1-viggle-turbo/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors")
    p.add_argument("--reference",type=Path);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    for key in ("binary","library","model","lora","reference"):
        if getattr(args,key) is not None:setattr(args,key,getattr(args,key).resolve(strict=True))
    pin=adapter_pin(args.lora)
    if pin["rank"]!="r128" or sha256_file(args.lora)!=pin["sha256"]:p.error("original pinned r128 required")
    manifest_path=args.library.parent/"runtime-build/runtime-build-manifest.json"
    manifest=json.loads(manifest_path.read_text())
    for item in manifest["inputs"]["sources"]:
        if sha256_file(ROOT/item["path"])!=item["sha256"]:p.error("library's native source no longer matches current headers")
    for name in ("diffusion_models/qwen_image_2.1_bf16.safetensors","text_encoders/qwen3vl_8b_bf16.safetensors"):
        if not (args.model/name).is_file():p.error("original BF16 components required")
    descriptor=json.loads(subprocess.check_output([str(args.binary),"--describe"],text=True,timeout=30))
    if (descriptor["recipe_version"]!="joint-base-a-v1" or descriptor["arms"]!=list(JOINT_NAMES) or
            descriptor["primary_recipe"]!="joint-base-a-mpp-b32" or descriptor["measured_samples_per_arm"]!=18 or
            descriptor["loaded_runtime_id"]!=manifest["runtime_build_id"] or
            descriptor["cpp_sha256"]!=sha256_file(ROOT/"tools/native/qwen21_qkv_lora_block_probe.mm") or
            descriptor["header_sha256"]!=sha256_file(ROOT/"tools/native/qwen21_qkv_lora_prepare.hpp")):
        p.error("current bound probe source/recipe/runtime required; rebuild into a fresh output")
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),ROOT/"tools/native/qwen21_qkv_lora_block_probe.mm",ROOT/"tools/native/qwen21_qkv_lora_prepare.hpp",
           ROOT/"tools/native/build_qwen21_qkv_lora_block_probe.sh"]
    hashes={str(path.relative_to(ROOT)):sha256_file(path) for path in paths}
    summary=dict(schema="tc-qwen-corrected-qkv-screen-v1",status="running",qualification_passed=False,
        source_sha256=hashes,binary_sha256=sha256_file(args.binary),library_sha256=sha256_file(args.library),
        runtime_build_id=manifest["runtime_build_id"],manifest_sha256=sha256_file(manifest_path),
        adapter_sha256=pin["sha256"],case="edit" if args.reference else "generate",
        probe_descriptor=descriptor,
        gates=dict(primary_recipe="joint-base-a-mpp-b32",minimum_complete_block_reduction_percent=3.,maximum_block_relative_l2=.001,
                   minimum_block_cosine=.999,strong_gpu_replay_exact=True),
        scope="bounded step1/layer0; complete online A/B and all block consumers; original GPU denominator, not full request")
    if args.reference:summary["reference_sha256"]=sha256_file(args.reference)
    write_json(args.output/"source.json",{str(path.relative_to(ROOT)):path.read_text() for path in paths})
    write_json(args.output/"summary.json",summary)
    command=[str(args.binary),str(args.model),str(args.lora),str(args.output/"trial"),str(args.reference) if args.reference else "-"]
    observer=LoadObservation(args.output/"load.jsonl")
    try:
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            summary["memory"]=run_sampled(command,repo=ROOT,output=args.output,stem="trial",env=benchmark_environment(),
                stdout=stdout,stderr=stderr,timeout=1200,interval_ms=100,max_gap_ms=500,observer=observer)
        native=json.loads((args.output/"trial/native.json").read_text())
        if native["loaded_runtime_id"]!=manifest["runtime_build_id"]:raise ValueError("actual loaded runtime differs")
        summary["analysis"]=analyze(native);summary["status"]="complete"
    except Exception as error:
        summary.update(status="failed",error=str(error))
        path=args.output/"trial/native.json"
        if path.is_file():
            data=json.loads(path.read_text());summary["native_error"]=data.get("error");summary["initial_quality"]=data.get("initial_quality")
    try:summary["load"]=observer.verify()
    except ValueError as error:summary["load_error"]=str(error)
    if any(sha256_file(path)!=hashes[str(path.relative_to(ROOT))] for path in paths) or sha256_file(args.binary)!=summary["binary_sha256"] or\
            sha256_file(args.library)!=summary["library_sha256"]:summary.update(status="failed",error="source/artifact changed during observation")
    write_json(args.output/"summary.json",summary);print(json.dumps(summary,indent=2))
    if summary["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
