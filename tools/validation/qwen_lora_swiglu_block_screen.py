#!/usr/bin/env python3
"""Real r128 complete-block replay, ordinary/direct-QKV strong controls.

Strict complete BF16 equality, all online ranks/down/residual included. Local
primary signal requires >=1% reduction versus same-window direct-QKV, not an
approximate-ANE adoption threshold or a full-image speed certificate.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sysconfig

from flux9_bf16_framework_screen import write_json
from qwen_student_prefix_screen import adapter_pin
from runtime_ane_common import benchmark_environment,sha256_file
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ROOT=Path(__file__).resolve().parents[2]
ARMS=("gpu-original","gpu-replay","direct-qkv","paired-b-swiglu-b16","paired-b-swiglu-b32",
      "direct-qkv-paired-b-swiglu-b16","direct-qkv-paired-b-swiglu-b32")
PRIMARY=ARMS[5];CONTROL="direct-qkv";WARMUPS=3;SAMPLES=21
SOURCES=("tools/validation/qwen_lora_swiglu_block_screen.py","tools/native/qwen21_lora_swiglu_probe.mm",
         "tools/native/qwen21_lora_swiglu_prepare.hpp","tools/native/build_qwen21_lora_swiglu_probe.sh",
         "tools/native/qwen21_qkv_lora_block_probe.mm","tools/native/qwen21_qkv_lora_prepare.hpp")


def quality(q):
    if (type(q.get("changed")) is not int or q["changed"]!=0 or q.get("elements")!=4194304 or
            type(q.get("relative_l2")) not in (int,float) or q["relative_l2"]!=0 or q.get("max_abs")!=0 or
            type(q.get("cosine")) not in (int,float) or not math.isfinite(q["cosine"]) or q["cosine"]<.999999999):
        raise ValueError("strict complete original-consumer BF16 equality required")


def analyze(native,order):
    if (native.get("status")!="measured" or native.get("image_export") is not False or native.get("order")!=order or
            native.get("runtime_lora_applied_terms")!=227 or native.get("extra_owned_weight_bytes")!=0 or
            native.get("completed_full_dit_steps")!=1 or native.get("stopped_step")!=1 or native.get("stopped_layer")!=0 or
            native.get("fresh_arguments_verified") is not True or native.get("borrowed_values_unchanged") is not True or
            native.get("all_outputs_lazy_before_eval") is not True or native.get("bitwise_outputs_checked") is not True or
            len(native.get("actual_layer_source_sha256",{}))!=22 or not isinstance(native.get("ordinary_block_sha256"),str) or len(native["ordinary_block_sha256"])!=64):
        raise ValueError("bounded full-block work/source/ownership receipt differs")
    first=native.get("first_replays",[])
    if [row["arm"] for row in first]!=list(ARMS[1:]):raise ValueError("all first complete replays required")
    for row in first:
        quality(row["quality"])
        if row.get("output_sha256")!=native["ordinary_block_sha256"]:raise ValueError("complete first-replay bits differ")
        if type(row["seconds"]) not in (int,float) or not math.isfinite(row["seconds"]) or row["seconds"]<=0:raise ValueError("finite first-use wall required")
    rows=native.get("samples",[])
    expected=[(iteration,visit,ARMS[((6-visit if order=="reverse" else visit)+iteration+7)%7])
              for iteration in range(-WARMUPS,SAMPLES) for visit in range(7)]
    if len(rows)!=len(expected):raise ValueError("all balanced warm/measured complete calls required")
    samples={arm:[] for arm in ARMS}
    for row,(iteration,visit,arm) in zip(rows,expected):
        if (row["iteration"],row["visit"],row["arm"])!=(iteration,visit,arm):raise ValueError("frozen rotation/order differs")
        wall=row["seconds"]
        if type(wall) not in (int,float) or not math.isfinite(wall) or wall<=0:raise ValueError("finite completed whole-block time required")
        quality(row["quality"])
        if row.get("output_sha256")!=native["ordinary_block_sha256"]:raise ValueError("complete live output bits differ")
        if iteration>=0:samples[arm].append(wall)
    medians={arm:statistics.median(values) for arm,values in samples.items()}
    primary=100*(1-medians[PRIMARY]/medians[CONTROL])
    return dict(median_seconds=medians,primary=PRIMARY,strong_control=CONTROL,primary_reduction_percent=primary,
        primary_local_signal=primary>=1.,all_complete_outputs_exact=True,all_complete_outputs_bitwise_exact=True,measured_calls=SAMPLES*7,warm_calls=WARMUPS*7,
        compared_scope="step1/layer0 full block, not six-step/VAE/PNG request latency; no multiplied historical gains")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--binary",type=Path,required=True);p.add_argument("--library",type=Path,required=True)
    p.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--lora",type=Path,default=ROOT/"models/Viggle-Qwen-Image-2.1-viggle-turbo/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors")
    p.add_argument("--reference",type=Path);p.add_argument("--order",choices=("forward","reverse"),default="forward");p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    for key in ("binary","library","model","lora"):setattr(args,key,getattr(args,key).resolve(strict=True))
    if args.reference:args.reference=args.reference.resolve(strict=True)
    args.output=args.output.resolve();pin=adapter_pin(args.lora)
    if pin["rank"]!="r128" or sha256_file(args.lora)!=pin["sha256"]:p.error("original released r128 required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    for source in manifest["inputs"]["sources"]:
        if sha256_file(ROOT/source["path"])!=source["sha256"]:p.error("native source differs from linked library")
    descriptor=json.loads(subprocess.check_output([str(args.binary),"--describe"],text=True,cwd=ROOT,env=benchmark_environment()))
    mlx=Path(sysconfig.get_paths()["purelib"])/"mlx/include/mlx/backend/metal/kernels"
    external={name:mlx/name for name in ("unary_ops.h","bf16_math.h")}
    hashes={name:sha256_file(ROOT/name) for name in SOURCES};external_hashes={name:sha256_file(path) for name,path in external.items()}
    if (descriptor.get("recipe")!="native-base-paired-b-swiglu-v2-bitwise" or descriptor.get("arms")!=list(ARMS) or descriptor.get("primary")!=PRIMARY or
            descriptor.get("strong_control")!=CONTROL or descriptor.get("warmups")!=WARMUPS or descriptor.get("samples_per_arm")!=SAMPLES or
            descriptor.get("loaded_runtime_id")!=manifest["runtime_build_id"] or
            descriptor.get("cpp_sha256")!=hashes[SOURCES[1]] or descriptor.get("header_sha256")!=hashes[SOURCES[2]] or descriptor.get("helper_sha256")!=hashes[SOURCES[4]] or
            descriptor.get("unary_header_sha256")!=external_hashes["unary_ops.h"] or descriptor.get("bf16_math_sha256")!=external_hashes["bf16_math.h"]):
        p.error("binary/source/runtime/typed-operator descriptor differs")
    assets={key:sha256_file(getattr(args,key)) for key in ("binary","library","lora")}
    if args.reference:assets["reference"]=sha256_file(args.reference)
    args.output.mkdir(parents=True,exist_ok=False);trial=args.output/"trial"
    summary=dict(schema="tc-qwen-lora-swiglu-block-screen-v1",status="running",qualification_passed=False,order=args.order,edit=bool(args.reference),
                 descriptor=descriptor,source_sha256=hashes,mlx_header_sha256=external_hashes,asset_sha256=assets)
    write_json(args.output/"source.json",{name:(ROOT/name).read_text() for name in SOURCES})
    write_json(args.output/"mlx-headers.json",{name:path.read_text() for name,path in external.items()});write_json(args.output/"summary.json",summary)
    command=[str(args.binary),str(args.model),str(args.lora),str(trial),str(args.reference) if args.reference else "-",args.order]
    observer=LoadObservation(args.output/"load.jsonl")
    try:
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            summary["memory"]=run_sampled(command,repo=ROOT,output=args.output,stem="trial",env=benchmark_environment(),stdout=stdout,stderr=stderr,
                timeout=1800,interval_ms=100,max_gap_ms=500,observer=observer)
        native=json.loads((trial/"native.json").read_text())
        if native["loaded_runtime_id"]!=descriptor["loaded_runtime_id"]:raise ValueError("actual loaded runtime differs")
        summary["analysis"]=analyze(native,args.order);summary["status"]="complete"
    except Exception as error:summary.update(status="failed",error=str(error))
    try:summary["load"]=observer.verify()
    except ValueError as error:summary["load_error"]=str(error)
    if (any(sha256_file(ROOT/name)!=digest for name,digest in hashes.items()) or any(sha256_file(path)!=external_hashes[name] for name,path in external.items()) or
            any(sha256_file(getattr(args,key))!=digest for key,digest in assets.items())):summary.update(status="failed",error="bound source/asset changed")
    write_json(args.output/"summary.json",summary);print(json.dumps(summary,indent=2))
    if summary["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
