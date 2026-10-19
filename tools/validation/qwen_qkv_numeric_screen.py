#!/usr/bin/env python3
"""Actual-input QKV stage counterfactuals and complete ordinary block controls."""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess

from flux9_bf16_framework_screen import write_json
from qwen_qkv_lora_block_screen import quality
from qwen_student_prefix_screen import adapter_pin
from runtime_ane_common import benchmark_environment,sha256_file
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ROOT=Path(__file__).resolve().parents[2]
NAMES=("gpu-original","gpu-replay","mpp-prepare","native-base-b16","native-base-b32")
ROPE_NAMES=NAMES+("no-contract-prepare",)
RECIPES=("mpp-base-b16","native-base-b16","mpp-base-b32","native-base-b32")


def analyze(native):
    version=native.get("numeric_recipe")
    if version not in (None,"rope-contract-v2"):raise ValueError("known counterfactual recipe required")
    rope=version is not None;names=ROPE_NAMES if rope else NAMES;count=12 if rope else 15
    if (native.get("status")!="measured" or native.get("runtime_lora_applied_terms")!=227 or native.get("image_export") is not False or
            native.get("actual_full_expression_exact") is not True or native.get("borrowed_values_unchanged") is not True or
            native.get("fresh_arguments_verified") is not True or native.get("completed_full_dit_steps")!=1 or
            native.get("stopped_step")!=1 or native.get("stopped_layer")!=0):
        raise ValueError("complete actual-input counterfactual contract required")
    stages=native["stage_quality"];expected=set()
    for recipe in RECIPES:
        expected.update((recipe,stage,i) for stage in range(4) for i in range(3))
        expected.update((recipe+"-matched-"+name,stage,i) for name,stage in (("norm",2),("rope",3)) for i in range(3))
        if rope:expected.update((recipe+"-no-contract",3,i) for i in range(3))
    seen=set()
    for row in stages:
        key=(row["recipe"],row["stage"],row["projection"])
        if key not in expected or key in seen:raise ValueError("all unique stage/side/projection controls required")
        seen.add(key);q=row["quality"]
        if (any(type(q[k]) not in (int,float) or not math.isfinite(q[k]) for k in ("relative_l2","cosine","max_abs")) or
                q["relative_l2"]<0 or not -1.000001<=q["cosine"]<=1.000001 or q["max_abs"]<0 or
                type(q["changed"]) is not int or not 0<=q["changed"]<=1024*4096 or
                type(q["elements"]) is not int or q["elements"]!=1024*4096):raise ValueError("finite complete head-major numerical evidence required")
        if row["recipe"].startswith("native-base-") and row["stage"]==0 and q["changed"]!=0:
            raise ValueError("supplied original base identity control failed")
    if seen!=expected:raise ValueError("missing a stage counterfactual")
    samples=native["samples"]
    if len(samples)!=(count+3)*len(names):raise ValueError("all complete warm/measured arms required")
    for iteration in range(-3,count):
        group=[row for row in samples if row["iteration"]==iteration]
        if [row["arm"] for row in group]!=[names[(visit+iteration+len(names))%len(names)] for visit in range(len(names))]:raise ValueError("balanced complete block order differs")
        for row in group:
            value=row["seconds"]
            if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError("finite full-block wall required")
            quality(row["quality"],row["arm"] in NAMES[:2])
    values={name:[row["seconds"] for row in samples if row["arm"]==name and row["iteration"]>=0] for name in names}
    medians={name:statistics.median(rows) for name,rows in values.items()}
    summaries=[]
    for recipe in RECIPES:
        rows=[row for row in stages if row["recipe"]==recipe]
        summaries.append(dict(recipe=recipe,original_chain_changed_by_stage=[sum(row["quality"]["changed"] for row in rows if row["stage"]==stage) for stage in range(4)],
            matched_norm_changed=sum(row["quality"]["changed"] for row in stages if row["recipe"]==recipe+"-matched-norm"),
            matched_rope_changed=sum(row["quality"]["changed"] for row in stages if row["recipe"]==recipe+"-matched-rope")))
        if rope:summaries[-1]["no_contract_changed"]=sum(row["quality"]["changed"] for row in stages if row["recipe"]==recipe+"-no-contract")
    return dict(stage_summaries=summaries,whole_block_seconds=values,medians=medians,
        reduction_vs_original_gpu_percent={name:100*(1-value/medians["gpu-original"]) for name,value in medians.items()},primary="no-contract-prepare" if rope else "native-base-b16",
        scope="one real layer0 decode block; matched-input local stage attribution, not a unique 32-layer/image cause or quality certificate")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--binary",type=Path,required=True);p.add_argument("--library",type=Path,required=True)
    p.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--lora",type=Path,default=ROOT/"models/Viggle-Qwen-Image-2.1-viggle-turbo/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors")
    p.add_argument("--reference",type=Path);p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    for key in ("binary","library","model","lora","reference"):
        if getattr(args,key) is not None:setattr(args,key,getattr(args,key).resolve(strict=True))
    pin=adapter_pin(args.lora)
    if pin["rank"]!="r128" or sha256_file(args.lora)!=pin["sha256"]:p.error("pinned original r128 required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    for item in manifest["inputs"]["sources"]:
        if sha256_file(ROOT/item["path"])!=item["sha256"]:p.error("native tree must match the linked library")
    sources=[Path(__file__),ROOT/"tools/native/qwen21_qkv_numeric_probe.mm",ROOT/"tools/native/qwen21_qkv_numeric_prepare.hpp",
        ROOT/"tools/native/qwen21_qkv_lora_block_probe.mm",ROOT/"tools/native/qwen21_qkv_lora_prepare.hpp",ROOT/"tools/native/build_qwen21_qkv_numeric_probe.sh"]
    hashes={str(path.relative_to(ROOT)):sha256_file(path) for path in sources}
    descriptor=json.loads(subprocess.check_output([str(args.binary),"--describe"],text=True,timeout=30))
    if (descriptor.get("recipe")!="rope-contract-v2" or descriptor.get("primary")!="no-contract-prepare" or descriptor.get("warmups")!=3 or
            descriptor.get("samples_per_arm")!=12 or descriptor.get("runtime_build_id")!=manifest["runtime_build_id"] or
            any(descriptor.get(key)!=hashes["tools/native/"+name] for key,name in (("cpp_sha256","qwen21_qkv_numeric_probe.mm"),
                ("header_sha256","qwen21_qkv_numeric_prepare.hpp"),("helper_sha256","qwen21_qkv_lora_block_probe.mm"),("lab_sha256","qwen21_qkv_lora_prepare.hpp")))):
        p.error("current source-bound numeric probe required")
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    assets={key:sha256_file(getattr(args,key)) for key in ("binary","library","lora")}
    if args.reference:assets["reference"]=sha256_file(args.reference)
    summary=dict(schema="tc-qwen-qkv-numeric-screen-v1",status="running",case="edit" if args.reference else "generate",qualification_passed=False,
        source_sha256=hashes,asset_sha256=assets,descriptor=descriptor,runtime_build_id=manifest["runtime_build_id"],
        scope="actual original block arguments; complete stage arrays and block consumers; no full-image claim")
    write_json(args.output/"source.json",{str(path.relative_to(ROOT)):path.read_text() for path in sources});write_json(args.output/"summary.json",summary)
    observer=LoadObservation(args.output/"load.jsonl")
    try:
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            summary["memory"]=run_sampled([str(args.binary),str(args.model),str(args.lora),str(args.output/"trial"),str(args.reference) if args.reference else "-"],
                repo=ROOT,output=args.output,stem="trial",env=benchmark_environment(),stdout=stdout,stderr=stderr,timeout=1200,interval_ms=100,max_gap_ms=500,observer=observer)
        native=json.loads((args.output/"trial/native.json").read_text())
        if native["runtime_build_id"]!=manifest["runtime_build_id"]:raise ValueError("actual linked runtime differs")
        summary["analysis"]=analyze(native);summary["status"]="complete"
    except Exception as error:
        summary.update(status="failed",error=str(error))
        if (args.output/"trial/native.json").is_file():summary["native_error"]=json.loads((args.output/"trial/native.json").read_text()).get("error")
    try:summary["load"]=observer.verify()
    except ValueError as error:summary["load_error"]=str(error)
    if any(sha256_file(path)!=hashes[str(path.relative_to(ROOT))] for path in sources) or any(sha256_file(getattr(args,key))!=digest for key,digest in assets.items()):
        summary.update(status="failed",error="bound source or artifact changed during window")
    write_json(args.output/"summary.json",summary);print(json.dumps(summary,indent=2))
    if summary["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
