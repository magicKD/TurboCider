#!/usr/bin/env python3
"""Complete 512 requests: ordinary/direct QKV x typed-SwiGLU, both samplers.

Raw input/latent parity, broader RGBA controls, independent cache lifetimes,
then fixed-arm no-dump request timing. No cross-NFE equivalence inference.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import statistics
import sys

import numpy as np

from flux9_bf16_framework_screen import write_json
from image_framework_native import generate_requests
import qwen_qkv_prefix_composition_screen as prefix_screen
from qwen_student_prefix_screen import PROMPTS,adapter_pin
from qwen_student_sampler_screen import QUALITY_CASES
from qwen_student_sampler_raw import tensor
from qwen21_dit_cache_report import image_metrics
from runtime_ane_common import benchmark_environment,sha256_file
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ROOT=Path(__file__).resolve().parents[2]
FLAG="TURBOCIDER_QWEN21_LORA_SWIGLU"
RECIPE="native-base-paired-b-swiglu-b16-v1"
SOURCES=("tools/validation/qwen_lora_swiglu_request_screen.py","tools/validation/image_framework_native.py",
    "tools/validation/qwen_qkv_prefix_composition_screen.py","tools/validation/qwen_student_prefix_screen.py",
    "tools/validation/qwen_student_sampler_screen.py","tools/validation/qwen_student_sampler_raw.py",
    "tools/validation/qwen21_dit_cache_report.py","tools/validation/flux9_bf16_framework_screen.py",
    "tools/validation/runtime_ane_common.py","tools/validation/runtime_ane_memory.py","tools/validation/runtime_ane_load.py",
    "tools/native/process_tree_sampler.py")


class Sequence:
    def __init__(self):self.points=[];self.condition={};self.bank=None

    def add(self,case="edit",steps=6,qkv=True,ffn=True,prefix=True,seed=73,prompt=None,case_id=None,reference="primary"):
        prompt=prompt or PROMPTS[case];reference=reference if case=="edit" else None
        condition=(prompt,reference);hit=self.condition.get(case)==condition;self.condition[case]=condition
        bank=(case,condition,steps,qkv,ffn) if prefix else None
        prefix_hit=self.bank==bank if prefix else None;self.bank=bank
        self.points.append(dict(case=case,steps=steps,qkv=qkv,ffn=ffn,prefix=prefix,seed=seed,prompt=prompt,
            case_id=case_id or ("photo" if case=="generate" else "edit-color"),reference=reference,conditioning_hit=hit,prefix_hit=prefix_hit))


def points(mode,case="generate",steps=5,ffn=False,prefix=False):
    s=Sequence()
    if mode=="timing":
        if case=="generate" and prefix:raise ValueError("generation cannot claim an edit prefix")
        for i in range(6):s.add(case,steps,True,ffn,prefix,73+i%2)
    elif mode=="raw":
        for task in ("generate","edit"):
            for tier in (6,5):
                for qkv in (False,True):
                    for fused in (False,True):s.add(task,tier,qkv,fused,False)
    elif mode=="quality":
        for cid,task,prompt in QUALITY_CASES:
            for tier in (6,5):
                for qkv,fused,seed in ((False,False,73),(False,True,73),(True,False,73),(True,True,73),
                                       (True,True,74),(True,False,74),(False,True,74),(False,False,74)):
                    s.add(task,tier,qkv,fused,False,seed,prompt,cid)
                if task=="edit":
                    for qkv,fused in ((False,False),(False,True),(True,False),(True,True)):
                        for seed in (73,73,74):s.add(task,tier,qkv,fused,True,seed,prompt,cid)
    elif mode=="restoration":
        for tier in (6,5):
            for fused,seed in ((False,73),(False,73),(True,73),(True,73),(True,74),(True,73),(False,73),(False,73)):
                s.add(steps=tier,ffn=fused,seed=seed)
        for _ in range(2):s.add()
        s.add(prefix=False)
        for _ in range(2):s.add()
        for qkv,fused in ((False,True),(False,False),(True,True)):
            for _ in range(2):s.add(qkv=qkv,ffn=fused)
        material=next(prompt for cid,_,prompt in QUALITY_CASES if cid=="edit-material")
        for prompt in (material,PROMPTS["edit"]):
            for _ in range(2):s.add(prompt=prompt)
        for reference in ("alternate","primary"):
            for _ in range(2):s.add(reference=reference)
        for fused,seed in ((True,73),(False,73),(True,73),(True,74),(True,73)):
            s.add("generate",6,True,fused,False,seed)
        for _ in range(2):s.add()
    else:raise ValueError("known frozen complete request mode required")
    return s.points


def base_arm(p):return "combined" if p["qkv"] and p["prefix"] else "direct" if p["qkv"] else "prefix" if p["prefix"] else "original"


def validate(report,sequence):
    prefix_screen.validate(report,[{**p,"arm":base_arm(p)} for p in sequence])
    for i,(row,p) in enumerate(zip(report["runs"],sequence)):
        m=row["metrics"];f=m.get("qwen_lora_swiglu")
        if not p["ffn"]:
            if f is not None:raise ValueError("ordinary FFN control claimed fusion")
            continue
        if (not f or f.get("enabled") is not True or f.get("recipe")!=RECIPE or f.get("bm")!=16 or f.get("bn")!=128 or
                type(f.get("completed_decode_blocks")) is not int or f["completed_decode_blocks"]!=32*(p["steps"]-(0 if p["prefix_hit"] is True else 1)) or
                type(f.get("completed_sampler_steps")) is not int or f["completed_sampler_steps"]!=p["steps"] or
                type(f.get("completed_prefill_steps")) is not int or f["completed_prefill_steps"]!=(0 if p["prefix_hit"] is True else 1) or
                f.get("prefix_snapshot_enabled") is not p["prefix"] or f.get("prefix_snapshot_hit") is not (p["prefix_hit"] is True) or
                type(f.get("extra_weight_bytes")) is not int or f["extra_weight_bytes"]!=0 or f.get("native_base_gemm") is not True or
                f.get("online_independent_ranks") is not True or f.get("rank_math")!="f32" or f.get("complete_down_lora") is not True or f.get("prefill_changed") is not False or
                "qwen21_typed_lora_swiglu_decode_experimental" not in m["plan"]["algorithm_approximations"] or
                "typed LoRA B/SwiGLU decode fusion" not in m["acceleration_selection"]):
            raise ValueError(f"complete actual typed SwiGLU consumer coverage differs at request {i}")


def signature(p,*,with_ffn=False,with_prefix=True,with_hit=True):
    names=["case","prompt","reference","steps","qkv","seed"]
    if with_ffn:names.append("ffn")
    if with_prefix:names.append("prefix")
    if with_hit:names.append("prefix_hit")
    return tuple(p[name] for name in names)


def raw_match(output,a,b,p):
    for name in ("initial","text","latents")+(("reference_0",) if p["case"]=="edit" else ()):
        x,y=(tensor(output/f"raw-{i}/qwen21_{name}.safetensors") for i in (a,b))
        if x[0]!=y[0] or not np.array_equal(x[1],y[1]):raise ValueError("complete raw tensor changed: "+name)
        if name in ("initial","latents") and (x[0]!="BF16" or x[1].shape!=(1,1024,64)):raise ValueError("complete BF16 image state required")


def analyze(output,report,sequence,mode):
    validate(report,sequence);pixels=prefix_screen.images(output,len(sequence))
    if mode=="timing":
        for i in range(3,6):
            if not np.array_equal(pixels[i],pixels[1+(i-1)%2]):raise ValueError("hot same-policy seed restoration differs")
        samples=[row["request_wall_including_export"] for row in report["runs"]][1:]
        return dict(full_request_seconds=samples,median_seconds=statistics.median(samples),cold_request_seconds=report["runs"][0]["request_wall_including_export"],
                    hot_seed_restored=True,scope="five complete conditioning-hot requests with VAE/PNG; prefix arms actual hits; first load/JIT/prime separate")
    groups={};repeated=0;pairs=[]
    for i,p in enumerate(sequence):
        group=groups.setdefault(signature(p),{})
        if p["ffn"] in group:
            j=group[p["ffn"]]
            if not np.array_equal(pixels[i],pixels[j]):raise ValueError("policy/sampler/condition/operation restore changed RGBA")
            repeated+=1
        else:group[p["ffn"]]=i
    for group in groups.values():
        if set(group)!={False,True}:continue
        a,b=group[False],group[True];p=sequence[a]
        if not np.array_equal(pixels[a],pixels[b]):raise ValueError("same QKV/prefix/sampler fusion changed complete RGBA")
        if mode=="raw":raw_match(output,a,b,p)
        pairs.append(dict(case=p["case"],case_id=p["case_id"],steps=p["steps"],qkv=p["qkv"],prefix=p["prefix"],prefix_hit=p["prefix_hit"],seed=p["seed"],
            off_index=a,on_index=b,rgba_exact=True,raw_latents_exact=mode=="raw"))
    if mode=="restoration":
        for case,steps in (("generate",6),("edit",6),("edit",5)):
            a=next(i for i,p in enumerate(sequence) if p["case"]==case and p["steps"]==steps and p["ffn"] and p["seed"]==73 and (case=="generate" or p["prefix_hit"] is True))
            b=next(i for i,p in enumerate(sequence) if p["case"]==case and p["steps"]==steps and p["ffn"] and p["seed"]==74 and (case=="generate" or p["prefix_hit"] is True))
            if np.array_equal(pixels[a],pixels[b]):raise ValueError("changed seed reused an old target")
        return dict(kernel_pairs=pairs,repeated_state_exact_pairs=repeated,changed_seed_recomputed=True,
                    scope="complete RGBA cache/policy/ref/prompt/sampler/operation lifecycle; no timing qualification or cached full-image latent capture")
    if mode=="raw":
        if len(pairs)!=8:raise ValueError("both sampler/task/QKV original/fused raw pairs required")
        return dict(kernel_pairs=pairs,all_raw_inputs_latents_rgba_exact=True,scope="raw parity requests only; no performance denominators or cross-NFE final equivalence")
    if len(pairs)!=92:raise ValueError("complete seven-task two-tier ordinary/direct/prefix fusion matrix required")
    off={signature(p,with_ffn=True,with_prefix=False,with_hit=False):i for i,p in enumerate(sequence) if not p["prefix"]}
    rounding=[]
    for i,p in enumerate(sequence):
        if not p["prefix"]:continue
        j=off[signature(p,with_ffn=True,with_prefix=False,with_hit=False)]
        if p["prefix_hit"] is False:
            if not np.array_equal(pixels[i],pixels[j]):raise ValueError("prefix prime changed matched native full prefill")
        else:
            rgb=image_metrics(pixels[j][:,:,:3],pixels[i][:,:,:3]);alpha=int(np.abs(pixels[j][:,:,3].astype(int)-pixels[i][:,:,3].astype(int)).max())
            rounding.append(dict(steps=p["steps"],case_id=p["case_id"],qkv=p["qkv"],ffn=p["ffn"],seed=p["seed"],rgb=rgb,alpha_max_abs=alpha,rgba_exact=bool(np.array_equal(pixels[j],pixels[i]))))
    if any(r["rgb"]["ssim"]<.99 or r["alpha_max_abs"]!=0 for r in rounding):raise ValueError("existing prefix pixel gate failed; no gate relaxation")
    return dict(kernel_pairs=pairs,all_same_tier_rgba_exact=True,prefix_rounding=rounding,prime_full_prefill_exact=True,instruction_quality_qualified=False,
                scope="full original/direct x fusion matrix, both tiers and existing prefix paths; no semantic/cross-NFE equivalence or performance claim")


def child(args,sequence):
    lib=C.CDLL(str(args.library));lib.tc_runtime_build_identity.restype=C.c_void_p;lib.tc_string_free.argtypes=[C.c_void_p]
    identity=lib.tc_runtime_build_identity()
    if not identity:raise ValueError("loaded runtime identity missing")
    try:runtime=C.string_at(identity).decode()
    finally:lib.tc_string_free(identity)
    write_json(args.output/"loaded-runtime.json",dict(runtime_build_id=runtime))
    requests=[]
    for i,p in enumerate(sequence):
        r=dict(model="qwen-image-2.1",operation="image."+p["case"],prompt=p["prompt"],width=512,height=512,steps=p["steps"],frames=1,audio=False,seed=p["seed"],
            execution="gpu",residency="resident",allow_approximation=True,qwen21_dit_cache="off",lora_strategy="inference_time",
            loras=[dict(path=str(args.lora),strength=1.,role="transformer")],output=str(args.output/f"{i}.png"))
        if p["case"]=="edit":r.update(qwen21_reference_size=512,inputs=[dict(kind="image",role="reference",path=str(args.alternate_reference if p["reference"]=="alternate" else args.reference))])
        if args.mode=="raw":r["dump_tensors"]=str(args.output/f"raw-{i}")
        requests.append(r)
    def before(i,request):
        p=sequence[i]
        for name,value in {FLAG:p["ffn"],"TURBOCIDER_QWEN21_CORRECTED_QKV":p["qkv"],"TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT":p["qkv"],
            "TURBOCIDER_QWEN21_CORRECTED_QKV_PREFIX":p["qkv"] and p["prefix"],"TURBOCIDER_QWEN21_PREFIX_SNAPSHOT":p["prefix"],
            "TURBOCIDER_QWEN21_LORA_BF16_AB":True,"TURBOCIDER_QWEN21_STUDENT_FIVE_STEP":p["steps"]==5}.items():os.environ[name]=str(int(value))
        print(json.dumps(dict(index=i,**p)),flush=True)
    report=generate_requests(args.library,"qwen-image-2.1",args.model,requests,args.output,"typed-swiglu-complete-request",before_request=before)
    report["loaded_runtime_id"]=runtime;write_json(args.output/"report.json",report);validate(report,sequence)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--library",type=Path,required=True)
    p.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--lora",type=Path,default=ROOT/"models/Viggle-Qwen-Image-2.1-viggle-turbo/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors")
    p.add_argument("--reference",type=Path,default=ROOT/"results/qwen21/teapot-blue-edit-native-512.png")
    p.add_argument("--alternate-reference",type=Path);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mode",choices=("raw","restoration","quality","timing"),default="raw");p.add_argument("--case",choices=("generate","edit"),default="generate")
    p.add_argument("--steps",type=int,choices=(5,6),default=5);p.add_argument("--ffn",choices=("off","on"),default="off");p.add_argument("--prefix",action="store_true");p.add_argument("--child",action="store_true");args=p.parse_args()
    for name in ("library","model","lora","reference"):setattr(args,name,getattr(args,name).resolve(strict=True))
    if args.mode=="restoration":
        args.alternate_reference=(args.alternate_reference or ROOT/"outputs/qwen512-qkv-prefix-quality-20261018/trial/8.png").resolve(strict=True)
        if sha256_file(args.alternate_reference)==sha256_file(args.reference):p.error("alternate reference must have different image bytes")
    args.output=args.output.resolve();sequence=points(args.mode,args.case,args.steps,args.ffn=="on",args.prefix)
    if args.child:child(args,sequence);return
    pin=adapter_pin(args.lora)
    if pin["rank"]!="r128" or sha256_file(args.lora)!=pin["sha256"]:p.error("original released pinned r128 required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    def native_sources():
        for item in manifest["inputs"]["sources"]:
            if sha256_file(ROOT/item["path"])!=item["sha256"]:raise ValueError("native source differs from linked build")
    native_sources();args.output.mkdir(parents=True,exist_ok=False);trial=args.output/"trial";trial.mkdir()
    hashes={name:sha256_file(ROOT/name) for name in SOURCES};assets={name:sha256_file(getattr(args,name)) for name in ("library","lora","reference")}
    if args.alternate_reference:assets["alternate_reference"]=sha256_file(args.alternate_reference)
    summary=dict(schema="tc-qwen-lora-swiglu-request-v1",status="running",qualification_passed=False,mode=args.mode,case=args.case,steps=args.steps,ffn=args.ffn,prefix=args.prefix,
        frozen_sequence=sequence,source_sha256=hashes,asset_sha256=assets,runtime_build_id=manifest["runtime_build_id"],scope="default-off full requests; actual samplers, VAE/PNG, all source/coverage checks; no App/default or physical-overlap claim")
    write_json(args.output/"source.json",{name:(ROOT/name).read_text() for name in SOURCES});write_json(args.output/"summary.json",summary)
    command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child","--library",str(args.library),"--model",str(args.model),"--lora",str(args.lora),"--reference",str(args.reference),
        "--output",str(trial),"--mode",args.mode,"--case",args.case,"--steps",str(args.steps),"--ffn",args.ffn]
    if args.prefix:command.append("--prefix")
    if args.alternate_reference:command.extend(["--alternate-reference",str(args.alternate_reference)])
    observer=LoadObservation(args.output/"load.jsonl")
    try:
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            summary["memory"]=run_sampled(command,repo=ROOT,output=args.output,stem="trial",env=benchmark_environment(),stdout=stdout,stderr=stderr,
                timeout=4800,interval_ms=100,max_gap_ms=500,observer=observer)
        report=json.loads((trial/"report.json").read_text())
        if report["loaded_runtime_id"]!=manifest["runtime_build_id"]:raise ValueError("actual loaded runtime differs")
        summary["analysis"]=analyze(trial,report,sequence,args.mode);native_sources();summary["status"]="complete"
    except Exception as error:summary.update(status="failed",error=str(error))
    try:summary["load"]=observer.verify()
    except ValueError as error:summary["load_error"]=str(error)
    if any(sha256_file(ROOT/name)!=digest for name,digest in hashes.items()) or any(sha256_file(getattr(args,name))!=digest for name,digest in assets.items()):
        summary.update(status="failed",error="bound request source/asset changed")
    write_json(args.output/"summary.json",summary);print(json.dumps(summary,indent=2))
    if summary["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
