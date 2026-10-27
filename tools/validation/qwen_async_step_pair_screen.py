#!/usr/bin/env python3
"""Full sync/async pair controls, preserving every sampler/finite operation."""
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
import qwen_lora_swiglu_request_screen as base
from runtime_ane_common import benchmark_environment,sha256_file
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ROOT=Path(__file__).resolve().parents[2]
FLAG="TURBOCIDER_QWEN21_ASYNC_STEP_PAIR"
SOURCES=("tools/validation/qwen_async_step_pair_screen.py",)+base.SOURCES


class Sequence:
    def __init__(self):self.points=[];self.condition={};self.bank=None
    def add(self,case="edit",steps=6,async_pair=False,prefix=True,seed=73,prompt=None,case_id=None,reference="primary",qkv=True,ffn=True):
        prompt=prompt or base.PROMPTS[case];reference=reference if case=="edit" else None
        condition=(prompt,reference);hit=self.condition.get(case)==condition;self.condition[case]=condition
        bank=(case,condition,steps,qkv,ffn,async_pair) if prefix else None
        phit=self.bank==bank if prefix else None;self.bank=bank
        self.points.append(dict(case=case,steps=steps,async_pair=async_pair,prefix=prefix,seed=seed,prompt=prompt,case_id=case_id or ("photo" if case=="generate" else "edit-color"),
            reference=reference,qkv=qkv,ffn=ffn,conditioning_hit=hit,prefix_hit=phit))


def points(mode,case="generate",steps=5,async_pair=False,prefix=False):
    s=Sequence()
    if mode=="timing":
        if case=="generate" and prefix:raise ValueError("generation cannot claim edit prefix hit")
        for i in range(6):s.add(case,steps,async_pair,prefix,73+i%2)
    elif mode=="raw":
        for task in ("generate","edit"):
            for tier in (6,5):
                for qkv in (False,True):
                    for ffn in (False,True):
                        for enabled in (False,True):s.add(task,tier,enabled,False,qkv=qkv,ffn=ffn)
    elif mode=="quality":
        for cid,task,prompt in base.QUALITY_CASES:
            for tier in (6,5):
                for enabled,seed in ((False,73),(True,73),(True,74),(False,74)):s.add(task,tier,enabled,False,seed,prompt,cid)
                if task=="edit":
                    for enabled in (False,True):
                        for seed in (73,73,74):s.add(task,tier,enabled,True,seed,prompt,cid)
    elif mode=="restoration":
        for tier in (6,5):
            for enabled,seed in ((False,73),(False,73),(True,73),(True,73),(True,74),(True,73),(False,73),(False,73)):
                s.add(steps=tier,async_pair=enabled,seed=seed)
        for _ in range(2):s.add(async_pair=True)
        s.add(async_pair=True,prefix=False)
        for _ in range(2):s.add(async_pair=True)
        material=next(prompt for cid,_,prompt in base.QUALITY_CASES if cid=="edit-material")
        for prompt in (material,base.PROMPTS["edit"]):
            for _ in range(2):s.add(async_pair=True,prompt=prompt)
        for reference in ("alternate","primary"):
            for _ in range(2):s.add(async_pair=True,reference=reference)
        for enabled,seed in ((False,73),(True,73),(True,74),(True,73),(False,73)):
            s.add("generate",6,enabled,False,seed)
        for _ in range(2):s.add(async_pair=True)
    else:raise ValueError("known bounded async workload required")
    return s.points


def validate(report,sequence):
    base.validate(report,sequence)
    for i,(row,p) in enumerate(zip(report["runs"],sequence)):
        m=row["metrics"];a=m.get("qwen_async_step_pair")
        if not p["async_pair"]:
            if a is not None:raise ValueError("sync control claimed async step work")
            continue
        if (not a or a.get("enabled") is not True or a.get("recipe")!="bounded-async-step-pair-v1" or a.get("max_batch_steps")!=2 or
                any(type(a.get(name)) is not int or a[name]!=p["steps"] for name in ("submitted_steps","checked_steps","accepted_steps")) or
                type(a.get("completion_batches")) is not int or a["completion_batches"]!=(p["steps"]+1)//2 or
                type(a.get("evaluation_calls")) is not int or a["evaluation_calls"]!=2*p["steps"] or type(a.get("peak_pending_steps")) is not int or a["peak_pending_steps"]!=2 or
                type(a.get("peak_logical_state_bytes")) is not int or a["peak_logical_state_bytes"]!=262146 or type(a.get("host_ahead_observations")) is not int or not 0<=a["host_ahead_observations"]<=p["steps"]//2 or
                a.get("final_drain_complete") is not True or a.get("diagnostic_submit_observer") is not False or
                a.get("per_step_finite_masks") is not True or a.get("progress_after_checked_completion") is not True or type(a.get("extra_weight_bytes")) is not int or a["extra_weight_bytes"]!=0 or
                "qwen21_bounded_async_step_pair_experimental" not in m["plan"]["algorithm_approximations"] or
                "bounded async step pairs" not in m["acceleration_selection"]):
            raise ValueError(f"actual bounded submit/check/drain/complete coverage differs at request {i}")


def signature(p,include_async=False):
    value=base.signature(p,with_ffn=True)
    return value+(p["async_pair"],) if include_async else value


def analyze(output,report,seq,mode):
    validate(report,seq);pixels=base.prefix_screen.images(output,len(seq))
    if mode=="timing":
        for i in range(3,6):
            if not np.array_equal(pixels[i],pixels[1+(i-1)%2]):raise ValueError("async/sync hot seed restore changed complete RGBA")
        values=[row["request_wall_including_export"] for row in report["runs"]][1:]
        return dict(full_request_seconds=values,median_seconds=statistics.median(values),first_request_seconds=report["runs"][0]["request_wall_including_export"],
            hot_seed_restored=True,scope="five complete hot requests including original finite operations, VAE/PNG; first load/JIT/prime separate")
    groups={};repeat=0
    for i,p in enumerate(seq):
        group=groups.setdefault(signature(p),{})
        if p["async_pair"] in group:
            if not np.array_equal(pixels[i],pixels[group[p["async_pair"]]]):raise ValueError("async/sampler/condition/reference restoration changed PNG")
            repeat+=1
        else:group[p["async_pair"]]=i
    pairs=[]
    for g in groups.values():
        if set(g)!={False,True}:continue
        x,y=g[False],g[True];p=seq[x]
        if not np.array_equal(pixels[x],pixels[y]):raise ValueError("bounded async scheduling changed complete RGBA")
        if mode=="raw":base.raw_match(output,x,y,p)
        pairs.append(dict(case=p["case"],case_id=p["case_id"],steps=p["steps"],qkv=p["qkv"],ffn=p["ffn"],prefix=p["prefix"],prefix_hit=p["prefix_hit"],seed=p["seed"],
            off_index=x,on_index=y,rgba_exact=True,raw_latents_exact=mode=="raw"))
    if mode=="raw" and len(pairs)!=16:raise ValueError("all four operator policies and both task/tier raw controls required")
    if mode=="quality" and len(pairs)!=46:raise ValueError("seven tasks/two tiers/same-prefix sync/async complete controls required")
    if mode=="restoration":
        for task,tier in (("generate",6),("edit",6),("edit",5)):
            indices=[i for i,p in enumerate(seq) if p["async_pair"] and p["case"]==task and p["steps"]==tier and p["reference"] in (None,"primary") and p["prompt"]==base.PROMPTS[task] and (task=="generate" or p["prefix_hit"] is True)]
            a=next(i for i in indices if seq[i]["seed"]==73);b=next(i for i in indices if seq[i]["seed"]==74)
            if np.array_equal(pixels[a],pixels[b]):raise ValueError("changed seed reused old output")
    return dict(kernel_pairs=pairs,repeated_state_exact_pairs=repeat,all_matched_rgba_exact=True,raw_latents_exact=mode=="raw",
        scope="same sampler/conditioning/operator/prefix controls; no cross-NFE or semantic quality equivalence, no dump performance denominator")


def child(args,seq):
    lib=C.CDLL(str(args.library));lib.tc_runtime_build_identity.restype=C.c_void_p;lib.tc_string_free.argtypes=[C.c_void_p]
    ptr=lib.tc_runtime_build_identity()
    if not ptr:raise ValueError("loaded runtime identity missing")
    try:runtime=C.string_at(ptr).decode()
    finally:lib.tc_string_free(ptr)
    write_json(args.output/"loaded-runtime.json",dict(runtime_build_id=runtime));requests=[]
    for i,p in enumerate(seq):
        r=dict(model="qwen-image-2.1",operation="image."+p["case"],prompt=p["prompt"],width=512,height=512,steps=p["steps"],frames=1,audio=False,
            seed=p["seed"],execution="gpu",residency="resident",allow_approximation=True,qwen21_dit_cache="off",lora_strategy="inference_time",
            loras=[dict(path=str(args.lora),strength=1.,role="transformer")],output=str(args.output/f"{i}.png"))
        if p["case"]=="edit":r.update(qwen21_reference_size=512,inputs=[dict(kind="image",role="reference",path=str(args.alternate_reference if p["reference"]=="alternate" else args.reference))])
        if args.mode=="raw":r["dump_tensors"]=str(args.output/f"raw-{i}")
        requests.append(r)
    def before(i,request):
        p=seq[i]
        for name,value in {FLAG:p["async_pair"],"TURBOCIDER_QWEN21_LORA_SWIGLU":p["ffn"],"TURBOCIDER_QWEN21_CORRECTED_QKV":p["qkv"],
            "TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT":p["qkv"],"TURBOCIDER_QWEN21_CORRECTED_QKV_PREFIX":p["qkv"] and p["prefix"],
            "TURBOCIDER_QWEN21_PREFIX_SNAPSHOT":p["prefix"],"TURBOCIDER_QWEN21_LORA_BF16_AB":True,"TURBOCIDER_QWEN21_STUDENT_FIVE_STEP":p["steps"]==5}.items():
            os.environ[name]=str(int(value))
        print(json.dumps(dict(index=i,**p)),flush=True)
    report=generate_requests(args.library,"qwen-image-2.1",args.model,requests,args.output,"bounded-async-step-pairs",before_request=before)
    report["loaded_runtime_id"]=runtime;write_json(args.output/"report.json",report);validate(report,seq)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--library",type=Path,required=True)
    p.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--lora",type=Path,default=ROOT/"models/Viggle-Qwen-Image-2.1-viggle-turbo/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors")
    p.add_argument("--reference",type=Path,default=ROOT/"results/qwen21/teapot-blue-edit-native-512.png");p.add_argument("--alternate-reference",type=Path)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--mode",choices=("raw","quality","restoration","timing"),default="raw")
    p.add_argument("--case",choices=("generate","edit"),default="generate");p.add_argument("--steps",type=int,choices=(5,6),default=5)
    p.add_argument("--async-pair",choices=("off","on"),default="off");p.add_argument("--prefix",action="store_true");p.add_argument("--child",action="store_true");args=p.parse_args()
    for name in ("library","model","lora","reference"):setattr(args,name,getattr(args,name).resolve(strict=True))
    if args.mode=="restoration":
        args.alternate_reference=(args.alternate_reference or ROOT/"outputs/qwen512-qkv-prefix-quality-20261018/trial/8.png").resolve(strict=True)
        if sha256_file(args.reference)==sha256_file(args.alternate_reference):p.error("alternate reference must change input bytes")
    args.output=args.output.resolve();seq=points(args.mode,args.case,args.steps,args.async_pair=="on",args.prefix)
    if args.child:child(args,seq);return
    pin=base.adapter_pin(args.lora)
    if pin["rank"]!="r128" or sha256_file(args.lora)!=pin["sha256"]:p.error("original released r128 required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    def native_sources():
        for entry in manifest["inputs"]["sources"]:
            if sha256_file(ROOT/entry["path"])!=entry["sha256"]:raise ValueError("native source differs from linked library")
    native_sources();args.output.mkdir(parents=True,exist_ok=False);trial=args.output/"trial";trial.mkdir()
    hashes={name:sha256_file(ROOT/name) for name in SOURCES};assets={name:sha256_file(getattr(args,name)) for name in ("library","lora","reference")}
    if args.alternate_reference:assets["alternate_reference"]=sha256_file(args.alternate_reference)
    s=dict(schema="tc-qwen-async-step-pair-screen-v1",status="running",qualification_passed=False,mode=args.mode,case=args.case,steps=args.steps,async_pair=args.async_pair,prefix=args.prefix,
        frozen_sequence=seq,source_sha256=hashes,asset_sha256=assets,runtime_build_id=manifest["runtime_build_id"])
    write_json(args.output/"source.json",{name:(ROOT/name).read_text() for name in SOURCES});write_json(args.output/"summary.json",s)
    command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child","--library",str(args.library),"--model",str(args.model),"--lora",str(args.lora),"--reference",str(args.reference),
        "--output",str(trial),"--mode",args.mode,"--case",args.case,"--steps",str(args.steps),"--async-pair",args.async_pair]
    if args.prefix:command.append("--prefix")
    if args.alternate_reference:command.extend(["--alternate-reference",str(args.alternate_reference)])
    observer=LoadObservation(args.output/"load.jsonl")
    try:
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            s["memory"]=run_sampled(command,repo=ROOT,output=args.output,stem="trial",env=benchmark_environment(),stdout=stdout,stderr=stderr,
                timeout=3600,interval_ms=100,max_gap_ms=500,observer=observer)
        report=json.loads((trial/"report.json").read_text())
        if report["loaded_runtime_id"]!=manifest["runtime_build_id"]:raise ValueError("actual loaded runtime differs")
        s["analysis"]=analyze(trial,report,seq,args.mode);native_sources();s["status"]="complete"
    except Exception as error:s.update(status="failed",error=str(error))
    try:s["load"]=observer.verify()
    except ValueError as error:s["load_error"]=str(error)
    if any(sha256_file(ROOT/name)!=digest for name,digest in hashes.items()) or any(sha256_file(getattr(args,name))!=digest for name,digest in assets.items()):
        s.update(status="failed",error="bound async workload source/asset changed")
    write_json(args.output/"summary.json",s);print(json.dumps(s,indent=2))
    if s["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
