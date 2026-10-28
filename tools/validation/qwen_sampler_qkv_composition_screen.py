#!/usr/bin/env python3
"""Six/five authored nodes x original/direct QKV, complete matched requests.

Kernel pairs are equivalent-work controls. Five-vs-six comparisons are a
different quality tier/NFE, never a certificate of equivalent arithmetic.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np
from PIL import Image,ImageDraw

from flux9_bf16_framework_screen import write_json
import qwen_corrected_qkv_request_screen as base
from qwen_student_sampler_screen import QUALITY_CASES
from runtime_ane_common import benchmark_environment,sha256_file
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ROOT=Path(__file__).resolve().parents[2]
ARMS={"six-original":(6,False),"six-direct":(6,True),"five-original":(5,False),"five-direct":(5,True)}


def point(case,arm,seed,hit,**extra):
    steps,on=ARMS[arm]
    return dict(case=case,arm=arm,steps=steps,on=on,storage="direct" if on else "off",seed=seed,
        bank_reused=False,conditioning_hit=hit,**extra)


def points(mode,case="generate",arm="six-original"):
    if mode=="timing":return [point(case,arm,73+i%2,i>0) for i in range(6)]
    if mode=="raw":
        order=(("six-original",73),("six-direct",73),("five-original",73),("five-direct",73),("five-direct",74),
               ("five-direct",73),("six-original",73),("six-direct",73),("six-original",73))
        return [point(task,name,seed,i>0) for task in ("generate","edit") for i,(name,seed) in enumerate(order)]
    if mode!="quality":raise ValueError("known composition mode required")
    order=(("six-original",73),("six-direct",73),("five-original",73),("five-direct",73),
           ("five-direct",74),("five-original",74),("six-direct",74),("six-original",74))
    return [point(task,name,seed,i>0,case_id=case_id,prompt=prompt) for case_id,task,prompt in QUALITY_CASES
            for i,(name,seed) in enumerate(order)]


def validate(report,sequence):
    base.validate(report,sequence,base.DIRECT_RECIPE)
    for row,p in zip(report["runs"],sequence):
        if (p["steps"],p["on"])!=ARMS[p["arm"]]:raise ValueError("actual arm/sampler mapping differs")
        if row["metrics"].get("qwen_corrected_qkv"):
            q=row["metrics"]["qwen_corrected_qkv"]
            if type(q.get("completed_sampler_steps")) is not int or q["completed_sampler_steps"]!=p["steps"]:raise ValueError("all actual sampler steps must complete")


def load_images(output,count):
    result=[]
    for i in range(count):
        with Image.open(output/f"{i}.png") as image:
            image.load()
            if image.size!=(512,512):raise ValueError("complete 512 composition PNG required")
            result.append(np.asarray(image.convert("RGBA")))
    return result


def raw_equal(output,a,b,case,*,final=False):
    names=("latents",) if final else ("initial","text")+(("reference_0",) if case=="edit" else ())
    for name in names:
        values=[base.tensor(output/f"raw-{i}/qwen21_{name}.safetensors") for i in (a,b)]
        if values[0][0]!=values[1][0] or not np.array_equal(values[0][1],values[1][1]):raise ValueError("matched raw composition differs: "+name)
        if name in ("initial","latents") and (values[0][0]!="BF16" or values[0][1].shape!=(1,1024,64)):
            raise ValueError("complete BF16 image state required")


def analyze(output,report,sequence,mode,*,write_sheet=True):
    validate(report,sequence);pixels=load_images(output,len(sequence))
    if mode=="timing":
        for i in range(2,6):
            if not np.array_equal(pixels[i],pixels[i%2]):raise ValueError("fixed arm/seed PNG was not restored")
        samples=[row["request_wall_including_export"] for row in report["runs"]][1:]
        return dict(full_request_seconds=samples,median_seconds=statistics.median(samples),cold_request_seconds=report["runs"][0]["request_wall_including_export"],
            same_arm_seed_restored=True,scope="five conditioning-hot complete requests with VAE/PNG; cold/load/JIT retained separately; no dumps")
    kernel_rows=[];tier_rows=[];active_rows=[]
    if mode=="raw":
        for start in (0,9):
            task=sequence[start]["case"]
            for a,b in ((0,1),(0,6),(0,8),(1,7),(2,3),(3,5)):
                a+=start;b+=start
                raw_equal(output,a,b,task);raw_equal(output,a,b,task,final=True)
                if not np.array_equal(pixels[a],pixels[b]):raise ValueError("sampler/kernel restoration did not reproduce exact RGBA")
            raw_equal(output,start,start+2,task)
            initial=[base.tensor(output/f"raw-{start+i}/qwen21_initial.safetensors")[1] for i in (3,4)]
            if np.array_equal(initial[0],initial[1]) or np.array_equal(pixels[start+3],pixels[start+4]):raise ValueError("five/direct changed seed reused old state")
            for a,b in ((0,1),(2,3)):
                kernel_rows.append(dict(case=task,steps=sequence[start+a]["steps"],original_index=start+a,direct_index=start+b,latents_exact=True,rgba_exact=True))
            tier_rows.append(dict(case=task,seed=73,six_index=start,five_index=start+3,rgb=base.image_metrics(pixels[start][:,:,:3],pixels[start+3][:,:,:3]),
                alpha_max_abs=int(np.abs(pixels[start][:,:,3].astype(int)-pixels[start+3][:,:,3].astype(int)).max()),equivalent_work=False))
            active=[row["metrics"]["memory"]["mlx_active_bytes"] for row in report["runs"][start:start+9]]
            if any(active[i]!=active[0] for i in (1,6,7,8)) or any(active[i]!=active[2] for i in (3,4,5)):
                raise ValueError("direct introduced an extra active owner within a sampler tier")
            active_rows.append(dict(case=task,mlx_active_bytes=active,within_tier_extra_active_bytes=0))
        return dict(kernel_pairs=kernel_rows,tier_differences=tier_rows,active_ownership=active_rows,restoration_exact=True,
            scope="raw parity-only requests; original/direct exact within each tier; five differs from six, not performance evidence")
    for start in range(0,len(sequence),8):
        task=sequence[start]["case"]
        for a,b in ((0,1),(2,3),(5,4),(7,6)):
            a+=start;b+=start;raw_equal(output,a,b,task);raw_equal(output,a,b,task,final=True)
            if not np.array_equal(pixels[a],pixels[b]):raise ValueError("equivalent-work kernel pair changed the full RGBA image")
            kernel_rows.append(dict(case_id=sequence[a]["case_id"],case=task,steps=sequence[a]["steps"],seed=sequence[a]["seed"],
                original_index=a,direct_index=b,latents_exact=True,rgba_exact=True))
        for a,b in ((0,3),(7,4)):
            a+=start;b+=start;raw_equal(output,a,b,task)
            tier_rows.append(dict(case_id=sequence[a]["case_id"],case=task,seed=sequence[a]["seed"],six_index=a,five_index=b,
                rgb=base.image_metrics(pixels[a][:,:,:3],pixels[b][:,:,:3]),
                alpha_max_abs=int(np.abs(pixels[a][:,:,3].astype(int)-pixels[b][:,:,3].astype(int)).max()),equivalent_work=False))
    if write_sheet:
        sheet=Image.new("RGB",(512,len(tier_rows)*276),"white");draw=ImageDraw.Draw(sheet)
        for row,pair in enumerate(tier_rows):
            for col,key in enumerate(("six_index","five_index")):
                image=Image.fromarray(pixels[pair[key]],"RGBA");canvas=Image.new("RGBA",image.size,(220,220,220,255));canvas.alpha_composite(image)
                sheet.paste(canvas.convert("RGB").resize((256,256)),(col*256,row*276+20));draw.text((col*256+3,row*276+3),f'{pair["case_id"]} seed{pair["seed"]} '+("six-original" if col==0 else "five-direct"),fill="black")
        sheet.save(output/"sampler-quality-tier-comparison.png")
    return dict(kernel_pairs=kernel_rows,all_within_tier_latents_rgba_exact=True,tier_differences=tier_rows,instruction_quality_qualified=False,
        scope="seven tasks/two seeds/four arms, complete requests; cross-NFE differences are a quality tradeoff, not an equivalence gate or semantic score")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--library",type=Path,required=True)
    p.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--lora",type=Path,default=ROOT/"models/Viggle-Qwen-Image-2.1-viggle-turbo/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors")
    p.add_argument("--reference",type=Path,default=ROOT/"results/qwen21/teapot-blue-edit-native-512.png")
    p.add_argument("--output",type=Path,required=True);p.add_argument("--mode",choices=("raw","quality","timing"),default="raw")
    p.add_argument("--case",choices=("generate","edit"),default="generate");p.add_argument("--arm",choices=tuple(ARMS),default="six-original")
    p.add_argument("--child",action="store_true");args=p.parse_args()
    for key in ("library","model","lora","reference"):setattr(args,key,getattr(args,key).resolve(strict=True))
    args.output=args.output.resolve();args.direct=True;sequence=points(args.mode,args.case,args.arm)
    if args.child:
        report=base.run_points(args,sequence,dump=args.mode!="timing");validate(report,sequence);return
    pin=base.adapter_pin(args.lora)
    if pin["rank"]!="r128" or sha256_file(args.lora)!=pin["sha256"]:p.error("original pinned r128 required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    for item in manifest["inputs"]["sources"]:
        if sha256_file(ROOT/item["path"])!=item["sha256"]:p.error("native source differs from linked library")
    args.output.mkdir(parents=True,exist_ok=False);trial=args.output/"trial";trial.mkdir()
    sources=[Path(__file__),Path(base.__file__),ROOT/"tools/validation/qwen_student_sampler_screen.py",ROOT/"tools/validation/qwen_student_prefix_screen.py",
        ROOT/"tools/validation/qwen21_dit_cache_report.py",ROOT/"tools/validation/qwen_student_sampler_raw.py",ROOT/"tools/validation/image_framework_native.py"]
    hashes={str(path.relative_to(ROOT)):sha256_file(path) for path in sources};assets={key:sha256_file(getattr(args,key)) for key in ("library","lora","reference")}
    summary=dict(schema="tc-qwen-sampler-qkv-composition-v1",status="running",mode=args.mode,case=args.case,arm=args.arm,qualification_passed=False,
        source_sha256=hashes,asset_sha256=assets,runtime_build_id=manifest["runtime_build_id"],frozen_sequence=sequence,
        scope="actual five/six x original/direct; same-tier kernel equivalence and distinct NFE quality tier kept separate; no default/App change")
    write_json(args.output/"source.json",{str(path.relative_to(ROOT)):path.read_text() for path in sources});write_json(args.output/"summary.json",summary)
    command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child","--library",str(args.library),"--model",str(args.model),"--lora",str(args.lora),
        "--reference",str(args.reference),"--output",str(trial),"--mode",args.mode,"--case",args.case,"--arm",args.arm];observer=LoadObservation(args.output/"load.jsonl")
    try:
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            summary["memory"]=run_sampled(command,repo=ROOT,output=args.output,stem="trial",env=benchmark_environment(),stdout=stdout,stderr=stderr,
                timeout=2400,interval_ms=100,max_gap_ms=500,observer=observer)
        report=json.loads((trial/"report.json").read_text())
        if report["loaded_runtime_id"]!=manifest["runtime_build_id"]:raise ValueError("actual composition runtime differs")
        summary["analysis"]=analyze(trial,report,sequence,args.mode);summary["status"]="complete"
    except Exception as error:summary.update(status="failed",error=str(error))
    try:summary["load"]=observer.verify()
    except ValueError as error:summary["load_error"]=str(error)
    if any(sha256_file(path)!=hashes[str(path.relative_to(ROOT))] for path in sources) or any(sha256_file(getattr(args,key))!=digest for key,digest in assets.items()):
        summary.update(status="failed",error="bound composition source/asset changed")
    write_json(args.output/"summary.json",summary);print(json.dumps(summary,indent=2))
    if summary["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
