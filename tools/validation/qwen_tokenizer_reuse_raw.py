#!/usr/bin/env python3
"""Parity-only off/on captures for CPU tokenizer reuse; no speed denominator."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from flux9_bf16_framework_screen import write_json
from image_framework_native import generate_requests
from qwen_student_prefix_screen import PROMPTS,adapter_pin
from qwen_student_sampler_raw import tensor
from qwen_tokenizer_reuse_screen import FLAG,PROCESSOR_SHA
from runtime_ane_common import benchmark_environment,sha256_file
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ROOT=Path(__file__).resolve().parents[2]
POINTS=(("generate",False),("generate",True),("edit",False),("edit",True))


def child(args):
    requests=[]
    for i,(case,reuse) in enumerate(POINTS):
        request=dict(model="qwen-image-2.1",operation="image."+case,prompt=PROMPTS[case],width=512,height=512,
            steps=args.steps,frames=1,audio=False,seed=73,execution="gpu",residency="resident",allow_approximation=True,
            qwen21_dit_cache="off",lora_strategy="inference_time",loras=[dict(path=str(args.lora),strength=1.,role="transformer")],
            output=str(args.output/f"{i}.png"),dump_tensors=str(args.output/f"dump-{i}"))
        if case=="edit":request.update(qwen21_reference_size=512,
            inputs=[dict(kind="image",role="reference",path=str(args.reference))])
        requests.append(request)
    def before(i,request):
        os.environ[FLAG]=str(int(POINTS[i][1]));os.environ["TURBOCIDER_QWEN21_LORA_BF16_AB"]="1"
        os.environ["TURBOCIDER_QWEN21_STUDENT_FIVE_STEP"]=str(int(args.steps==5))
        os.environ["TURBOCIDER_QWEN21_PREFIX_SNAPSHOT"]="0"
    generate_requests(args.library,"qwen-image-2.1",args.model,requests,args.output,"tokenizer-raw-parity-only",before_request=before)


def analyze(output,steps=6):
    report=json.loads((output/"report.json").read_text())
    if len(report["runs"])!=4:raise ValueError("four complete parity-only requests required")
    for row,(case,reuse) in zip(report["runs"],POINTS):
        m=row["metrics"]
        if ((m["model"],m["operation"],m["steps"],m["actual_denoise_steps"],m["seed"],m["runtime_backend"],m["runtime_precision"],
             m["encoder_gpu_graph"],m["lora_applied_projections"])!=
                ("qwen-image-2.1","image."+case,steps,steps,73,"mlx_cpp_metal","bf16","qwen3_vl_deepstack_gpu_only",227) or
                "joint BF16 LoRA A/B operands" not in m["acceleration_selection"] or m["prompt_cache_hit"] is not False):
            raise ValueError("raw capture work differs")
        cache=m.get("qwen_tokenizer_reuse")
        if (cache is not None)!=reuse:raise ValueError("actual raw CPU policy differs")
        if reuse and (cache["processor_sha256"]!=PROCESSOR_SHA or cache["constructed_this_request"] is not True or
                      cache["reused"] is not False or cache["tokenized_this_request"] is not True):
            raise ValueError("off -> on did not create a new verified CPU owner")
    rows=[]
    for case,a,b in (("generate",0,1),("edit",2,3)):
        checks=[]
        for name in ("initial","text","latents","pixels")+(("reference_0",) if case=="edit" else ()):
            paths=[output/f"dump-{i}"/f"qwen21_{name}.safetensors" for i in (a,b)]
            values=[tensor(path) for path in paths]
            if values[0][0]!=values[1][0] or not np.array_equal(values[0][1],values[1][1]):
                raise ValueError("CPU cache changed full raw tensor: "+case+"/"+name)
            checks.append(dict(name=name,dtype=values[0][0],shape=list(values[0][1].shape),exact=True,
                               file_sha256=[sha256_file(path) for path in paths]))
        images=[]
        for i in (a,b):
            with Image.open(output/f"{i}.png") as image:
                image.load()
                if image.size!=(512,512):raise ValueError("complete output required")
                images.append(np.asarray(image.convert("RGBA")))
        if not np.array_equal(*images):raise ValueError("raw CPU arm changed final RGBA")
        rows.append(dict(case=case,raw_checks=checks,rgba_exact=True))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--lora",type=Path,default=ROOT/"models/Viggle-Qwen-Image-2.1-viggle-turbo/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors")
    p.add_argument("--reference",type=Path,default=ROOT/"results/qwen21/teapot-blue-edit-native-512.png")
    p.add_argument("--output",type=Path,required=True);p.add_argument("--child",action="store_true")
    p.add_argument("--steps",type=int,choices=(5,6),default=6)
    args=p.parse_args()
    for key in ("library","model","lora","reference"):setattr(args,key,getattr(args,key).resolve(strict=True))
    args.output=args.output.resolve()
    if args.child:child(args);return
    processor=args.model/"processor/tokenizer.json";pin=adapter_pin(args.lora)
    if sha256_file(processor)!=PROCESSOR_SHA or pin["rank"]!="r128" or sha256_file(args.lora)!=pin["sha256"]:
        p.error("fixed original processor and pinned r128 required")
    args.output.mkdir(parents=True,exist_ok=False);trial=args.output/"trial";trial.mkdir()
    paths=[Path(__file__),ROOT/"tools/validation/qwen_tokenizer_reuse_screen.py",
           ROOT/"tools/validation/qwen_student_sampler_raw.py",ROOT/"tools/validation/image_framework_native.py"]
    hashes={str(path.relative_to(ROOT)):sha256_file(path) for path in paths}
    assets={key:sha256_file(getattr(args,key)) for key in ("library","lora","reference")};assets["processor"]=sha256_file(processor)
    summary=dict(schema="tc-qwen-tokenizer-reuse-raw-v1",status="running",qualification_passed=False,
                 source_sha256=hashes,asset_sha256=assets,steps=args.steps,performance_denominator=False)
    write_json(args.output/"source.json",{str(path.relative_to(ROOT)):path.read_text() for path in paths})
    write_json(args.output/"summary.json",summary)
    command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child","--library",str(args.library),"--model",str(args.model),
             "--lora",str(args.lora),"--reference",str(args.reference),"--output",str(trial),"--steps",str(args.steps)]
    observer=LoadObservation(args.output/"load.jsonl")
    try:
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            summary["memory"]=run_sampled(command,repo=ROOT,output=args.output,stem="trial",env=benchmark_environment(),
                stdout=stdout,stderr=stderr,timeout=600,interval_ms=100,max_gap_ms=500,observer=observer)
        summary["analysis"]=analyze(trial,args.steps);summary["status"]="complete"
    except Exception as error:summary.update(status="failed",error=str(error))
    try:summary["load"]=observer.verify()
    except ValueError as error:summary["load_error"]=str(error)
    if any(sha256_file(ROOT/key)!=digest for key,digest in hashes.items()) or any(
            sha256_file(processor if key=="processor" else getattr(args,key))!=digest for key,digest in assets.items()):
        summary.update(status="failed",error="source/asset changed during parity capture")
    write_json(args.output/"summary.json",summary);print(json.dumps(summary,indent=2))
    if summary["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
