#!/usr/bin/env python3
"""Four diagnostic captures: same raw noise/conditioning, different sampler.

Dump/export makes these parity-only requests, never performance denominators.
Uses the same pinned r128, joint A/B F32 ranks and original BF16 components.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np

from flux9_bf16_framework_screen import write_json
from image_framework_native import generate_requests
from qwen_student_prefix_screen import PROMPTS,adapter_pin
from qwen_student_sampler_screen import FLAG,FIVE,SIX
from runtime_ane_common import benchmark_environment,sha256_file
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ROOT=Path(__file__).resolve().parents[2]
POINTS=(("generate",6),("generate",5),("edit",6),("edit",5))


def tensor(path):
    with path.open("rb") as stream:
        size=struct.unpack("<Q",stream.read(8))[0]
        if not 2<=size<=1<<20: raise ValueError("bounded header required")
        header=json.loads(stream.read(size))
        value=header["tensor"]; dtype={"BF16":"<u2","F32":"<f4"}[value["dtype"]]
        a,b=value["data_offsets"]; stream.seek(8+size+a); payload=stream.read(b-a)
        if len(payload)!=b-a: raise ValueError("truncated raw tensor")
        return value["dtype"],np.frombuffer(payload,dtype=dtype).reshape(value["shape"]).copy()


def child(args):
    requests=[]
    for index,(case,steps) in enumerate(POINTS):
        request=dict(model="qwen-image-2.1",operation="image."+case,prompt=PROMPTS[case],
            width=512,height=512,steps=steps,frames=1,audio=False,seed=73,execution="gpu",residency="resident",
            allow_approximation=True,qwen21_dit_cache="off",lora_strategy="inference_time",
            loras=[dict(path=str(args.lora),strength=1.,role="transformer")],
            dump_tensors=str(args.output/f"{case}-{steps}"),output=str(args.output/f"{index}.png"))
        if case=="edit": request.update(qwen21_reference_size=512,
            inputs=[dict(kind="image",role="reference",path=str(args.reference))])
        requests.append(request)
    def before(index,request):
        os.environ[FLAG]=str(int(request["steps"]==5))
        os.environ["TURBOCIDER_QWEN21_LORA_BF16_AB"]="1"
        os.environ["TURBOCIDER_QWEN21_PREFIX_SNAPSHOT"]="0"
    generate_requests(args.library,"qwen-image-2.1",args.model,requests,args.output,
                      "sampler-raw-parity-only",before_request=before)


def analyze(output):
    report=json.loads((output/"report.json").read_text())
    if len(report["runs"])!=4: raise ValueError("four complete requests required")
    for i,(row,(case,steps)) in enumerate(zip(report["runs"],POINTS)):
        m=row["metrics"]; selection=m["acceleration_selection"]
        if ((m["operation"],m["steps"],m["actual_denoise_steps"],m["seed"],m["lora_applied_projections"],
             m["runtime_backend"],m["runtime_precision"],m["prompt_cache_hit"])!=
                ("image."+case,steps,steps,73,227,"mlx_cpp_metal","bf16",i%2==1) or
                "joint BF16 LoRA A/B operands" not in selection or (FIVE if steps==5 else SIX) not in selection or
                "prefix KV snapshot" in selection):
            raise ValueError("actual raw request differs")
    rows=[]
    for case in ("generate","edit"):
        checks=[]
        for name in ("initial","text")+(("reference_0",) if case=="edit" else ()):
            paths=[output/f"{case}-{step}"/f"qwen21_{name}.safetensors" for step in (6,5)]
            values=[tensor(path) for path in paths]
            if values[0][0]!=values[1][0] or not np.array_equal(values[0][1],values[1][1]):
                raise ValueError("raw conditioning/noise differs: "+case+"/"+name)
            if name=="initial" and (values[0][0]!="BF16" or values[0][1].shape!=(1,1024,64)):
                raise ValueError("actual BF16 image noise required")
            checks.append(dict(name=name,dtype=values[0][0],shape=list(values[0][1].shape),exact=True,
                               tensor_file_sha256=[sha256_file(path) for path in paths]))
        ends=[tensor(output/f"{case}-{step}"/"qwen21_latents.safetensors") for step in (6,5)]
        rows.append(dict(case=case,raw_checks=checks,denoised_latents_equal=bool(np.array_equal(ends[0][1],ends[1][1]))))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--lora",type=Path,default=ROOT/"models/Viggle-Qwen-Image-2.1-viggle-turbo/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors")
    p.add_argument("--reference",type=Path,default=ROOT/"results/qwen21/teapot-blue-edit-native-512.png")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--child",action="store_true")
    args=p.parse_args()
    for key in ("library","model","lora","reference"):
        setattr(args,key,getattr(args,key).resolve(strict=True))
    args.output=args.output.resolve()
    if args.child: child(args); return
    pin=adapter_pin(args.lora)
    if pin["rank"]!="r128" or sha256_file(args.lora)!=pin["sha256"]: p.error("pinned r128 required")
    args.output.mkdir(parents=True,exist_ok=False); trial=args.output/"trial"; trial.mkdir()
    assets={key:sha256_file(getattr(args,key)) for key in ("library","lora","reference")}
    source_paths=[Path(__file__),ROOT/"tools/validation/qwen_student_sampler_screen.py",
                  ROOT/"tools/validation/image_framework_native.py"]
    sources={str(path.relative_to(ROOT)):sha256_file(path) for path in source_paths}
    summary=dict(schema="tc-qwen-student-sampler-raw-v1",status="running",qualification_passed=False,
                 asset_sha256=assets,source_sha256=sources,performance_denominator=False)
    write_json(args.output/"source.json",{str(path.relative_to(ROOT)):path.read_text() for path in source_paths})
    write_json(args.output/"summary.json",summary)
    command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child","--library",str(args.library),
             "--model",str(args.model),"--lora",str(args.lora),"--reference",str(args.reference),"--output",str(trial)]
    observer=LoadObservation(args.output/"load.jsonl")
    try:
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            summary["memory"]=run_sampled(command,repo=ROOT,output=args.output,stem="trial",env=benchmark_environment(),
                stdout=stdout,stderr=stderr,timeout=600,interval_ms=100,max_gap_ms=500,observer=observer)
        summary["analysis"]=analyze(trial); summary["status"]="complete"
    except Exception as error: summary.update(status="failed",error=str(error))
    try: summary["load"]=observer.verify()
    except ValueError as error: summary["load_error"]=str(error)
    if any(sha256_file(getattr(args,key))!=digest for key,digest in assets.items()) or any(
            sha256_file(ROOT/path)!=digest for path,digest in sources.items()):
        summary.update(status="failed",error="source/asset changed during capture")
    write_json(args.output/"summary.json",summary); print(json.dumps(summary,indent=2))
    if summary["status"]!="complete": raise SystemExit(1)


if __name__=="__main__": main()
