#!/usr/bin/env python3
"""Actual original -> packed -> direct -> packed -> original lifetime screen."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from flux9_bf16_framework_screen import write_json
import qwen_corrected_qkv_request_screen as base
from runtime_ane_common import benchmark_environment,sha256_file
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ROOT=Path(__file__).resolve().parents[2]


def points(case="generate"):
    return [dict(case=case,on=storage!="off",storage=storage,seed=seed,bank_reused=False,conditioning_hit=i>0)
        for i,(storage,seed) in enumerate((("off",73),("packed",73),("direct",73),("direct",74),("direct",73),("packed",73),("off",73)))]


def analyze(output,report,case):
    sequence=points(case);base.validate(report,sequence);pixels=[];latents=[]
    for i in range(7):
        with Image.open(output/f"{i}.png") as im:
            if im.size!=(512,512):raise ValueError("complete switch PNG required")
            pixels.append(np.asarray(im.convert("RGBA")))
        value=base.tensor(output/f"raw-{i}/qwen21_latents.safetensors")
        if value[0]!="BF16" or value[1].shape!=(1,1024,64):raise ValueError("complete switch latents required")
        latents.append(value[1])
    for i in (1,2,4,5,6):
        if not np.array_equal(pixels[i],pixels[0]) or not np.array_equal(latents[i],latents[0]):raise ValueError("storage-switch output is not bitwise restored")
        for name in ("initial","text")+(("reference_0",) if case=="edit" else ()):
            a,b=(base.tensor(output/f"raw-{index}/qwen21_{name}.safetensors") for index in (0,i))
            if a[0]!=b[0] or not np.array_equal(a[1],b[1]):raise ValueError("switch raw noise/conditioning changed")
    if np.array_equal(pixels[3],pixels[0]) or np.array_equal(latents[3],latents[0]):raise ValueError("changed seed reused a previous output")
    active=[row["metrics"]["memory"]["mlx_active_bytes"] for row in report["runs"]]
    delta=[value-active[0] for value in active];expected=[0,3<<30,0,0,0,3<<30,0]
    if delta!=expected:raise ValueError("packed bank was not physically released from MLX active ownership before direct/optout")
    return dict(same_seed_latents_rgba_exact=True,changed_seed_recomputed=True,mlx_active_bytes=active,extra_active_bytes=delta,
        scope="seven complete requests; MLX active owners, not OS/driver release; switch/raw-dump timing is not performance evidence")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--library",type=Path,required=True)
    p.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--lora",type=Path,default=ROOT/"models/Viggle-Qwen-Image-2.1-viggle-turbo/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors")
    p.add_argument("--reference",type=Path,default=ROOT/"results/qwen21/teapot-blue-edit-native-512.png")
    p.add_argument("--case",choices=("generate","edit"),default="generate");p.add_argument("--output",type=Path,required=True)
    p.add_argument("--child",action="store_true");args=p.parse_args()
    for key in ("library","model","lora","reference"):setattr(args,key,getattr(args,key).resolve(strict=True))
    args.output=args.output.resolve()
    if args.child:base.run_points(args,points(args.case),dump=True);return
    pin=base.adapter_pin(args.lora)
    if pin["rank"]!="r128" or sha256_file(args.lora)!=pin["sha256"]:p.error("pinned original r128 required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    for item in manifest["inputs"]["sources"]:
        if sha256_file(ROOT/item["path"])!=item["sha256"]:p.error("current native source differs from linked library")
    args.output.mkdir(parents=True,exist_ok=False);trial=args.output/"trial";trial.mkdir()
    sources=[Path(__file__),Path(base.__file__),ROOT/"tools/validation/image_framework_native.py",ROOT/"tools/validation/qwen_student_prefix_screen.py"]
    hashes={str(path.relative_to(ROOT)):sha256_file(path) for path in sources};assets={key:sha256_file(getattr(args,key)) for key in ("library","lora","reference")}
    summary=dict(schema="tc-qwen-corrected-qkv-storage-switch-v1",status="running",case=args.case,qualification_passed=False,
        source_sha256=hashes,asset_sha256=assets,runtime_build_id=manifest["runtime_build_id"],frozen_sequence=points(args.case))
    write_json(args.output/"source.json",{str(path.relative_to(ROOT)):path.read_text() for path in sources});write_json(args.output/"summary.json",summary)
    command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child","--library",str(args.library),"--model",str(args.model),"--lora",str(args.lora),
        "--reference",str(args.reference),"--case",args.case,"--output",str(trial)];observer=LoadObservation(args.output/"load.jsonl")
    try:
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            summary["memory"]=run_sampled(command,repo=ROOT,output=args.output,stem="trial",env=benchmark_environment(),stdout=stdout,stderr=stderr,
                timeout=1800,interval_ms=100,max_gap_ms=500,observer=observer)
        report=json.loads((trial/"report.json").read_text())
        if report["loaded_runtime_id"]!=manifest["runtime_build_id"]:raise ValueError("actual switch runtime differs")
        summary["analysis"]=analyze(trial,report,args.case);summary["status"]="complete"
    except Exception as error:summary.update(status="failed",error=str(error))
    try:summary["load"]=observer.verify()
    except ValueError as error:summary["load_error"]=str(error)
    if any(sha256_file(path)!=hashes[str(path.relative_to(ROOT))] for path in sources) or any(sha256_file(getattr(args,key))!=digest for key,digest in assets.items()):
        summary.update(status="failed",error="bound switch source or artifact changed")
    write_json(args.output/"summary.json",summary);print(json.dumps(summary,indent=2))
    if summary["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
