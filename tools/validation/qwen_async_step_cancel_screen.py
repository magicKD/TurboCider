#!/usr/bin/env python3
"""Real model cancel at one/two submitted states, retry and sync restoration."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from flux9_bf16_framework_screen import write_json
from qwen_student_prefix_screen import adapter_pin
from runtime_ane_common import benchmark_environment,sha256_file
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ROOT=Path(__file__).resolve().parents[2]
SOURCES=("tools/native/qwen21_async_step_cancel_probe.mm","tools/native/build_qwen21_async_step_cancel_probe.sh",
         "tools/validation/qwen_async_step_cancel_screen.py")


def analyze(root,native):
    if (native.get("status")!="complete" or native.get("successful_full_requests")!=8 or native.get("cancelled_requests")!=2 or
            len(native.get("cases",[]))!=2):raise ValueError("complete same-engine model cancellation/retry windows required")
    comparisons=0
    for case,target,labels in (("generate",1,["sync","retry","restore"]),("edit",2,["sync","sync-hit","retry","retry-hit","restore"])):
        c=next(c for c in native["cases"] if c["case"]==case)
        if (c["cancel_after_submit"]!=target or c["retained_steps_at_cancel"]!=target or c["completed_progress_before_cancel"]!=0 or
                c["cancelled_png_absent"] is not True or c["cancelled"] is not True or (root/f"{case}-must-not-export.png").exists()):
            raise ValueError("cancelled request published progress or output before checked drain")
        if [entry["label"] for entry in c["successful"]]!=labels:raise ValueError("complete retry and original-policy restoration required")
        reference=None
        for entry in c["successful"]:
            with Image.open(root/entry["filename"]) as image:
                image.load()
                if image.size!=(512,512):raise ValueError("complete retry PNG required")
                pixels=np.asarray(image.convert("RGBA"))
            if reference is None:reference=pixels
            elif not np.array_equal(reference,pixels):raise ValueError("cancel/retry/restore changed complete RGBA")
            else:comparisons+=1
    return dict(successful_full_requests=8,cancelled_requests=2,rgba_exact_comparisons=comparisons,
                no_unchecked_completed_progress=True,no_cancelled_png=True,retry_and_sync_restoration_exact=True,
                scope="actual submit-hook one/two-pending cancellation and reuse; no timing, arbitrary device-loss/timeout or physical-overlap qualification")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--binary",type=Path,required=True);p.add_argument("--library",type=Path,required=True)
    p.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--lora",type=Path,default=ROOT/"models/Viggle-Qwen-Image-2.1-viggle-turbo/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors")
    p.add_argument("--reference",type=Path,default=ROOT/"results/qwen21/teapot-blue-edit-native-512.png");p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    for name in ("binary","library","model","lora","reference"):setattr(args,name,getattr(args,name).resolve(strict=True))
    args.output=args.output.resolve();pin=adapter_pin(args.lora)
    if pin["rank"]!="r128" or sha256_file(args.lora)!=pin["sha256"]:p.error("original released r128 required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    for item in manifest["inputs"]["sources"]:
        if sha256_file(ROOT/item["path"])!=item["sha256"]:p.error("native sources differ from linked library")
    hashes={name:sha256_file(ROOT/name) for name in SOURCES};assets={name:sha256_file(getattr(args,name)) for name in ("binary","library","lora","reference")}
    descriptor=json.loads(subprocess.check_output([str(args.binary),"--describe"],text=True,cwd=ROOT,env=benchmark_environment()))
    if descriptor.get("recipe")!="bounded-async-cancel-v1" or descriptor.get("cpp_sha256")!=hashes[SOURCES[0]] or descriptor.get("loaded_runtime_id")!=manifest["runtime_build_id"]:
        p.error("actual cancel probe/source/runtime descriptor differs")
    args.output.mkdir(parents=True,exist_ok=False);trial=args.output/"trial"
    s=dict(schema="tc-qwen-async-cancel-screen-v1",status="running",qualification_passed=False,descriptor=descriptor,source_sha256=hashes,asset_sha256=assets)
    write_json(args.output/"source.json",{name:(ROOT/name).read_text() for name in SOURCES});write_json(args.output/"summary.json",s)
    observer=LoadObservation(args.output/"load.jsonl")
    try:
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            s["memory"]=run_sampled([str(args.binary),str(args.model),str(args.lora),str(args.reference),str(trial)],repo=ROOT,output=args.output,stem="trial",env=benchmark_environment(),
                stdout=stdout,stderr=stderr,timeout=1200,interval_ms=100,max_gap_ms=500,observer=observer)
        native=json.loads((trial/"native.json").read_text())
        if native["loaded_runtime_id"]!=descriptor["loaded_runtime_id"]:raise ValueError("loaded runtime differs")
        s["analysis"]=analyze(trial,native);s["status"]="complete"
    except Exception as error:s.update(status="failed",error=str(error))
    try:s["load"]=observer.verify()
    except ValueError as error:s["load_error"]=str(error)
    if any(sha256_file(ROOT/name)!=digest for name,digest in hashes.items()) or any(sha256_file(getattr(args,name))!=digest for name,digest in assets.items()):
        s.update(status="failed",error="bound cancel source/asset changed")
    write_json(args.output/"summary.json",s);print(json.dumps(s,indent=2))
    if s["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
