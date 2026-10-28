#!/usr/bin/env python3
"""Serial full-GPU/joint base512 quality cases; no concurrent model windows."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import collect
from collect import ROOT, model_snapshot, sha256_file, write_json

TASKS=(
    ("poster","generate",'A flat minimalist poster on a plain white background. Exactly two lines of large, sharp black text: "QWEN IMAGE" and "512 TEST". A small solid blue square beneath the text. Straight-on view, no other text.'),
    ("portrait","generate","Studio portrait photograph of an adult East Asian woman with short black hair, round glasses and a dark green cotton jacket, neutral grey background, natural skin texture, soft lighting, centered composition."),
    ("materials","generate","A product photograph of exactly three objects in one horizontal row on a neutral grey tabletop: a matte red cube on the left, a polished steel sphere in the center, and a transparent clear glass bottle on the right. Soft neutral studio light, realistic materials, no text."),
    ("matte-edit","edit","Change only the teapot color to matte forest green. Preserve its original teapot geometry, ceramic texture, table and lighting. Do not add gloss, objects or text."),
)


def execution_options(arm,channels,activation):
    if arm not in ("gpu","joint") or channels not in (5120,6144) or activation not in collect.FFN_ACTIVATIONS:
        raise ValueError("explicit ordinary-GPU/joint quality recipe required")
    if arm=="gpu":return []
    flags=["--channels",str(channels),"--private-qkv"]
    if activation!="exp":flags.extend(["--fixed-program","--ffn-activation",activation])
    return flags


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--channels",type=int,choices=(5120,6144),default=6144)
    parser.add_argument("--ffn-activation",choices=collect.FFN_ACTIVATIONS,default="exp")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--describe",action="store_true")
    args=parser.parse_args()
    if args.describe:
        print(json.dumps(TASKS,indent=2));return
    args.library=args.library.resolve(strict=True);args.output=args.output.resolve()
    source_sha=sha256_file(Path(__file__))
    library_sha=sha256_file(args.library)
    model=ROOT/"models/Comfy-Org-Qwen-Image-2.1"
    identity=model_snapshot(model)
    args.output.mkdir(parents=True,exist_ok=False)
    receipt=dict(schema="tc-qwen-base-joint-quality-collection-v1",status="running",channels=args.channels,ffn_activation=args.ffn_activation,
        source_sha256=source_sha,library_sha256=library_sha,model_snapshot=identity,tasks=TASKS,completed=[])
    write_json(args.output/"recipe.json",receipt)
    try:
        for label,case,prompt in TASKS:
            for arm,route in (("gpu","gpu"),("joint","runtime")):
                destination=args.output/f"{label}-{arm}"
                command=[sys.executable,"-B",str(ROOT/"tools/profiling/qwen_base/collect.py"),
                    "--library",str(args.library),"--route",route,"--case",case,"--mode","stock","--requests","1",
                    "--prompt",prompt,"--output",str(destination)]
                command.extend(execution_options(arm,args.channels,args.ffn_activation))
                with (args.output/f"{label}-{arm}-driver.log").open("x") as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                receipt["completed"].append(dict(label=label,arm=arm,output=destination.name))
                write_json(args.output/"recipe.json",receipt)
        if (sha256_file(Path(__file__))!=source_sha or sha256_file(args.library)!=library_sha or
                model_snapshot(model)!=identity):raise ValueError("bound quality recipe/library/model changed")
        receipt["status"]="complete"
    except Exception as error:
        receipt.update(status="failed",error=str(error));raise
    finally:write_json(args.output/"recipe.json",receipt)


if __name__=="__main__":main()
