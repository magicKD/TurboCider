#!/usr/bin/env python3
"""Seven prompt tasks/two seeds, original vs rounding-fixed complete requests."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from flux9_bf16_framework_screen import write_json
import qwen_corrected_qkv_request_screen as base
from qwen_student_sampler_screen import QUALITY_CASES
from runtime_ane_common import benchmark_environment,sha256_file
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ROOT=Path(__file__).resolve().parents[2]


def points(direct=False):
    result=[dict(case_id=case_id,case=case,prompt=prompt,on=on,seed=seed,bank_reused=reused,conditioning_hit=i>0)
        for case_id,case,prompt in QUALITY_CASES
        for i,(on,seed,reused) in enumerate(((False,73,False),(True,73,False),(True,74,True),(False,74,False)))]
    if direct:
        for point in result:point.update(bank_reused=False,storage="direct" if point["on"] else "off")
    return result


def analyze(output,report,direct=False):
    sequence=points(direct);base.validate(report,sequence,base.DIRECT_RECIPE if direct else base.RECIPE);rows=[]
    for start in range(0,len(sequence),4):
        for a,b in ((start,start+1),(start+3,start+2)):
            point=sequence[a];checks=[]
            for name in ("initial","text")+(("reference_0",) if point["case"]=="edit" else ()):
                values=[base.tensor(output/f"raw-{i}"/f"qwen21_{name}.safetensors") for i in (a,b)]
                if values[0][0]!=values[1][0] or not np.array_equal(values[0][1],values[1][1]):raise ValueError("matrix raw noise/conditioning differs")
                checks.append(dict(name=name,exact=True,dtype=values[0][0],shape=list(values[0][1].shape)))
            values=[base.tensor(output/f"raw-{i}/qwen21_latents.safetensors") for i in (a,b)]
            if any(value[0]!="BF16" or value[1].shape!=(1,1024,64) for value in values):raise ValueError("complete final matrix latents required")
            x,y=((value[1].astype(np.uint32)<<16).view(np.float32).astype(np.float64) for value in values)
            if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError("nonfinite matrix latents")
            l2=float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-30));cosine=float(np.sum(x*y)/max(np.linalg.norm(x)*np.linalg.norm(y),1e-30))
            pixels=[]
            for i in (a,b):
                with Image.open(output/f"{i}.png") as im:
                    if im.size!=(512,512):raise ValueError("complete matrix PNG required")
                    pixels.append(np.asarray(im.convert("RGBA")))
            rgba_exact=bool(np.array_equal(*pixels));rgb=base.image_metrics(pixels[0][:,:,:3],pixels[1][:,:,:3])
            alpha=int(np.abs(pixels[0][:,:,3].astype(int)-pixels[1][:,:,3].astype(int)).max())
            passed=l2<=.01 and cosine>=.999 and rgb["ssim"]>=.99 and alpha==0
            rows.append(dict(case_id=point["case_id"],case=point["case"],seed=point["seed"],original_index=a,corrected_index=b,raw_checks=checks,
                latents_relative_l2=l2,latents_cosine=cosine,latents_exact=bool(np.array_equal(values[0][1],values[1][1])),
                rgba_exact=rgba_exact,rgb=rgb,alpha_max_abs=alpha,local_quality_gate_passed=passed))
    return dict(rows=rows,local_quality_gate_passed=all(row["local_quality_gate_passed"] for row in rows),
        all_rgba_exact=all(row["rgba_exact"] for row in rows),all_latents_exact=all(row["latents_exact"] for row in rows),
        instruction_quality_qualified=False,scope="seven prompt tasks/two seeds; parity-only full six-step/VAE/PNG requests; not timing samples or semantic scoring")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--library",type=Path,required=True)
    p.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--lora",type=Path,default=ROOT/"models/Viggle-Qwen-Image-2.1-viggle-turbo/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors")
    p.add_argument("--reference",type=Path,default=ROOT/"results/qwen21/teapot-blue-edit-native-512.png")
    p.add_argument("--output",type=Path,required=True);p.add_argument("--child",action="store_true");p.add_argument("--direct",action="store_true");args=p.parse_args()
    for key in ("library","model","lora","reference"):setattr(args,key,getattr(args,key).resolve(strict=True))
    args.output=args.output.resolve()
    if args.child:base.run_points(args,points(args.direct),dump=True);return
    pin=base.adapter_pin(args.lora)
    if pin["rank"]!="r128" or sha256_file(args.lora)!=pin["sha256"]:p.error("pinned original r128 required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    for item in manifest["inputs"]["sources"]:
        if sha256_file(ROOT/item["path"])!=item["sha256"]:p.error("current native source differs from linked library")
    args.output.mkdir(parents=True,exist_ok=False);trial=args.output/"trial";trial.mkdir()
    sources=[Path(__file__),Path(base.__file__),ROOT/"tools/validation/qwen_student_sampler_screen.py",ROOT/"tools/validation/qwen_student_prefix_screen.py",
        ROOT/"tools/validation/qwen21_dit_cache_report.py",ROOT/"tools/validation/qwen_student_sampler_raw.py",ROOT/"tools/validation/image_framework_native.py"]
    hashes={str(path.relative_to(ROOT)):sha256_file(path) for path in sources};assets={key:sha256_file(getattr(args,key)) for key in ("library","lora","reference")}
    summary=dict(schema="tc-qwen-corrected-qkv-quality-matrix-v1",status="running",qualification_passed=False,source_sha256=hashes,asset_sha256=assets,
        runtime_build_id=manifest["runtime_build_id"],expected_recipe=base.requested_recipe(args),frozen_sequence=points(args.direct),direct=args.direct,
        gates=dict(max_latents_relative_l2=.01,min_latents_cosine=.999,min_rgb_ssim=.99,alpha_exact=True))
    write_json(args.output/"source.json",{str(path.relative_to(ROOT)):path.read_text() for path in sources});write_json(args.output/"summary.json",summary)
    command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child","--library",str(args.library),"--model",str(args.model),"--lora",str(args.lora),
        "--reference",str(args.reference),"--output",str(trial)];observer=LoadObservation(args.output/"load.jsonl")
    if args.direct:command.append("--direct")
    try:
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            summary["memory"]=run_sampled(command,repo=ROOT,output=args.output,stem="trial",env=benchmark_environment(),stdout=stdout,stderr=stderr,
                timeout=2400,interval_ms=100,max_gap_ms=500,observer=observer)
        report=json.loads((trial/"report.json").read_text())
        if report["loaded_runtime_id"]!=manifest["runtime_build_id"]:raise ValueError("actual matrix runtime differs")
        summary["analysis"]=analyze(trial,report,args.direct);summary["status"]="complete"
    except Exception as error:summary.update(status="failed",error=str(error))
    try:summary["load"]=observer.verify()
    except ValueError as error:summary["load_error"]=str(error)
    if any(sha256_file(path)!=hashes[str(path.relative_to(ROOT))] for path in sources) or any(sha256_file(getattr(args,key))!=digest for key,digest in assets.items()):
        summary.update(status="failed",error="bound matrix source or artifact changed")
    write_json(args.output/"summary.json",summary);print(json.dumps(summary,indent=2))
    if summary["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
