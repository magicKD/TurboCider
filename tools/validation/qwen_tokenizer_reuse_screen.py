#!/usr/bin/env python3
"""Standalone verified CPU tokenizer cache: same full six-step r128 GPU work.

Fresh prompts throughout; no conditioning/prefix hit can masquerade as CPU
tokenizer reuse. Each process primes generation and reference state separately.
Control-before/reuse/control-after include every complete request and cold row.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys

from flux9_bf16_framework_screen import write_json
from image_framework_native import generate_requests
from qwen_student_prefix_screen import PROMPTS,adapter_pin
from qwen_student_sampler_screen import QUALITY_CASES
from runtime_ane_common import benchmark_environment,sha256_file
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ROOT=Path(__file__).resolve().parents[2]
FLAG="TURBOCIDER_QWEN21_TOKENIZER_REUSE"
PROCESSOR_SHA="aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"


def points():
    result=[dict(case_id=case_id,case=case,prompt=prompt,score=case_id not in ("photo","edit-color"))
            for case_id,case,prompt in QUALITY_CASES]
    result.append(dict(case_id="edit-table",case="edit",score=True,
        prompt="Change only the wooden table surface to white marble. Preserve the blue teapot, its shape, handle, spout, position and the background."))
    return result


def requests(args):
    values=[]
    for i,point in enumerate(points()):
        request=dict(model="qwen-image-2.1",operation="image."+point["case"],prompt=point["prompt"],
            width=512,height=512,steps=6,frames=1,audio=False,seed=73,execution="gpu",residency="resident",
            allow_approximation=True,qwen21_dit_cache="off",lora_strategy="inference_time",
            loras=[dict(path=str(args.lora),strength=1.,role="transformer")],output=str(args.output/f"{i}.png"))
        if point["case"]=="edit":request.update(qwen21_reference_size=512,
            inputs=[dict(kind="image",role="reference",path=str(args.reference))])
        values.append(request)
    return values


def finite_positive(value):
    return type(value) in (int,float) and math.isfinite(value) and value>0


def validate(report,reuse):
    if len(report["runs"])!=8:raise ValueError("complete eight-request fresh prompt cohort required")
    for i,(row,point) in enumerate(zip(report["runs"],points())):
        m=row["metrics"];selected=m["acceleration_selection"]
        if ((m["model"],m["operation"],m["width"],m["height"],m["steps"],m["actual_denoise_steps"],m["seed"])!=
                ("qwen-image-2.1","image."+point["case"],512,512,6,6,73) or
                not finite_positive(row["request_wall_including_export"])):
            raise ValueError("actual complete six-step task/wall differs")
        if (m["runtime_backend"]!="mlx_cpp_metal" or m["runtime_precision"]!="bf16" or
                m["encoder_runtime_backend"]!="mlx_cpp_metal" or m["encoder_runtime_precision"]!="bf16" or
                m["encoder_gpu_graph"]!="qwen3_vl_deepstack_gpu_only" or
                m.get("hybrid") or m.get("encoder_hybrid") or m.get("qwen_encoder_prefill") or
                m.get("student_ffn_reuse") or m.get("qwen21_dbcache") or
                m["lora_strategy"]!="inference_time" or m["lora_applied_projections"]!=227 or
                m["prompt_cache_hit"] is not False or "joint BF16 LoRA A/B operands" not in selected or
                "six-step student schedule" not in selected or "prefix KV snapshot" in selected):
            raise ValueError("fixed full GPU/joint arithmetic and fresh conditions required")
        weights=m["encoder_weight_residency"]
        if weights["enabled"] or weights["weights_retained"] or weights["weights_reused"] or weights["loads_session_total"]!=i+1:
            raise ValueError("encoder weight retention confounds standalone CPU cache")
        if ("reused reference encodings=1" in selected)!=(i>=5):
            raise ValueError("matched warm reference geometry differs")
        if not finite_positive(m["timings_seconds"]["text_encode"]):raise ValueError("complete encoder span required")
        cache=m.get("qwen_tokenizer_reuse")
        if (cache is not None)!=reuse:raise ValueError("actual tokenizer cache route differs")
        if reuse:
            if (cache["enabled"] is not True or cache["retained"] is not True or
                    cache["reused"] is not (i>0) or cache["constructed_this_request"] is not (i==0) or
                    cache["tokenized_this_request"] is not True or cache["processor_sha256"]!=PROCESSOR_SHA or
                    cache["processor_source_bytes"]!=11422654 or
                    not finite_positive(cache["validation_seconds_this_request"]) or
                    not finite_positive(cache["tokenization_seconds_this_request"]) or
                    (finite_positive(cache["construction_seconds_this_request"]) if i==0 else
                     cache["construction_seconds_this_request"]==0) is not True):
                raise ValueError("verified CPU owner lifecycle/source differs")


def child(args):
    os.environ[FLAG]="1" if args.child=="reuse" else "0"
    os.environ["TURBOCIDER_QWEN21_LORA_BF16_AB"]="1"
    os.environ["TURBOCIDER_QWEN21_PREFIX_SNAPSHOT"]="0"
    def before(index,request):print(json.dumps(dict(index=index,**points()[index])),flush=True)
    report=generate_requests(args.library,"qwen-image-2.1",args.model,requests(args),args.output,
                             "cpu-tokenizer-"+args.child,before_request=before)
    validate(report,args.child=="reuse");report["status"]="complete";write_json(args.output/"report.json",report)


def analyze(root):
    import numpy as np
    from PIL import Image
    names=("control-before","reuse","control-after")
    reports={name:json.loads((root/name/"report.json").read_text()) for name in names}
    for name in names:validate(reports[name],name=="reuse")
    rows=[]
    for i,point in enumerate(points()):
        reqs=[json.loads((root/name/f"request-{i}.json").read_text()) for name in names]
        normalize=lambda r:{k:v for k,v in r.items() if k!="output"}
        if len({json.dumps(normalize(r),sort_keys=True) for r in reqs})!=1:
            raise ValueError("request workload changed between CPU policies")
        images=[]
        for name in names:
            with Image.open(root/name/f"{i}.png") as im:
                im.load()
                if im.size!=(512,512):raise ValueError("complete image required")
                images.append(np.asarray(im.convert("RGBA")))
        if any(not np.array_equal(images[0],value) for value in images[1:]):
            raise ValueError("CPU-only reuse changed RGBA or return-to-control")
        spans={key:[reports[name]["runs"][i]["request_wall_including_export"] if key=="wall" else
                    reports[name]["runs"][i]["metrics"]["timings_seconds"]["text_encode"] for name in names]
               for key in ("wall","encoder")}
        control_wall=statistics.mean((spans["wall"][0],spans["wall"][2]))
        control_encoder=statistics.mean((spans["encoder"][0],spans["encoder"][2]))
        rows.append(dict(index=i,**point,rgba_exact=True,full_request_seconds=spans["wall"],encoder_seconds=spans["encoder"],
            matched_control_wall=control_wall,matched_control_encoder=control_encoder,
            wall_saved_seconds=control_wall-spans["wall"][1],encoder_saved_seconds=control_encoder-spans["encoder"][1],
            wall_reduction_percent=100*(1-spans["wall"][1]/control_wall)))
    groups=[]
    for case in ("generate","edit"):
        samples=[row for row in rows if row["case"]==case and row["score"]]
        saved=statistics.median(row["encoder_saved_seconds"] for row in samples)
        reduction=statistics.median(row["wall_reduction_percent"] for row in samples)
        groups.append(dict(case=case,scored_fresh_prompts=len(samples),median_encoder_saved_seconds=saved,
                           median_paired_wall_reduction_percent=reduction,
                           local_observation_gate_passed=saved>=.075 and reduction>=-1.))
    return dict(rows=rows,groups=groups,all_24_rgba_match=True,
        scope="fresh conditioning; both policies use warm reference-feature reuse after edit prime; no prefix/conditioning hit; CPU cache only")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--lora",type=Path,default=ROOT/"models/Viggle-Qwen-Image-2.1-viggle-turbo/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors")
    p.add_argument("--reference",type=Path,default=ROOT/"results/qwen21/teapot-blue-edit-native-512.png")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--child",choices=("off","reuse"))
    args=p.parse_args()
    for key in ("library","model","lora","reference"):setattr(args,key,getattr(args,key).resolve(strict=True))
    args.output=args.output.resolve()
    if args.child:child(args);return
    processor=args.model/"processor/tokenizer.json";pin=adapter_pin(args.lora)
    if sha256_file(processor)!=PROCESSOR_SHA or pin["rank"]!="r128" or sha256_file(args.lora)!=pin["sha256"]:
        p.error("fixed original BF16 processor and pinned r128 required")
    args.output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),ROOT/"native/models/qwen21/diagnostic_options.hpp",ROOT/"native/models/qwen21/encoder_prefill.hpp",
        ROOT/"native/models/qwen21/pipeline.cpp",ROOT/"native/models/qwen21/pipeline.hpp",ROOT/"native/models/qwen21_module.cpp",
        ROOT/"native/platform/apple/results.mm",ROOT/"native/runtime/session.hpp",ROOT/"tools/validation/image_framework_native.py"]
    hashes={str(path.relative_to(ROOT)):sha256_file(path) for path in sources}
    assets={key:sha256_file(getattr(args,key)) for key in ("library","lora","reference")};assets["processor"]=sha256_file(processor)
    summary=dict(schema="tc-qwen-tokenizer-reuse-screen-v1",status="running",qualification_passed=False,
        model_path=str(args.model),source_sha256=hashes,asset_sha256=assets,trials=[],
        gates=dict(minimum_median_fresh_encoder_saved_seconds=.075,maximum_median_full_request_regression_percent=1.,rgba_exact=True),
        scope="separate CPU processor reuse, original encoder/DiT/VAE and full six-step joint r128; not a GPU/ANE/kernel speedup")
    write_json(args.output/"source.json",{str(path.relative_to(ROOT)):path.read_text() for path in sources})
    write_json(args.output/"summary.json",summary)
    try:
        for name,mode in (("control-before","off"),("reuse","reuse"),("control-after","off")):
            output=args.output/name;output.mkdir();observer=LoadObservation(output/"load.jsonl")
            command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child",mode,"--library",str(args.library),
                     "--model",str(args.model),"--lora",str(args.lora),"--reference",str(args.reference),"--output",str(output)]
            with (output/"stdout.log").open("x") as stdout,(output/"stderr.log").open("x") as stderr:
                memory=run_sampled(command,repo=ROOT,output=output,stem="trial",env=benchmark_environment(),
                    stdout=stdout,stderr=stderr,timeout=1200,interval_ms=100,max_gap_ms=500,observer=observer)
            trial=dict(name=name,memory=memory)
            try:trial["load"]=observer.verify()
            except ValueError as error:trial["load_error"]=str(error)
            summary["trials"].append(trial);write_json(args.output/"summary.json",summary)
        summary["analysis"]=analyze(args.output);summary["status"]="complete"
    except Exception as error:summary.update(status="failed",error=str(error))
    if any(sha256_file(path)!=hashes[str(path.relative_to(ROOT))] for path in sources) or any(
            sha256_file(processor if key=="processor" else getattr(args,key))!=digest for key,digest in assets.items()):
        summary.update(status="failed",error="source/asset changed during observation")
    write_json(args.output/"summary.json",summary);print(json.dumps(summary,indent=2,ensure_ascii=False))
    if summary["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
