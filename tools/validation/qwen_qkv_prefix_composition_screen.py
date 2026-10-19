#!/usr/bin/env python3
"""Four actual edit arms: ordinary/direct QKV x prefix off/hit, both samplers.

Same-prefix kernel pairs require exact RGBA. Prefix-vs-prefill rounding is
reported separately, never silently called arithmetic or quality equivalence.
"""
import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import statistics
import sys

import numpy as np
from PIL import Image

from flux9_bf16_framework_screen import write_json
from image_framework_native import generate_requests
from qwen21_dit_cache_report import image_metrics
from qwen_student_prefix_screen import PROMPTS,adapter_pin
from qwen_student_sampler_screen import FIVE,SIX,QUALITY_CASES
from runtime_ane_common import benchmark_environment,sha256_file
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ROOT=Path(__file__).resolve().parents[2]
ARMS={"original":(False,False),"direct":(True,False),"prefix":(False,True),"combined":(True,True)}
EDITS=tuple(case for case in QUALITY_CASES if case[1]=="edit")
FLAG="TURBOCIDER_QWEN21_CORRECTED_QKV_PREFIX"


def point(steps,arm,seed,hit,*,prefix_hit=None,case_id="edit-color",prompt=None,case="edit"):
    return dict(steps=steps,arm=arm,seed=seed,conditioning_hit=hit,prefix_hit=prefix_hit,
                case_id=case_id,case=case,prompt=prompt or PROMPTS[case])


def points(mode,steps=5,arm="combined"):
    if mode=="timing":
        return [point(steps,arm,73+i%2,i>0,prefix_hit=i>0 if ARMS[arm][1] else None) for i in range(6)]
    if mode=="quality":
        order=(("original",73,None),("direct",73,None),("original",74,None),("direct",74,None),
               ("prefix",73,False),("prefix",73,True),("prefix",74,True),
               ("combined",73,False),("combined",73,True),("combined",74,True))
        return [point(tier,name,seed,i>0,prefix_hit=hit,case_id=cid,prompt=prompt)
                for tier in (6,5) for cid,_,prompt in EDITS for i,(name,seed,hit) in enumerate(order)]
    if mode!="restoration":raise ValueError("known frozen composition mode required")
    # One bank: sampler, arithmetic policy, prefix opt-out, conditioning and
    # operation changes must invalidate. Changed seed alone must not do so.
    sequence=[]
    for tier in (6,5,6):
        for i,seed in enumerate((73,73,74,73)):
            sequence.append(point(tier,"combined",seed,len(sequence)>0,prefix_hit=i>0))
    for name in ("prefix","combined","direct","combined"):
        for i in range(2 if ARMS[name][1] else 1):
            sequence.append(point(6,name,73,True,prefix_hit=i>0 if ARMS[name][1] else None))
    for prompt in (EDITS[1][2],PROMPTS["edit"]):
        for i in range(2):sequence.append(point(6,"combined",73,i>0,prefix_hit=i>0,prompt=prompt))
    for name in ("original","direct"):
        sequence.append(point(6,name,73,name=="direct",case="generate",case_id="photo"))
    # Text-only and edit conditioning have separate entries. Generation
    # clears the K/V snapshot, but does not evict the immutable edit inputs.
    for i in range(2):sequence.append(point(6,"combined",73,True,prefix_hit=i>0))
    return sequence


def validate(report,sequence):
    if len(report["runs"])!=len(sequence):raise ValueError("all complete requests required")
    for index,(row,p) in enumerate(zip(report["runs"],sequence)):
        m=row["metrics"];direct,prefix=ARMS[p["arm"]];selection=m["acceleration_selection"]
        if ((m["model"],m["operation"],m["width"],m["height"],m["steps"],m["actual_denoise_steps"],m["seed"],m["lora_applied_projections"])!=
                ("qwen-image-2.1","image."+p["case"],512,512,p["steps"],p["steps"],p["seed"],227) or
                m["runtime_backend"]!="mlx_cpp_metal" or m["runtime_precision"]!="bf16" or m["encoder_runtime_backend"]!="mlx_cpp_metal" or
                m["lora_strategy"]!="inference_time" or m["prompt_cache_hit"] is not p["conditioning_hit"] or
                "joint BF16 LoRA A/B operands" not in selection or (FIVE if p["steps"]==5 else SIX) not in selection or
                any(m.get(key) for key in ("hybrid","encoder_hybrid","student_ffn_reuse","qwen21_dbcache"))):
            raise ValueError(f"actual bounded joint GPU sampler recipe differs at request {index}")
        wall=row["request_wall_including_export"]
        if type(wall) not in (int,float) or not math.isfinite(wall) or wall<=0:raise ValueError("complete finite request wall required")
        if (("edit prefix KV snapshot hit" in selection)!=(p["prefix_hit"] is True) or
                ("edit prefix KV snapshot miss" in selection)!=(p["prefix_hit"] is False)):
            raise ValueError("actual prefix lifecycle differs")
        q=m.get("qwen_corrected_qkv")
        if not direct:
            if q is not None:raise ValueError("ordinary control claimed fusion")
            continue
        expected_blocks=32*(p["steps"]-(0 if p["prefix_hit"] is True else 1))
        if (not q or q.get("enabled") is not True or q.get("recipe")!="decode-only-corrected-qkv-v3-original-buffers" or
                q.get("weight_storage")!="original-buffers" or type(q.get("bank_bytes")) is not int or q["bank_bytes"]!=0 or q.get("bank_reused") is not False or q.get("pack_seconds_this_request")!=0 or
                type(q.get("completed_decode_blocks")) is not int or q["completed_decode_blocks"]!=expected_blocks or
                type(q.get("completed_sampler_steps")) is not int or q["completed_sampler_steps"]!=p["steps"] or
                type(q.get("completed_prefill_steps")) is not int or q["completed_prefill_steps"]!=(0 if p["prefix_hit"] is True else 1) or
                q.get("prefix_snapshot_enabled") is not prefix or q.get("prefix_snapshot_hit") is not (p["prefix_hit"] is True) or
                type(q.get("prefix_snapshot_bytes")) is not int or q["prefix_snapshot_bytes"]!=((m["text_tokens"]+m["reference_tokens"])*2*32*32*128*2 if prefix else 0) or
                q.get("rank_math")!="f32" or q.get("online_lora") is not True or q.get("prefill_changed") is not False):
            raise ValueError("complete actual direct/prefix receipt required")
        if prefix and "qwen21_direct_qkv_immutable_prefix_composition_experimental" not in m["plan"]["algorithm_approximations"]:
            raise ValueError("independent composition label missing")


def images(output,count):
    result=[]
    for i in range(count):
        with Image.open(output/f"{i}.png") as im:
            im.load()
            if im.size!=(512,512):raise ValueError("complete 512 PNG required")
            result.append(np.asarray(im.convert("RGBA")))
    return result


def analyze(output,report,sequence,mode):
    validate(report,sequence);pixels=images(output,len(sequence))
    if mode=="timing":
        for i in range(3,6):
            if not np.array_equal(pixels[i],pixels[1+(i-1)%2]):raise ValueError("hot same-arm seed restoration changed")
        samples=[row["request_wall_including_export"] for row in report["runs"]][1:]
        return dict(full_request_seconds=samples,median_seconds=statistics.median(samples),cold_request_seconds=report["runs"][0]["request_wall_including_export"],
                    same_hot_seed_restored=True,scope="five conditioning-hot full requests including VAE/PNG; prefix arms all actual hits; prime/cold separate")
    if mode=="restoration":
        seen={};restored=0
        for p,pixel in zip(sequence,pixels):
            key=(p["steps"],p["arm"],p["seed"],p["prompt"],p["prefix_hit"],p["case"])
            if key in seen:
                if not np.array_equal(pixel,seen[key]):raise ValueError("sampler/policy/operation restoration changed PNG")
                restored+=1
            seen[key]=pixel
        if np.array_equal(pixels[1],pixels[2]) or np.array_equal(pixels[5],pixels[6]):raise ValueError("changed seed reused old target")
        gen=[i for i,p in enumerate(sequence) if p["case"]=="generate"]
        if not np.array_equal(pixels[gen[0]],pixels[gen[1]]):raise ValueError("generation kernel regression")
        return dict(repeated_state_exact_pairs=restored,changed_seed_recomputed=True,generation_pair_exact=True,
                    scope="PNG/lifecycle restoration only, not full-image cached latent proof or a performance denominator")
    pairs=[];prefix_differences=[]
    for start in range(0,len(sequence),10):
        for a,b in ((0,4),(1,7)):
            if not np.array_equal(pixels[start+a],pixels[start+b]):raise ValueError("prefix prime changed its matched full-prefill image")
        for a,b in ((0,1),(2,3),(5,8),(6,9)):
            a+=start;b+=start
            if not np.array_equal(pixels[a],pixels[b]):raise ValueError("same-prefix same-sampler kernel changed complete RGBA")
            pairs.append(dict(steps=sequence[a]["steps"],case_id=sequence[a]["case_id"],seed=sequence[a]["seed"],prefix=sequence[a]["prefix_hit"] is True,
                              original_index=a,direct_index=b,rgba_exact=True))
        for a,b in ((0,5),(2,6)):
            a+=start;b+=start;x,y=pixels[a],pixels[b]
            rgb=image_metrics(x[:,:,:3],y[:,:,:3]);alpha=int(np.abs(x[:,:,3].astype(int)-y[:,:,3].astype(int)).max())
            prefix_differences.append(dict(steps=sequence[a]["steps"],case_id=sequence[a]["case_id"],seed=sequence[a]["seed"],rgb=rgb,alpha_max_abs=alpha,
                rgba_exact=bool(np.array_equal(x,y)),local_pixel_gate_passed=rgb["ssim"]>=.99 and alpha==0))
    return dict(kernel_pairs=pairs,all_same_prefix_kernel_rgba_exact=True,prime_full_prefill_rgba_exact=True,prefix_differences=prefix_differences,
                prefix_pixel_gate_passed=all(row["local_pixel_gate_passed"] for row in prefix_differences),instruction_quality_qualified=False,
                scope="same-prefix exact fusion gate; full-prefill vs prefix-hit rounding separate; no semantic or cross-NFE equivalence claim")


def child(args,sequence):
    requests=[]
    for i,p in enumerate(sequence):
        r=dict(model="qwen-image-2.1",operation="image."+p["case"],prompt=p["prompt"],width=512,height=512,steps=p["steps"],frames=1,audio=False,
               seed=p["seed"],execution="gpu",residency="resident",allow_approximation=True,qwen21_dit_cache="off",lora_strategy="inference_time",
               loras=[dict(path=str(args.lora),strength=1.,role="transformer")],output=str(args.output/f"{i}.png"))
        if p["case"]=="edit":r.update(qwen21_reference_size=512,inputs=[dict(kind="image",role="reference",path=str(args.reference))])
        requests.append(r)
    def before(index,request):
        p=sequence[index];direct,prefix=ARMS[p["arm"]]
        for key,value in {"TURBOCIDER_QWEN21_CORRECTED_QKV":direct,"TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT":direct,
                          FLAG:direct and prefix,"TURBOCIDER_QWEN21_PREFIX_SNAPSHOT":prefix,"TURBOCIDER_QWEN21_STUDENT_FIVE_STEP":p["steps"]==5,
                          "TURBOCIDER_QWEN21_LORA_BF16_AB":True}.items():os.environ[key]=str(int(value))
        print(json.dumps(dict(index=index,**p)),flush=True)
    report=generate_requests(args.library,"qwen-image-2.1",args.model,requests,args.output,"direct-qkv-prefix-composition",before_request=before)
    lib=C.CDLL(str(args.library));lib.tc_runtime_build_identity.restype=C.c_void_p;lib.tc_string_free.argtypes=[C.c_void_p]
    identity=lib.tc_runtime_build_identity()
    if not identity:raise ValueError("loaded runtime identity missing")
    try:report["loaded_runtime_id"]=C.string_at(identity).decode()
    finally:lib.tc_string_free(identity)
    write_json(args.output/"report.json",report)
    validate(report,sequence);write_json(args.output/"report.json",report)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--library",type=Path,required=True)
    p.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--lora",type=Path,default=ROOT/"models/Viggle-Qwen-Image-2.1-viggle-turbo/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors")
    p.add_argument("--reference",type=Path,default=ROOT/"results/qwen21/teapot-blue-edit-native-512.png");p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mode",choices=("restoration","quality","timing"),default="restoration");p.add_argument("--steps",type=int,choices=(5,6),default=5)
    p.add_argument("--arm",choices=tuple(ARMS),default="combined");p.add_argument("--child",action="store_true");args=p.parse_args()
    for key in ("library","model","lora","reference"):setattr(args,key,getattr(args,key).resolve(strict=True))
    args.output=args.output.resolve();sequence=points(args.mode,args.steps,args.arm)
    if args.child:child(args,sequence);return
    pin=adapter_pin(args.lora)
    if pin["rank"]!="r128" or sha256_file(args.lora)!=pin["sha256"]:p.error("original pinned r128 required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    for item in manifest["inputs"]["sources"]:
        if sha256_file(ROOT/item["path"])!=item["sha256"]:p.error("native source differs from linked library")
    args.output.mkdir(parents=True,exist_ok=False);trial=args.output/"trial";trial.mkdir()
    paths=[Path(__file__),ROOT/"tools/validation/image_framework_native.py",ROOT/"tools/validation/qwen_student_sampler_screen.py",
           ROOT/"tools/validation/qwen_student_prefix_screen.py",ROOT/"tools/validation/qwen21_dit_cache_report.py"]
    hashes={str(path.relative_to(ROOT)):sha256_file(path) for path in paths};assets={key:sha256_file(getattr(args,key)) for key in ("library","lora","reference")}
    summary=dict(schema="tc-qwen-direct-qkv-prefix-composition-v1",status="running",qualification_passed=False,mode=args.mode,steps=args.steps,arm=args.arm,
                 frozen_sequence=sequence,runtime_build_id=manifest["runtime_build_id"],source_sha256=hashes,asset_sha256=assets)
    write_json(args.output/"source.json",{str(path.relative_to(ROOT)):path.read_text() for path in paths});write_json(args.output/"summary.json",summary)
    command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child","--library",str(args.library),"--model",str(args.model),"--lora",str(args.lora),
             "--reference",str(args.reference),"--output",str(trial),"--mode",args.mode,"--steps",str(args.steps),"--arm",args.arm]
    observer=LoadObservation(args.output/"load.jsonl")
    try:
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            summary["memory"]=run_sampled(command,repo=ROOT,output=args.output,stem="trial",env=benchmark_environment(),stdout=stdout,stderr=stderr,
                timeout=2400,interval_ms=100,max_gap_ms=500,observer=observer)
        report=json.loads((trial/"report.json").read_text())
        if report["loaded_runtime_id"]!=manifest["runtime_build_id"]:raise ValueError("loaded runtime differs")
        summary["analysis"]=analyze(trial,report,sequence,args.mode)
        if args.mode=="quality" and not summary["analysis"]["prefix_pixel_gate_passed"]:raise ValueError("prefill/prefix pixel gate failed; no gate relaxation")
        summary["status"]="complete"
    except Exception as error:summary.update(status="failed",error=str(error))
    try:summary["load"]=observer.verify()
    except ValueError as error:summary["load_error"]=str(error)
    if any(sha256_file(path)!=hashes[str(path.relative_to(ROOT))] for path in paths) or any(sha256_file(getattr(args,key))!=digest for key,digest in assets.items()):
        summary.update(status="failed",error="bound source/asset changed")
    write_json(args.output/"summary.json",summary);print(json.dumps(summary,indent=2))
    if summary["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
