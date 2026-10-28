#!/usr/bin/env python3
"""Default-off corrected-QKV: complete six-step images, separate raw/timing windows."""
import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import statistics
import sys

from flux9_bf16_framework_screen import write_json
from image_framework_native import generate_requests
from qwen21_dit_cache_report import image_metrics
from qwen_student_prefix_screen import PROMPTS,adapter_pin
from qwen_student_sampler_raw import tensor
from qwen_student_sampler_screen import FIVE,SIX
from runtime_ane_common import benchmark_environment,sha256_file
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ROOT=Path(__file__).resolve().parents[2]
FLAG="TURBOCIDER_QWEN21_CORRECTED_QKV"
RECIPE="decode-only-corrected-qkv-v2-rope-rounding"
DIRECT_RECIPE="decode-only-corrected-qkv-v3-original-buffers"
DIRECT_FLAG="TURBOCIDER_QWEN21_CORRECTED_QKV_DIRECT"


def points(mode,case,arm,direct=False):
    if mode=="raw":
        result=[dict(case=task,on=on,seed=seed,bank_reused=reused,conditioning_hit=i!=0)
            for task in ("generate","edit")
            for i,(on,seed,reused) in enumerate(((False,73,False),(True,73,False),(True,74,True),(True,73,True),(False,73,False)))]
    elif mode=="timing" and case in ("generate","edit") and arm in ("off","on"):
        result=[dict(case=case,on=arm=="on",seed=73+i%2,bank_reused=arm=="on" and i>0,conditioning_hit=i>0) for i in range(6)]
    else:
        raise ValueError("known frozen full-request recipe required")
    if direct:
        for point in result:point.update(bank_reused=False,storage="direct" if point["on"] else "off")
    return result


def requested_recipe(args):return DIRECT_RECIPE if getattr(args,"direct",False) else RECIPE


def validate(report,sequence,expected_recipe=None):
    if len(report["runs"])!=len(sequence):raise ValueError("all complete requests required")
    for row,point in zip(report["runs"],sequence):
        steps=point.get("steps",6)
        if type(steps) is not int or steps not in (5,6):raise ValueError("explicit five/six sampler tier required")
        m=row["metrics"];wall=row["request_wall_including_export"]
        if (type(wall) not in (int,float) or not math.isfinite(wall) or wall<=0 or
                (m["model"],m["operation"],m["width"],m["height"],m["steps"],m["actual_denoise_steps"],m["seed"],m["lora_applied_projections"])!=
                ("qwen-image-2.1","image."+point["case"],512,512,steps,steps,point["seed"],227) or
                m["runtime_backend"]!="mlx_cpp_metal" or m["runtime_precision"]!="bf16" or
                m["encoder_runtime_backend"]!="mlx_cpp_metal" or m["lora_strategy"]!="inference_time" or
                m.get("hybrid") or m.get("encoder_hybrid") or m.get("student_ffn_reuse") or m.get("qwen21_dbcache") or
                m["prompt_cache_hit"] is not point["conditioning_hit"] or
                "joint BF16 LoRA A/B operands" not in m["acceleration_selection"] or "prefix KV snapshot" in m["acceleration_selection"]):
                raise ValueError("complete ordinary joint r128 six-step recipe differs")
        if (FIVE if steps==5 else SIX) not in m["acceleration_selection"]:raise ValueError("actual sampler tier receipt differs")
        if steps==5:
            labels=m.get("plan",{}).get("algorithm_approximations",[])
            if "qwen21_viggle_v021_r128_5step_sampler_experimental" not in labels or "qwen21_viggle_v021_r128_6step_distillation" in labels:
                raise ValueError("authored five-step quality tier must be explicit")
            if point["on"] and "qwen21_five_step_direct_qkv_composition_experimental" not in labels:
                raise ValueError("actual five/direct composition receipt required")
        q=m.get("qwen_corrected_qkv")
        if not point["on"]:
            if q is not None:raise ValueError("ordinary control claimed corrected-QKV work")
        else:
            required=DIRECT_RECIPE if point.get("storage")=="direct" else RECIPE if point.get("storage")=="packed" else expected_recipe
            direct=q is not None and q.get("recipe")==DIRECT_RECIPE
            if (not q or q.get("enabled") is not True or q.get("recipe") not in ("decode-only-corrected-qkv-v1",RECIPE,DIRECT_RECIPE) or
                (required is not None and q.get("recipe")!=required) or
                type(q.get("completed_decode_blocks")) is not int or q["completed_decode_blocks"]!=32*(steps-1) or
                (steps==5 and q.get("completed_sampler_steps")!=5) or
                type(q.get("bank_bytes")) is not int or q["bank_bytes"]!=(0 if direct else 3<<30) or q.get("bank_reused") is not point["bank_reused"] or
                (direct and (q.get("weight_storage")!="original-buffers" or point["bank_reused"])) or
                q.get("prefill_changed") is not False or q.get("online_lora") is not True or q.get("rank_math")!="f32" or
                type(q.get("pack_seconds_this_request")) not in (int,float) or not math.isfinite(q["pack_seconds_this_request"]) or
                q["pack_seconds_this_request"]<0 or (q["pack_seconds_this_request"]==0) is not (direct or point["bank_reused"])):
                raise ValueError("actual complete corrected-QKV/storage lifecycle required")


def run_points(args,sequence,*,dump=False):
    requests=[]
    for i,point in enumerate(sequence):
        r=dict(model="qwen-image-2.1",operation="image."+point["case"],prompt=point.get("prompt",PROMPTS[point["case"]]),width=512,height=512,
            steps=point.get("steps",6),frames=1,audio=False,seed=point["seed"],execution="gpu",residency="resident",allow_approximation=True,
            qwen21_dit_cache="off",lora_strategy="inference_time",loras=[dict(path=str(args.lora),role="transformer",strength=1.)],
            output=str(args.output/f"{i}.png"))
        if point["case"]=="edit":r.update(qwen21_reference_size=512,inputs=[dict(kind="image",role="reference",path=str(args.reference))])
        if dump:r["dump_tensors"]=str(args.output/f"raw-{i}")
        requests.append(r)
    def before(index,request):
        os.environ[FLAG]="1" if sequence[index]["on"] else "0"
        direct=sequence[index].get("storage")=="direct" if "storage" in sequence[index] else getattr(args,"direct",False)
        os.environ[DIRECT_FLAG]="1" if direct and sequence[index]["on"] else "0"
        os.environ["TURBOCIDER_QWEN21_LORA_BF16_AB"]="1";os.environ["TURBOCIDER_QWEN21_PREFIX_SNAPSHOT"]="0"
        os.environ["TURBOCIDER_QWEN21_STUDENT_FIVE_STEP"]="1" if sequence[index].get("steps",6)==5 else "0"
        print(json.dumps(dict(index=index,**sequence[index])),flush=True)
    report=generate_requests(args.library,"qwen-image-2.1",args.model,requests,args.output,"corrected-qkv-full-request",before_request=before)
    lib=C.CDLL(str(args.library));lib.tc_runtime_build_identity.restype=C.c_void_p;lib.tc_string_free.argtypes=[C.c_void_p]
    identity=lib.tc_runtime_build_identity()
    if not identity:raise ValueError("loaded library lacks runtime identity")
    try:report["loaded_runtime_id"]=C.string_at(identity).decode()
    finally:lib.tc_string_free(identity)
    mixed=any(point.get("storage")=="packed" for point in sequence)
    validate(report,sequence,None if mixed else requested_recipe(args));write_json(args.output/"report.json",report)
    return report


def child(args):
    run_points(args,points(args.mode,args.case,args.arm,args.direct),dump=args.mode=="raw")


def analyze(output,report,sequence,mode):
    import numpy as np
    from PIL import Image,ImageDraw
    validate(report,sequence);pixels=[]
    for i in range(len(sequence)):
        with Image.open(output/f"{i}.png") as im:
            im.load()
            if im.size!=(512,512):raise ValueError("complete 512 PNG required")
            pixels.append(np.asarray(im.convert("RGBA")))
    if mode=="timing":
        for i in range(2,6):
            if not np.array_equal(pixels[i],pixels[i%2]):raise ValueError("same-seed full image failed to restore")
        samples=[row["request_wall_including_export"] for row in report["runs"]][1:]
        return dict(full_request_seconds=samples,median_seconds=statistics.median(samples),
            cold_request_seconds=report["runs"][0]["request_wall_including_export"],same_seed_restored=True,
            scope="one process/arm; cold/setup separately retained; five conditioning-hot images including VAE/PNG; no pixel readback inside request wall")
    rows=[]
    for base in (0,5):
        point=sequence[base]
        if not np.array_equal(pixels[base],pixels[base+4]) or not np.array_equal(pixels[base+1],pixels[base+3]):
            raise ValueError("off/on/seed-change/on/off image restoration failed")
        if np.array_equal(pixels[base+1],pixels[base+2]):raise ValueError("changed seed reused an old corrected image")
        checks=[]
        for name in ("initial","text")+(("reference_0",) if point["case"]=="edit" else ()):
            values=[tensor(output/f"raw-{base+i}"/f"qwen21_{name}.safetensors") for i in (0,1,3,4)]
            if any(value[0]!=values[0][0] or not np.array_equal(value[1],values[0][1]) for value in values[1:]):
                raise ValueError("raw original noise/conditioning changed: "+name)
            checks.append(dict(name=name,exact=True,dtype=values[0][0],shape=list(values[0][1].shape)))
        a,b=(tensor(output/f"raw-{base+i}"/"qwen21_latents.safetensors") for i in (0,1))
        if a[0]!=b[0] or a[0]!="BF16" or a[1].shape!=(1,1024,64) or b[1].shape!=a[1].shape:raise ValueError("complete final BF16 image latents required")
        x=(a[1].astype(np.uint32)<<16).view(np.float32).astype(np.float64);y=(b[1].astype(np.uint32)<<16).view(np.float32).astype(np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError("nonfinite complete latents")
        l2=float(np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-30));cosine=float(np.sum(x*y)/max(np.linalg.norm(x)*np.linalg.norm(y),1e-30))
        rgb=image_metrics(pixels[base][:,:,:3],pixels[base+1][:,:,:3]);alpha=int(np.abs(pixels[base][:,:,3].astype(int)-pixels[base+1][:,:,3].astype(int)).max())
        passed=l2<=.01 and cosine>=.999 and rgb["ssim"]>=.99 and alpha==0
        rows.append(dict(case=point["case"],raw_checks=checks,latents_relative_l2=l2,latents_cosine=cosine,rgb=rgb,alpha_max_abs=alpha,
            local_quality_gate_passed=passed,off_on_off_restored=True,changed_seed_recomputed=True))
    sheet=Image.new("RGB",(1024,2*536),"white");draw=ImageDraw.Draw(sheet)
    for row,base in enumerate((0,5)):
        for column,i in enumerate((base,base+1)):
            canvas=Image.new("RGBA",(512,512),(220,220,220,255));canvas.alpha_composite(Image.fromarray(pixels[i],"RGBA"))
            sheet.paste(canvas.convert("RGB"),(column*512,row*536+24));draw.text((column*512+4,row*536+4),sequence[i]["case"]+(" corrected" if sequence[i]["on"] else " original"),fill="black")
    sheet.save(output/"comparison.png")
    return dict(rows=rows,local_quality_gate_passed=all(row["local_quality_gate_passed"] for row in rows),instruction_quality_qualified=False,
        scope="parity-only raw dump requests, never performance denominators; SSIM/L2 do not establish instruction adherence")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--library",type=Path,required=True)
    p.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--lora",type=Path,default=ROOT/"models/Viggle-Qwen-Image-2.1-viggle-turbo/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors")
    p.add_argument("--reference",type=Path,default=ROOT/"results/qwen21/teapot-blue-edit-native-512.png")
    p.add_argument("--output",type=Path,required=True);p.add_argument("--mode",choices=("raw","timing"),default="raw")
    p.add_argument("--case",choices=("generate","edit"),default="generate");p.add_argument("--arm",choices=("off","on"),default="off");p.add_argument("--child",action="store_true")
    p.add_argument("--direct",action="store_true")
    args=p.parse_args()
    for key in ("library","model","lora","reference"):setattr(args,key,getattr(args,key).resolve(strict=True))
    args.output=args.output.resolve()
    if args.child:child(args);return
    pin=adapter_pin(args.lora)
    if pin["rank"]!="r128" or sha256_file(args.lora)!=pin["sha256"]:p.error("pinned original r128 required")
    manifest_path=args.library.parent/"runtime-build/runtime-build-manifest.json";manifest=json.loads(manifest_path.read_text())
    def verify_sources():
        for item in manifest["inputs"]["sources"]:
            if sha256_file(ROOT/item["path"])!=item["sha256"]:raise ValueError("current native sources differ from loaded library")
    verify_sources();args.output.mkdir(parents=True,exist_ok=False);trial=args.output/"trial";trial.mkdir()
    sources=[Path(__file__),ROOT/"tools/validation/image_framework_native.py",ROOT/"native/models/qwen21/metal/lora_qkv_prepare.hpp",
        ROOT/"native/models/qwen21/diagnostic_options.hpp",ROOT/"native/models/qwen21/pipeline.cpp",ROOT/"native/models/qwen21/transformer.cpp"]
    hashes={str(path.relative_to(ROOT)):sha256_file(path) for path in sources};assets={key:sha256_file(getattr(args,key)) for key in ("library","lora","reference")}
    summary=dict(schema="tc-qwen-corrected-qkv-request-v1",status="running",mode=args.mode,case=args.case,arm=args.arm,
        qualification_passed=False,source_sha256=hashes,asset_sha256=assets,runtime_build_id=manifest["runtime_build_id"],
        manifest_sha256=sha256_file(manifest_path),frozen_sequence=points(args.mode,args.case,args.arm,args.direct),direct=args.direct,
        expected_recipe=requested_recipe(args),
        gates=dict(max_latents_relative_l2=.01,min_latents_cosine=.999,min_rgb_ssim=.99,alpha_exact=True),
        scope="original joint r128, all six steps and VAE/PNG; unchanged first step; no defaults/deployment or physical-engine claim")
    write_json(args.output/"source.json",{str(path.relative_to(ROOT)):path.read_text() for path in sources});write_json(args.output/"summary.json",summary)
    command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child","--library",str(args.library),"--model",str(args.model),
        "--lora",str(args.lora),"--reference",str(args.reference),"--output",str(trial),"--mode",args.mode,"--case",args.case,"--arm",args.arm]
    if args.direct:command.append("--direct")
    observer=LoadObservation(args.output/"load.jsonl")
    try:
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            summary["memory"]=run_sampled(command,repo=ROOT,output=args.output,stem="trial",env=benchmark_environment(),stdout=stdout,stderr=stderr,
                timeout=1800,interval_ms=100,max_gap_ms=500,observer=observer)
        report=json.loads((trial/"report.json").read_text())
        if report["loaded_runtime_id"]!=manifest["runtime_build_id"]:raise ValueError("actual loaded runtime differs")
        summary["analysis"]=analyze(trial,report,summary["frozen_sequence"],args.mode);summary["status"]="complete";verify_sources()
    except Exception as error:summary.update(status="failed",error=str(error))
    try:summary["load"]=observer.verify()
    except ValueError as error:summary["load_error"]=str(error)
    if any(sha256_file(path)!=hashes[str(path.relative_to(ROOT))] for path in sources) or any(sha256_file(getattr(args,key))!=digest for key,digest in assets.items()):
        summary.update(status="failed",error="source/artifact changed during complete window")
    write_json(args.output/"summary.json",summary);print(json.dumps(summary,indent=2))
    if summary["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
