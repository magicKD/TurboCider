#!/usr/bin/env python3
"""CPU-only complete request/source/raw/PNG/performance denominator recheck."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image

from flux9_bf16_framework_screen import write_json
import qwen_lora_swiglu_request_screen as screen
from runtime_ane_common import sha256_file

ROOT=Path(__file__).resolve().parents[2]


def request(path):
    value=json.loads(path.read_text())
    for name in ("output","dump_tensors"):value.pop(name,None)
    return value


def window(root,library_sha,runtime):
    s=json.loads((root/"summary.json").read_text());r=json.loads((root/"trial/report.json").read_text())
    if s["status"]!="complete" or s["asset_sha256"]["library"]!=library_sha or r["loaded_runtime_id"]!=runtime:raise ValueError("complete same-library request window required")
    if json.loads((root/"trial/loaded-runtime.json").read_text())["runtime_build_id"]!=runtime:raise ValueError("pre-request loaded runtime differs")
    sources=json.loads((root/"source.json").read_text())
    for name,digest in s["source_sha256"].items():
        if hashlib.sha256(sources[name].encode()).hexdigest()!=digest or sha256_file(ROOT/name)!=digest:raise ValueError("bound full-request source differs")
    seq=screen.points(s["mode"],s["case"],s["steps"],s["ffn"]=="on",s["prefix"])
    if seq!=s["frozen_sequence"]:raise ValueError("frozen full workload differs")
    a=screen.analyze(root/"trial",r,seq,s["mode"])
    if a!=s["analysis"]:raise ValueError("full-array/pixel/denominator recheck differs")
    for kind in ("evidence","report"):
        if sha256_file(root/s["memory"][kind])!=s["memory"][kind+"_sha256"]:raise ValueError("process memory evidence changed")
    memory=s["memory"]["verified"]
    if not memory["complete"] or memory["command_exit_code"]!=0:raise ValueError("complete process window required")
    return s,r,seq,screen.prefix_screen.images(root/"trial",len(seq)),[request(root/f"trial/request-{i}.json") for i in range(len(seq))]


def finite_raw(value):
    dtype,data=value
    if dtype=="BF16":
        if np.any((data&0x7f80)==0x7f80):raise ValueError("nonfinite complete BF16 tensor")
    elif dtype!="F32" or not np.isfinite(data).all():raise ValueError("unsupported/nonfinite raw tensor")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--library",type=Path,required=True);p.add_argument("--native-source-archive",type=Path,required=True)
    p.add_argument("--raw",type=Path,required=True);p.add_argument("--restoration",type=Path,required=True);p.add_argument("--quality",type=Path,required=True)
    p.add_argument("--cohort",type=Path,action="append",required=True);p.add_argument("--repeat-cohort",type=Path,action="append",required=True)
    p.add_argument("--previous-no-prefix-quality",type=Path,required=True)
    p.add_argument("--previous-prefix-quality",type=Path,required=True);p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error("fresh audit output required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text());inputs=manifest["inputs"]["sources"]
    with tarfile.open(args.native_source_archive,"r:gz") as archive:
        members={m.name:m for m in archive.getmembers()}
        if set(members)!={item["path"] for item in inputs}:raise ValueError("complete native/build archive required")
        for item in inputs:
            f=archive.extractfile(members[item["path"]])
            if f is None or hashlib.sha256(f.read()).hexdigest()!=item["sha256"] or sha256_file(ROOT/item["path"])!=item["sha256"]:
                raise ValueError("archive/current native source differs from linked build")
    library_sha=sha256_file(args.library);runtime=manifest["runtime_build_id"]
    raw=window(args.raw,library_sha,runtime);restoration=window(args.restoration,library_sha,runtime);quality=window(args.quality,library_sha,runtime)
    if (raw[0]["mode"],restoration[0]["mode"],quality[0]["mode"])!=("raw","restoration","quality"):raise ValueError("separate full raw/restoration/quality windows required")
    lookup={screen.signature(point,with_ffn=True):i for i,point in enumerate(quality[2])}
    histories=[]
    for root,prefix in ((args.previous_no_prefix_quality,False),(args.previous_prefix_quality,True)):
        s=json.loads((root/"summary.json").read_text())
        if s["status"]!="complete":raise ValueError("previous sealed complete quality controls required")
        table={}
        for i,point in enumerate(s["frozen_sequence"]):
            qkv,pf=(screen.prefix_screen.ARMS[point["arm"]] if prefix else (point["on"],False))
            if prefix and not pf:continue
            key=(point["case"],point["prompt"],point["steps"],qkv,point["seed"],point["prefix_hit"] if prefix else None)
            table[key]=i
        histories.append((root,s,table))
    previous_counts=[0,0];previous_raw=0;matched_matrix=0;skipped_alternate=0
    ownership=[]
    for pair in quality[0]["analysis"]["kernel_pairs"]:
        a,b=pair["off_index"],pair["on_index"]
        values=[quality[1]["runs"][i]["metrics"]["memory"]["mlx_active_bytes"] for i in (a,b)]
        ownership.append(dict(case_id=pair["case_id"],steps=pair["steps"],qkv=pair["qkv"],prefix=pair["prefix"],prefix_hit=pair["prefix_hit"],
                              off_active_bytes=values[0],on_active_bytes=values[1],active_delta=values[1]-values[0]))
    def recheck(w,root):
        nonlocal previous_raw,matched_matrix,skipped_alternate
        for i,point in enumerate(w[2]):
            if point["reference"]=="alternate":skipped_alternate+=1;continue
            j=lookup[screen.signature(point,with_ffn=True)]
            if w[4][i]!=quality[4][j] or not np.array_equal(w[3][i],quality[3][j]):raise ValueError("raw/restoration/timing request or RGBA differs from matrix")
            matched_matrix+=1
            h=1 if point["prefix"] else 0;old_root,old_s,table=histories[h]
            if any(w[0]["asset_sha256"][asset]!=old_s["asset_sha256"][asset] for asset in ("lora","reference")):raise ValueError("historical original adapter/reference differs")
            k=point["case"],point["prompt"],point["steps"],point["qkv"],point["seed"],point["prefix_hit"] if point["prefix"] else None
            old_index=table[k]
            with Image.open(old_root/f"trial/{old_index}.png") as image:old=np.asarray(image.convert("RGBA"))
            if w[4][i]!=request(old_root/f"trial/request-{old_index}.json") or not np.array_equal(w[3][i],old):raise ValueError("request or PNG regressed against historical same-policy baseline")
            previous_counts[h]+=1
            if w[0]["mode"]=="raw":
                for name in ("initial","text","latents")+(("reference_0",) if point["case"]=="edit" else ()):
                    a=screen.tensor(root/f"trial/raw-{i}/qwen21_{name}.safetensors");b=screen.tensor(old_root/f"trial/raw-{old_index}/qwen21_{name}.safetensors")
                    finite_raw(a);finite_raw(b)
                    if a[0]!=b[0] or not np.array_equal(a[1],b[1]):raise ValueError("complete raw input/final latent regressed across builds")
                previous_raw+=1
    windows=[(args.raw,raw),(args.restoration,restoration),(args.quality,quality)];cohorts={};repeats={};observations=[]
    for group,roots in (("initial",args.cohort),("repeat",args.repeat_cohort)):
        table=cohorts if group=="initial" else repeats
        for root in roots:
            w=window(root,library_sha,runtime);s=w[0];key=s["case"],s["steps"],s["prefix"],s["ffn"]
            if s["mode"]!="timing" or key in table:raise ValueError("one fixed process per task/tier/prefix/FFN arm in each declared batch required")
            table[key]=s["analysis"];windows.append((root,w))
            loads=[json.loads(line) for line in (root/"load.jsonl").read_text().splitlines()]
            observations.append(dict(cohort=root.name,group=group,case=key[0],steps=key[1],prefix=key[2],ffn=key[3],analysis=s["analysis"],memory=s["memory"]["verified"],
                load_error=s.get("load_error"),competing_load_samples=sum(bool(row["competing_processes"]) for row in loads),summary_sha256=sha256_file(root/"summary.json")))
    expected={(case,steps,prefix,ffn) for case in ("generate","edit") for steps in (5,6) for prefix in ((False,) if case=="generate" else (False,True)) for ffn in ("off","on")}
    if set(cohorts)!=expected:raise ValueError("all twelve no-dump strong-baseline timing windows required")
    if set(repeats)!={(case,5,prefix,ffn) for case,prefix in (("generate",False),("edit",True)) for ffn in ("off","on")}:
        raise ValueError("both five-step primary categories need independent reverse-arm repeat windows")
    for root,w in windows:recheck(w,root)
    timings=[]
    for case in ("generate","edit"):
        for steps in (6,5):
            for prefix in ((False,) if case=="generate" else (False,True)):
                off=cohorts[case,steps,prefix,"off"]["median_seconds"];on=cohorts[case,steps,prefix,"on"]["median_seconds"]
                timings.append(dict(case=case,steps=steps,prefix=prefix,direct_qkv_off_seconds=off,direct_qkv_fused_seconds=on,reduction_percent=100*(1-on/off)))
    repeat_timings=[]
    for case,prefix in (("generate",False),("edit",True)):
        off=repeats[case,5,prefix,"off"]["median_seconds"];on=repeats[case,5,prefix,"on"]["median_seconds"]
        repeat_timings.append(dict(case=case,steps=5,prefix=prefix,direct_qkv_off_seconds=off,direct_qkv_fused_seconds=on,reduction_percent=100*(1-on/off),
                                   scope="independent reverse-arm process pair; not best-of-two or a confidence interval"))
    rows=[row for _,w in windows for row in w[1]["runs"]];fused=[row["metrics"]["qwen_lora_swiglu"] for row in rows if row["metrics"].get("qwen_lora_swiglu")]
    result=dict(schema="tc-qwen-lora-swiglu-request-audit-v1",status="complete",qualification_passed=False,library_sha256=library_sha,runtime_build_id=runtime,
        native_source_hashes_verified=len(inputs),native_source_archive_sha256=sha256_file(args.native_source_archive),raw_analysis=raw[0]["analysis"],
        restoration_analysis=restoration[0]["analysis"],quality_analysis=quality[0]["analysis"],active_ownership=ownership,observations=observations,timings=timings,repeat_timings=repeat_timings,
        completed_full_requests=len(rows),actual_denoise_steps=sum(row["metrics"]["actual_denoise_steps"] for row in rows),fused_requests=len(fused),
        completed_fused_decode_blocks=sum(row["completed_decode_blocks"] for row in fused),matrix_exact_requests=matched_matrix,alternate_reference_restoration_requests=skipped_alternate,
        historical_no_prefix_exact_requests=previous_counts[0],historical_prefix_exact_requests=previous_counts[1],historical_raw_exact_requests=previous_raw,
        historical_summaries=[dict(path=str(root),sha256=sha256_file(root/"summary.json")) for root,_,_ in histories],
        raw_tensor_sha256=[dict(path=str(path),sha256=sha256_file(path)) for path in sorted((args.raw/"trial").glob("raw-*/*.safetensors"))],script_sha256=sha256_file(Path(__file__)),
        scope="CPU full-request/source/raw/PNG/denominator verification; complete latents captured only with snapshot off; no cross-NFE quality equivalence or physical/GPU exclusivity qualification")
    write_json(args.output,result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
