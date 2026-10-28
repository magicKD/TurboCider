#!/usr/bin/env python3
"""CPU-only bounded-pair pilot/cancel/source/raw/PNG denominator verification."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image

from flux9_bf16_framework_screen import write_json
import qwen_async_step_pair_screen as screen
import qwen_async_step_cancel_screen as cancel_screen
from runtime_ane_common import sha256_file

ROOT=Path(__file__).resolve().parents[2]


def verify_sources(root,s):
    source=json.loads((root/"source.json").read_text())
    for name,digest in s["source_sha256"].items():
        if hashlib.sha256(source[name].encode()).hexdigest()!=digest or sha256_file(ROOT/name)!=digest:raise ValueError("bound source snapshot/current tree differs")
    for name in ("evidence","report"):
        if sha256_file(root/s["memory"][name])!=s["memory"][name+"_sha256"]:raise ValueError("process evidence changed")
    if not s["memory"]["verified"]["complete"] or s["memory"]["verified"]["command_exit_code"]!=0:raise ValueError("complete owned process window required")


def window(root,library_sha,runtime):
    s=json.loads((root/"summary.json").read_text());r=json.loads((root/"trial/report.json").read_text())
    if s["status"]!="complete" or s["asset_sha256"]["library"]!=library_sha or r["loaded_runtime_id"]!=runtime:raise ValueError("complete same-library window required")
    verify_sources(root,s);seq=screen.points(s["mode"],s["case"],s["steps"],s["async_pair"]=="on",s["prefix"])
    if seq!=s["frozen_sequence"] or screen.analyze(root/"trial",r,seq,s["mode"])!=s["analysis"]:raise ValueError("complete raw/pixel/coverage/time recheck differs")
    return s,r,seq,screen.base.prefix_screen.images(root/"trial",len(seq))


def normalized(path):
    r=json.loads(path.read_text())
    for key in ("output","dump_tensors"):r.pop(key,None)
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--library",type=Path,required=True);p.add_argument("--native-source-archive",type=Path,required=True)
    p.add_argument("--raw",type=Path,required=True);p.add_argument("--cancel",type=Path,required=True);p.add_argument("--cancel-binary",type=Path,required=True)
    p.add_argument("--cohort",type=Path,action="append",required=True);p.add_argument("--previous-quality",type=Path,required=True);p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error("fresh audit output required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text());inputs=manifest["inputs"]["sources"]
    with tarfile.open(args.native_source_archive,"r:gz") as archive:
        members={member.name:member for member in archive.getmembers()}
        if set(members)!={entry["path"] for entry in inputs}:raise ValueError("complete linked native source archive required")
        for entry in inputs:
            f=archive.extractfile(members[entry["path"]])
            if f is None or hashlib.sha256(f.read()).hexdigest()!=entry["sha256"] or sha256_file(ROOT/entry["path"])!=entry["sha256"]:
                raise ValueError("archive/current source differs from linked library")
    sha=sha256_file(args.library);runtime=manifest["runtime_build_id"];raw=window(args.raw,sha,runtime)
    if raw[0]["mode"]!="raw":raise ValueError("separate32-request raw matrix required")
    previous=json.loads((args.previous_quality/"summary.json").read_text())
    if previous["status"]!="complete" or not previous["analysis"]["all_same_tier_rgba_exact"]:raise ValueError("previous complete v7 matrix required")
    lookup={screen.base.signature(p,with_ffn=True):i for i,p in enumerate(previous["frozen_sequence"])}
    previous_png=0;previous_raw=0;windows=[(args.raw,raw)];cohorts={};observations=[]
    for root in args.cohort:
        w=window(root,sha,runtime);s=w[0];key=s["case"],s["prefix"],s["async_pair"]
        if s["mode"]!="timing" or s["steps"]!=5 or key in cohorts:raise ValueError("one fixed five-step pilot per task/arm required")
        cohorts[key]=s["analysis"];windows.append((root,w))
        loads=[json.loads(line) for line in (root/"load.jsonl").read_text().splitlines()]
        observations.append(dict(cohort=root.name,case=key[0],prefix=key[1],async_pair=key[2],analysis=s["analysis"],memory=s["memory"]["verified"],
            load_error=s.get("load_error"),competing_load_samples=sum(bool(row["competing_processes"]) for row in loads),summary_sha256=sha256_file(root/"summary.json")))
    if set(cohorts)!={(case,prefix,arm) for case,prefix in (("generate",False),("edit",True)) for arm in ("off","on")}:
        raise ValueError("both strong-baseline pilot pairs required")
    for root,w in windows:
        for i,pnt in enumerate(w[2]):
            j=lookup[screen.base.signature(pnt,with_ffn=True)]
            with Image.open(args.previous_quality/f"trial/{j}.png") as image:old=np.asarray(image.convert("RGBA"))
            if normalized(root/f"trial/request-{i}.json")!=normalized(args.previous_quality/f"trial/request-{j}.json") or not np.array_equal(w[3][i],old):
                raise ValueError("actual request or PNG regressed against previous same-policy matrix")
            previous_png+=1
        if w[0]["mode"]=="raw":
            for pair in w[0]["analysis"]["kernel_pairs"]:
                for name in ("initial","text","latents")+(("reference_0",) if pair["case"]=="edit" else ()):
                    x,y=(screen.base.tensor(root/f"trial/raw-{k}/qwen21_{name}.safetensors") for k in (pair["off_index"],pair["on_index"]))
                    if x[0]!=y[0] or not np.array_equal(x[1],y[1]):raise ValueError("full typed raw arrays differ")
                previous_raw+=1
    c=json.loads((args.cancel/"summary.json").read_text());native=json.loads((args.cancel/"trial/native.json").read_text())
    if c["status"]!="complete" or c["asset_sha256"]["library"]!=sha or c["asset_sha256"]["binary"]!=sha256_file(args.cancel_binary) or native["loaded_runtime_id"]!=runtime:
        raise ValueError("complete actual-library cancel/reuse model evidence required")
    verify_sources(args.cancel,c)
    if cancel_screen.analyze(args.cancel/"trial",native)!=c["analysis"]:raise ValueError("cancelled progress/PNG/restore recheck differs")
    timings=[]
    for case,prefix in (("generate",False),("edit",True)):
        off=cohorts[case,prefix,"off"]["median_seconds"];on=cohorts[case,prefix,"on"]["median_seconds"]
        timings.append(dict(case=case,prefix=prefix,sync_seconds=off,async_pair_seconds=on,reduction_percent=100*(1-on/off)))
    rows=[row for _,w in windows for row in w[1]["runs"]];async_rows=[row["metrics"]["qwen_async_step_pair"] for row in rows if row["metrics"].get("qwen_async_step_pair")]
    result=dict(schema="tc-qwen-async-step-pilot-audit-v1",status="complete",qualification_passed=False,library_sha256=sha,runtime_build_id=runtime,
        native_source_hashes_verified=len(inputs),native_source_archive_sha256=sha256_file(args.native_source_archive),raw_analysis=raw[0]["analysis"],cancel_analysis=c["analysis"],
        observations=observations,timings=timings,successful_full_requests=len(rows)+8,cancelled_partial_requests=2,raw_same_tier_exact_pairs=previous_raw,
        previous_matrix_exact_png_requests=previous_png,async_screen_requests=len(async_rows),submitted_steps=sum(a["submitted_steps"] for a in async_rows),
        checked_steps=sum(a["checked_steps"] for a in async_rows),accepted_steps=sum(a["accepted_steps"] for a in async_rows),
        host_ahead_observations=sum(a["host_ahead_observations"] for a in async_rows),completion_batches=sum(a["completion_batches"] for a in async_rows),
        previous_quality_sha256=sha256_file(args.previous_quality/"summary.json"),script_sha256=sha256_file(Path(__file__)),
        scope="CPU same-input raw/PNG/cancel/source and pilot denominators; no larger matrix, physical overlap, device-loss fault or formal performance qualification")
    write_json(args.output,result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
