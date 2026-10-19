#!/usr/bin/env python3
"""CPU-only four-arm composition recheck; no cross-NFE equivalence claim."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image

from flux9_bf16_framework_screen import write_json
import qwen_sampler_qkv_composition_screen as screen
from runtime_ane_common import sha256_file

ROOT=Path(__file__).resolve().parents[2]


def window(root,library_sha,runtime_id):
    s=json.loads((root/"summary.json").read_text());r=json.loads((root/"trial/report.json").read_text())
    if s["status"]!="complete" or s["asset_sha256"]["library"]!=library_sha or r["loaded_runtime_id"]!=runtime_id:
        raise ValueError("complete same-library composition window required")
    snapshot=json.loads((root/"source.json").read_text())
    for name,digest in s["source_sha256"].items():
        if hashlib.sha256(snapshot[name].encode()).hexdigest()!=digest or sha256_file(ROOT/name)!=digest:
            raise ValueError("bound composition source differs")
    sequence=screen.points(s["mode"],s["case"],s["arm"])
    if sequence!=s["frozen_sequence"]:raise ValueError("frozen four-arm workload changed")
    a=screen.analyze(root/"trial",r,sequence,s["mode"],write_sheet=False)
    if a!=s["analysis"]:raise ValueError("full-array/whole-request composition summary differs")
    memory=s["memory"]
    for key in ("evidence","report"):
        if sha256_file(root/memory[key])!=memory[key+"_sha256"]:raise ValueError("memory evidence changed")
    if not memory["verified"]["complete"] or memory["verified"]["command_exit_code"]!=0:raise ValueError("incomplete composition process window")
    pixels=screen.load_images(root/"trial",len(sequence));requests=[]
    for i in range(len(sequence)):
        request=json.loads((root/f"trial/request-{i}.json").read_text())
        for key in ("output","dump_tensors"):request.pop(key,None)
        requests.append(request)
    return s,r,sequence,pixels,requests


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--library",type=Path,required=True)
    p.add_argument("--native-source-archive",type=Path,required=True);p.add_argument("--raw",type=Path,required=True)
    p.add_argument("--quality",type=Path,required=True);p.add_argument("--cohort",type=Path,action="append",required=True)
    p.add_argument("--previous-six-quality",type=Path);p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error("fresh audit output required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text());sources=manifest["inputs"]["sources"]
    library_sha=sha256_file(args.library);runtime_id=manifest["runtime_build_id"]
    with tarfile.open(args.native_source_archive,"r:gz") as archive:
        members={member.name:member for member in archive.getmembers()}
        if set(members)!={item["path"] for item in sources}:raise ValueError("complete native source archive required")
        for item in sources:
            stream=archive.extractfile(members[item["path"]])
            if stream is None or hashlib.sha256(stream.read()).hexdigest()!=item["sha256"] or sha256_file(ROOT/item["path"])!=item["sha256"]:
                raise ValueError("archived/current source differs from linked library")
    raw=window(args.raw,library_sha,runtime_id);quality=window(args.quality,library_sha,runtime_id)
    if raw[0]["mode"]!="raw" or quality[0]["mode"]!="quality":raise ValueError("separate raw/quality windows required")
    lookup={(point["case_id"],point["arm"],point["seed"]):i for i,point in enumerate(quality[2])}
    for i,point in enumerate(raw[2]):
        key=("photo" if point["case"]=="generate" else "edit-color",point["arm"],point["seed"]);q=lookup[key]
        if raw[4][i]!=quality[4][q] or not np.array_equal(raw[3][i],quality[3][q]):raise ValueError("canonical/matrix request or PNG differs")
        a,b=(screen.base.tensor(path) for path in (args.raw/f"trial/raw-{i}/qwen21_latents.safetensors",args.quality/f"trial/raw-{q}/qwen21_latents.safetensors"))
        if a[0]!=b[0] or not np.array_equal(a[1],b[1]):raise ValueError("canonical/matrix final latents differ")
    previous=None
    if args.previous_six_quality:
        ps=json.loads((args.previous_six_quality/"summary.json").read_text())
        if ps["status"]!="complete" or not ps["analysis"]["all_rgba_exact"] or not ps["analysis"]["all_latents_exact"]:
            raise ValueError("previous complete six-step exact-control matrix required")
        count=0
        for task in range(7):
            for new_offset,old_offset in ((0,0),(1,1),(6,2),(7,3)):
                i=task*8+new_offset;j=task*4+old_offset
                with Image.open(args.previous_six_quality/f"trial/{j}.png") as im:old=np.asarray(im.convert("RGBA"))
                if not np.array_equal(quality[3][i],old):raise ValueError("six-step image regressed across builds")
                values=[screen.base.tensor(path) for path in (args.quality/f"trial/raw-{i}/qwen21_latents.safetensors",args.previous_six_quality/f"trial/raw-{j}/qwen21_latents.safetensors")]
                if values[0][0]!=values[1][0] or not np.array_equal(values[0][1],values[1][1]):raise ValueError("six-step latent regressed across builds")
                count+=1
        previous=dict(summary_sha256=sha256_file(args.previous_six_quality/"summary.json"),library_sha256=ps["asset_sha256"]["library"],six_step_exact_pairs=count)
    cohorts={};observations=[];completed=len(raw[2])+len(quality[2])
    for root in args.cohort:
        s,r,sequence,pixels,requests=window(root,library_sha,runtime_id);key=(s["case"],s["arm"])
        if s["mode"]!="timing" or key in cohorts:raise ValueError("one fixed-arm window per task/arm required")
        for i,point in enumerate(sequence):
            q=lookup[("photo" if point["case"]=="generate" else "edit-color",point["arm"],point["seed"])]
            if requests[i]!=quality[4][q] or not np.array_equal(pixels[i],quality[3][q]):raise ValueError("timing/quality request or output differs")
        loads=[json.loads(line) for line in (root/"load.jsonl").read_text().splitlines()]
        cohorts[key]=s["analysis"];completed+=len(sequence)
        observations.append(dict(cohort=root.name,case=s["case"],arm=s["arm"],analysis=s["analysis"],memory=s["memory"]["verified"],
            load_error=s.get("load_error"),competing_load_samples=sum(bool(row["competing_processes"]) for row in loads),summary_sha256=sha256_file(root/"summary.json")))
    if set(cohorts)!={(case,arm) for case in ("generate","edit") for arm in screen.ARMS}:raise ValueError("both complete four-arm task cohorts required")
    timings=[]
    for case in ("generate","edit"):
        m={arm:cohorts[case,arm]["median_seconds"] for arm in screen.ARMS}
        timings.append(dict(case=case,medians=m,six_kernel_reduction_percent=100*(1-m["six-direct"]/m["six-original"]),
            five_kernel_reduction_percent=100*(1-m["five-direct"]/m["five-original"]),
            nfe_only_reduction_vs_six_original_percent=100*(1-m["five-original"]/m["six-original"]),
            combined_reduction_vs_six_original_percent=100*(1-m["five-direct"]/m["six-original"]),
            equivalent_work_scope="kernel pairs within the same tier only; NFE/combined are distinct quality-tier comparisons"))
    result=dict(schema="tc-qwen-sampler-qkv-composition-audit-v1",status="complete",qualification_passed=False,library_sha256=library_sha,runtime_build_id=runtime_id,
        native_source_hashes_verified=len(sources),native_source_archive_sha256=sha256_file(args.native_source_archive),
        raw_summary_sha256=sha256_file(args.raw/"summary.json"),quality_summary_sha256=sha256_file(args.quality/"summary.json"),
        raw_analysis=raw[0]["analysis"],quality_analysis=quality[0]["analysis"],previous_six_step_control=previous,observations=observations,timings=timings,
        raw_tensor_sha256=[dict(path=str(path),sha256=sha256_file(path)) for root in (args.raw,args.quality) for path in sorted((root/"trial").glob("raw-*/*.safetensors"))],
        completed_full_requests=completed,script_sha256=sha256_file(Path(__file__)),
        scope="CPU/raw/pixel/source/denominator audit; four-arm actual measurements, no multiplied historical gains or cross-tier arithmetic/quality equivalence")
    write_json(args.output,result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
