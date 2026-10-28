#!/usr/bin/env python3
"""CPU-only independent whole-request/raw/source/bank-lifetime recheck."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image

from flux9_bf16_framework_screen import write_json
import qwen_corrected_qkv_request_screen as screen
import qwen_corrected_qkv_quality_matrix as matrix
import qwen_corrected_qkv_storage_switch as storage_switch
from qwen_student_sampler_raw import tensor
from runtime_ane_common import sha256_file

ROOT=Path(__file__).resolve().parents[2]


def check_window(root,library_sha,runtime_id):
    s=json.loads((root/"summary.json").read_text());r=json.loads((root/"trial/report.json").read_text())
    if s["status"]!="complete" or s["asset_sha256"]["library"]!=library_sha or r["loaded_runtime_id"]!=runtime_id:
        raise ValueError("complete same-library request window required")
    snapshots=json.loads((root/"source.json").read_text())
    for name,digest in s["source_sha256"].items():
        if hashlib.sha256(snapshots[name].encode()).hexdigest()!=digest or sha256_file(ROOT/name)!=digest:
            raise ValueError("bound request source differs")
    sequence=screen.points(s["mode"],s["case"],s["arm"],s.get("direct",False))
    if sequence!=s["frozen_sequence"]:raise ValueError("frozen complete-request sequence differs")
    screen.validate(r,sequence,s.get("expected_recipe"))
    actual=screen.analyze(root/"trial",r,sequence,s["mode"])
    if actual!=s["analysis"]:raise ValueError("raw quality or complete-request denominator differs")
    memory=s["memory"]
    for key in ("evidence","report"):
        if sha256_file(root/memory[key])!=memory[key+"_sha256"]:raise ValueError("memory evidence changed")
    if not memory["verified"]["complete"] or memory["verified"]["command_exit_code"]!=0:raise ValueError("incomplete process-tree window")
    images=[];requests=[]
    for i in range(len(sequence)):
        path=root/f"trial/{i}.png"
        with Image.open(path) as im:images.append(np.asarray(im.convert("RGBA")))
        request=json.loads((root/f"trial/request-{i}.json").read_text())
        for key in ("output","dump_tensors"):request.pop(key,None)
        requests.append(request)
    return s,r,sequence,images,requests


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--library",type=Path,required=True)
    p.add_argument("--raw",type=Path,required=True);p.add_argument("--cohort",type=Path,action="append",required=True)
    p.add_argument("--source-archive",type=Path,required=True);p.add_argument("--matrix",type=Path)
    p.add_argument("--switch",type=Path)
    p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error("fresh audit output required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    sources=manifest["inputs"]["sources"];library_sha=sha256_file(args.library);runtime_id=manifest["runtime_build_id"]
    with tarfile.open(args.source_archive,"r:gz") as archive:
        members={item.name:item for item in archive.getmembers()}
        if set(members)!={item["path"] for item in sources}:raise ValueError("complete native-source archive required")
        for item in sources:
            saved=archive.extractfile(members[item["path"]])
            if saved is None or hashlib.sha256(saved.read()).hexdigest()!=item["sha256"] or sha256_file(ROOT/item["path"])!=item["sha256"]:
                raise ValueError("archived/current native source differs from linked library")
    raw=check_window(args.raw,library_sha,runtime_id);rs,rr,rpoints,rimages,rrequests=raw
    if rs["mode"]!="raw":raise ValueError("parity-only raw window required")
    banks=[];raw_files=[]
    for base in (0,5):
        for a,b in ((base,base+4),(base+1,base+3)):
            values=[tensor(args.raw/f"trial/raw-{i}/qwen21_latents.safetensors") for i in (a,b)]
            if values[0][0]!=values[1][0] or not np.array_equal(values[0][1],values[1][1]):raise ValueError("raw same-policy latent restoration failed")
        noise=[tensor(args.raw/f"trial/raw-{i}/qwen21_initial.safetensors") for i in (base+1,base+2)]
        if np.array_equal(noise[0][1],noise[1][1]):raise ValueError("changed seed reused old raw noise")
        active=[rr["runs"][base+i]["metrics"]["memory"]["mlx_active_bytes"] for i in range(5)]
        extra=0 if rs.get("direct",False) else 3<<30
        if active[0]!=active[4] or any(active[i]-active[0]!=extra for i in (1,2,3)):
            raise ValueError("actual direct/packed active ownership did not restore")
        banks.append(dict(case=rpoints[base]["case"],mlx_active_bytes=active,extra_active_bank_bytes=extra,
            optout_restored_exact=True,scope="MLX active owners only, not OS/driver physical-memory release"))
    for path in sorted((args.raw/"trial").glob("raw-*/*.safetensors")):
        raw_files.append(dict(path=str(path.relative_to(args.raw)),sha256=sha256_file(path)))
    cohorts={};observations=[];completed_requests=len(rpoints)
    for root in args.cohort:
        s,r,sequence,images,requests=check_window(root,library_sha,runtime_id)
        key=(s["case"],s["arm"])
        if s["mode"]!="timing" or key in cohorts:raise ValueError("one complete timing window per task/arm required")
        for i,point in enumerate(sequence):
            raw_index=(0 if point["case"]=="generate" else 5)+(1 if point["on"] else 0)
            if point["seed"]==73 and (requests[i]!=rrequests[raw_index] or not np.array_equal(images[i],rimages[raw_index])):
                raise ValueError("unmatched task/recipe or raw/performance image changed")
        loads=[json.loads(line) for line in (root/"load.jsonl").read_text().splitlines()]
        cohorts[key]=s["analysis"]
        observations.append(dict(cohort=root.name,case=s["case"],arm=s["arm"],analysis=s["analysis"],
            summary_sha256=sha256_file(root/"summary.json"),memory=s["memory"]["verified"],
            competing_load_samples=sum(bool(row["competing_processes"]) for row in loads),load_error=s.get("load_error")))
        completed_requests+=len(sequence)
    if set(cohorts)!={(case,arm) for case in ("generate","edit") for arm in ("off","on")}:
        raise ValueError("both complete original/corrected task timing pairs required")
    timings=[]
    for case in ("generate","edit"):
        control=cohorts[case,"off"];candidate=cohorts[case,"on"]
        timings.append(dict(case=case,original=control,corrected=candidate,
            reduction_percent=100*(1-candidate["median_seconds"]/control["median_seconds"])))
    matrix_result=None
    if args.matrix:
        ms=json.loads((args.matrix/"summary.json").read_text());mr=json.loads((args.matrix/"trial/report.json").read_text())
        if ms["status"]!="complete" or ms["asset_sha256"]["library"]!=library_sha or mr["loaded_runtime_id"]!=runtime_id:
            raise ValueError("complete same-library quality matrix required")
        snapshot=json.loads((args.matrix/"source.json").read_text())
        for name,digest in ms["source_sha256"].items():
            if hashlib.sha256(snapshot[name].encode()).hexdigest()!=digest or sha256_file(ROOT/name)!=digest:
                raise ValueError("bound quality-matrix source differs")
        if ms["frozen_sequence"]!=matrix.points(ms.get("direct",False)):raise ValueError("frozen broader quality matrix changed")
        ma=matrix.analyze(args.matrix/"trial",mr,ms.get("direct",False))
        if ma!=ms["analysis"]:raise ValueError("quality matrix could not be independently reproduced")
        memory=ms["memory"]
        for key in ("evidence","report"):
            if sha256_file(args.matrix/memory[key])!=memory[key+"_sha256"]:raise ValueError("matrix memory evidence changed")
        if not memory["verified"]["complete"] or memory["verified"]["command_exit_code"]!=0:raise ValueError("incomplete matrix window")
        for matrix_index,raw_index in ((0,0),(1,1),(16,5),(17,6)):
            with Image.open(args.matrix/f"trial/{matrix_index}.png") as im:value=np.asarray(im.convert("RGBA"))
            if not np.array_equal(value,rimages[raw_index]):raise ValueError("matching matrix/canonical raw image differs")
        matrix_result=dict(summary_sha256=sha256_file(args.matrix/"summary.json"),analysis=ma,memory=memory["verified"],
            tensor_sha256=[dict(path=str(path.relative_to(args.matrix)),sha256=sha256_file(path)) for path in sorted((args.matrix/"trial").glob("raw-*/*.safetensors"))])
        completed_requests+=len(matrix.points(ms.get("direct",False)))
    switch_result=None
    if args.switch:
        ss=json.loads((args.switch/"summary.json").read_text());sr=json.loads((args.switch/"trial/report.json").read_text())
        if ss["status"]!="complete" or ss["asset_sha256"]["library"]!=library_sha or sr["loaded_runtime_id"]!=runtime_id:
            raise ValueError("complete same-library storage switch required")
        snapshot=json.loads((args.switch/"source.json").read_text())
        for name,digest in ss["source_sha256"].items():
            if hashlib.sha256(snapshot[name].encode()).hexdigest()!=digest or sha256_file(ROOT/name)!=digest:
                raise ValueError("bound storage-switch source differs")
        if ss["frozen_sequence"]!=storage_switch.points(ss["case"]):raise ValueError("frozen storage switch changed")
        sa=storage_switch.analyze(args.switch/"trial",sr,ss["case"])
        if sa!=ss["analysis"]:raise ValueError("actual bank-switch lifecycle differs")
        memory=ss["memory"]
        for key in ("evidence","report"):
            if sha256_file(args.switch/memory[key])!=memory[key+"_sha256"]:raise ValueError("storage-switch memory evidence changed")
        if not memory["verified"]["complete"] or memory["verified"]["command_exit_code"]!=0:raise ValueError("incomplete storage-switch window")
        switch_result=dict(summary_sha256=sha256_file(args.switch/"summary.json"),analysis=sa,memory=memory["verified"],
            tensor_sha256=[dict(path=str(path.relative_to(args.switch)),sha256=sha256_file(path)) for path in sorted((args.switch/"trial").glob("raw-*/*.safetensors"))])
        completed_requests+=len(storage_switch.points(ss["case"]))
    result=dict(schema="tc-qwen-corrected-qkv-request-audit-v1",status="complete",qualification_passed=False,
        runtime_build_id=runtime_id,library_sha256=library_sha,native_source_hashes_verified=len(sources),
        native_source_archive_sha256=sha256_file(args.source_archive),raw_summary_sha256=sha256_file(args.raw/"summary.json"),
        raw_analysis=rs["analysis"],raw_tensor_sha256=raw_files,bank_ownership=banks,observations=observations,timings=timings,
        completed_full_requests=completed_requests,quality_matrix=matrix_result,storage_switch=switch_result,script_sha256=sha256_file(Path(__file__)),
        scope="CPU-only recomputation and raw/performance PNG identity; original quality gates retained; no physical/exclusive/semantic qualification")
    write_json(args.output,result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
