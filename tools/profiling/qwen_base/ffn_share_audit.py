#!/usr/bin/env python3
"""Independent matched full-block neighboring FFN share audit; not image speed."""
import argparse
import json
import math
from pathlib import Path
import statistics
import tarfile

import ffn_projection_audit as block
from collect import ROOT,model_snapshot,sha256_file,write_json
import joint_audit

SHARES=(5632,6144,6656)


def neighbor_contract(summary,report,runtime_id):
    c=summary.get("channels")
    if (type(c) is not int or c not in SHARES or report.get("ane_channels")!=c or report.get("gpu_channels")!=12288-c or
        report.get("compiled_native_id")!=runtime_id or report.get("neighbor_share_screen") is not True or report.get("handoff_screen") is not True or
        report.get("gpu_complement_strategy")!="separate" or report.get("gpu_packed_weight_bytes")!=0 or report.get("gpu_packed_setup_seconds")!=0 or
        summary.get("projection")!="descriptor" or summary.get("activation")!="sigmoid-complement"):
        raise ValueError("explicit bound original neighboring fixed share/complement recipe required")
    result={k:[] for k in ("prediction_wait_seconds","output_wait_seconds","caller_wait_seconds")}
    repeats=summary["repeats"]
    for s in report["samples"]:
        if s["arm"]!="private-restaged":continue
        d=s.get("completion_deltas")
        if not isinstance(d,dict) or set(d)!=set(result) or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in d.values()):
            raise ValueError("actual finite native completion deltas for each private block required")
        if not s["warmup"]:
            for k in result:result[k].append(d[k])
    if any(len(x)!=repeats for x in result.values()):raise ValueError("complete measured private completion counts required")
    complement=report.get("gpu_complement_alone_samples",[])
    if len(complement)!=repeats+2 or [s["rep"] for s in complement]!=list(range(-2,repeats)):
        raise ValueError("all independent ready-input GPU complement samples required")
    for s in complement:
        if (s.get("warmup") is not (s["rep"]<0) or s.get("input_variant") is not bool(s["rep"]&1) or
            type(s.get("seconds")) not in (int,float) or not math.isfinite(s["seconds"]) or s["seconds"]<=0):raise ValueError("actual positive independent GPU-complement boundary required")
    return dict(native_hot_medians={k:statistics.median(x) for k,x in result.items()},
                gpu_complement_alone_seconds=statistics.median(s["seconds"] for s in complement if not s["warmup"]),
                scope="overlapping host completion/GPU-complement boundaries, not additive or pure hardware compute")


def matched(records):
    if not records or records[0]["channels"]!=6144 or records[0]["mode"]!="stock":raise ValueError("first current stock6144 strong control required")
    anchor=records[0];seen={};rows=[]
    for r in records:
        if r["case"]!=anchor["case"] or any(r["array_sha256"][k]!=anchor["array_sha256"][k] for k in
            ("ffn_input","block_reference","alternate_reference","other_weight_reference")):
            raise ValueError("actual same source and ordinary full-GPU controls required across shares")
        key=r["channels"]
        if key in seen and r["array_sha256"]!=seen[key]:raise ValueError("same-share repeat/timeline must restore all complete arrays")
        seen[key]=r["array_sha256"]
        median=r["analysis"]["median_seconds"]["private-restaged"]
        rows.append(dict(cohort=r["cohort"],channels=key,mode=r["mode"],seconds=median,
            reduction_vs_first6144=1-median/anchor["analysis"]["median_seconds"]["private-restaged"] if r["mode"]=="stock" else None,
            scope="complete real block candidate; not forty-step image speedup"))
    return dict(identical_source_and_gpu_controls=True,same_share_replays_exact=True,comparisons=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("native-library","native-source-archive","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--cohort",nargs=2,metavar=("ROOT","PROBE_LIBRARY"),action="append",required=True)
    p.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--tool-source-archive",type=Path,action="append",default=[]);args=p.parse_args()
    if args.output.exists():p.error("fresh share audit output required")
    manifest=json.loads((args.native_library.parent/"runtime-build/runtime-build-manifest.json").read_text());joint_audit.archive_contract(args.native_source_archive,manifest)
    historical={}
    for path in args.tool_source_archive:
        with tarfile.open(path,"r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():historical.setdefault(member.name,set()).add(archive.extractfile(member).read())
    identity=model_snapshot(args.model);payloads=block.original_hashes(args.model);records=[];traces=[]
    for root,probe in args.cohort:
        root=Path(root);summary=json.loads((root/"summary.json").read_text());report=json.loads((root/"trial/report.json").read_text())
        extra=neighbor_contract(summary,report,manifest["runtime_build_id"])
        r,events=block.window(root,args.native_library,Path(probe),manifest,identity,payloads,historical)
        r["completion"]=extra;records.append(r)
        if events:traces.append((root.name,events))
    result=dict(schema="tc-neighbor-ffn-share-audit-v1",status="complete",qualification_passed=False,runtime_build_id=manifest["runtime_build_id"],
        native_library_sha256=sha256_file(args.native_library),native_source_archive_sha256=sha256_file(args.native_source_archive),
        script_sha256=sha256_file(Path(__file__)),matched=matched(records),cohorts=records,
        scope="Real original BF16 base40 first-decode block capture/replays; fixed FFN only, QKV GPU; no full40 image qualification")
    args.output.mkdir(parents=True)
    for name,events in traces:write_json(args.output/(name+"-trace.json"),events)
    write_json(args.output/"audit.json",result);print(json.dumps(result["matched"],indent=2))


if __name__=="__main__":main()
