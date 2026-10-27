#!/usr/bin/env python3
"""CPU/source audit of the standalone tokenizer cache; no new inference."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from PIL import Image

from flux9_bf16_framework_screen import write_json
import qwen_tokenizer_reuse_screen as screen
from qwen_tokenizer_reuse_raw import analyze as raw_analyze
from runtime_ane_common import sha256_file

ROOT=Path(__file__).resolve().parents[2]


def snapshots(root,summary):
    values=json.loads((root/"source.json").read_text())
    for name,digest in summary["source_sha256"].items():
        if sha256_file(ROOT/name)!=digest or hashlib.sha256(values[name].encode()).hexdigest()!=digest:
            raise ValueError("source/snapshot differs: "+name)


def memory(root,data):
    for name in ("evidence","report"):
        if sha256_file(root/data[name])!=data[name+"_sha256"]:raise ValueError("memory artifact changed")
    actual=json.loads((root/data["report"]).read_text())
    if actual!=data["verified"] or not actual["complete"] or actual["command_exit_code"]!=0:
        raise ValueError("actual process-tree completion differs")
    return {key:actual[key] for key in ("tree_peak_phys_footprint_bytes","swap_in_bytes","swap_out_bytes","max_gap_ns","sample_count")}


def pixels(path):
    with Image.open(path) as image:
        image.load()
        if image.size!=(512,512):raise ValueError("complete 512 PNG required")
        return np.asarray(image.convert("RGBA"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library",type=Path,required=True);p.add_argument("--full",type=Path,required=True)
    p.add_argument("--raw",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--raw-five",type=Path)
    p.add_argument("--cpu",type=Path,default=ROOT/"outputs/qwen512-tokenizer-cpu-v2-20261018.json")
    p.add_argument("--cpu-binary",type=Path,default=ROOT/"build/qwen21-tokenizer-reuse-v2-20261018/qwen21-tokenizer-reuse-probe")
    p.add_argument("--historical",type=Path,default=ROOT/"outputs/qwen512-five-step-quality-20261018/trial")
    args=p.parse_args()
    for name in ("library","full","raw","cpu","cpu_binary","historical"):
        setattr(args,name,getattr(args,name).resolve(strict=True))
    if args.output.exists():p.error("fresh audit output required")
    if args.raw_five is not None:args.raw_five=args.raw_five.resolve(strict=True)
    library_sha=sha256_file(args.library)
    manifest_path=args.library.parent/"runtime-build/runtime-build-manifest.json"
    manifest=json.loads(manifest_path.read_text())
    for item in manifest["inputs"]["sources"]:
        if sha256_file(ROOT/item["path"])!=item["sha256"]:raise ValueError("source changed after build: "+item["path"])
    full=json.loads((args.full/"summary.json").read_text());raw=json.loads((args.raw/"summary.json").read_text())
    for root,summary in ((args.full,full),(args.raw,raw)):
        if summary["status"]!="complete" or summary["asset_sha256"]["library"]!=library_sha or summary["qualification_passed"] is not False:
            raise ValueError("complete same-library unqualified observation required")
        snapshots(root,summary)
    actual=screen.analyze(args.full)
    if actual!=full["analysis"]:raise ValueError("complete request comparison could not be reproduced")
    raw_actual=raw_analyze(args.raw/"trial",raw["steps"])
    if raw_actual!=raw["analysis"] or raw["performance_denominator"] is not False:
        raise ValueError("raw parity evidence differs")
    observations=[]
    for trial in full["trials"]:
        name=trial["name"];root=args.full/name;report=json.loads((root/"report.json").read_text())
        loads=[json.loads(line) for line in (root/"load.jsonl").read_text().splitlines()]
        events=[json.loads(line) for line in (root/"stdout.log").read_text().splitlines() if line.strip()]
        if events!=[dict(index=i,**point) for i,point in enumerate(screen.points())]:
            raise ValueError("actual configuration event sequence differs")
        observations.append(dict(name=name,memory=memory(root,trial["memory"]),
            competing_load_samples=sum(bool(row["competing_processes"]) for row in loads),load_samples=len(loads),
            load_error=trial.get("load_error"),report_sha256=sha256_file(root/"report.json"),
            request_sha256=[sha256_file(root/f"request-{i}.json") for i in range(8)],
            png_sha256=[sha256_file(root/f"{i}.png") for i in range(8)]))
    # Historical PNG parity is NOT a historical timing denominator.
    history=[]
    for current,old in ((0,0),(1,4),(2,8),(3,12),(4,16),(5,20),(6,24)):
        if not np.array_equal(pixels(args.full/"control-before"/f"{current}.png"),pixels(args.historical/f"{old}.png")):
            raise ValueError("ordinary six-step output changed from historical library")
        history.append(dict(current_index=current,historical_index=old,rgba_equal=True,
                            historical_png_sha256=sha256_file(args.historical/f"{old}.png")))
    for capture,point in ((0,0),(1,0),(2,4),(3,4)):
        if not np.array_equal(pixels(args.raw/"trial"/f"{capture}.png"),pixels(args.full/"reuse"/f"{point}.png")):
            raise ValueError("raw capture did not reproduce full-request image")
    raw_five=None
    if args.raw_five is not None:
        five=json.loads((args.raw_five/"summary.json").read_text())
        if five["status"]!="complete" or five["steps"]!=5 or five["performance_denominator"] is not False or\
                five["asset_sha256"]!=raw["asset_sha256"]:raise ValueError("same-library/source five-step parity capture required")
        snapshots(args.raw_five,five)
        actual_five=raw_analyze(args.raw_five/"trial",5)
        if actual_five!=five["analysis"]:raise ValueError("five-step raw audit differs")
        for capture,historical in ((0,1),(1,1),(2,17),(3,17)):
            if not np.array_equal(pixels(args.raw_five/"trial"/f"{capture}.png"),pixels(args.historical/f"{historical}.png")):
                raise ValueError("CPU reuse changed historical five-step image")
        raw_five=dict(analysis=actual_five,memory=memory(args.raw_five,five["memory"]),historical_rgba_restored=True,
                      summary_sha256=sha256_file(args.raw_five/"summary.json"))
    cpu=json.loads(args.cpu.read_text())
    if cpu["status"]!="complete" or cpu["loaded_runtime_id"]!=manifest["runtime_build_id"] or cpu["all_token_ids_equal"] is not True or\
            cpu["source_sha256"]!=screen.PROCESSOR_SHA:raise ValueError("current CPU component identity differs")
    cpu_medians={str(reuse):statistics.median(row["seconds"] for row in cpu["samples"] if row["iteration"]>=0 and row["reuse"] is reuse)
                 for reuse in (False,True)}
    result=dict(schema="tc-qwen-tokenizer-reuse-audit-v1",status="complete",qualification_passed=False,
        library_sha256=library_sha,runtime_build_id=manifest["runtime_build_id"],native_source_hashes_verified=len(manifest["inputs"]["sources"]),
        manifest_sha256=sha256_file(manifest_path),full_requests=24,raw_requests=8 if raw_five else 4,analysis=actual,observations=observations,
        raw=raw_actual,raw_memory=memory(args.raw,raw["memory"]),raw_five=raw_five,historical_rgba_restoration=history,
        cpu_component_median_seconds=cpu_medians,cpu_component_sha256=sha256_file(args.cpu),cpu_binary_sha256=sha256_file(args.cpu_binary),
        raw_capture_rgba_matches_full=True,script_sha256=sha256_file(Path(__file__)),
        full_summary_sha256=sha256_file(args.full/"summary.json"),raw_summary_sha256=sha256_file(args.raw/"summary.json"),
        scope="CPU/source recheck only; preserved full six-step arithmetic; no physical engine/exclusive-load proof")
    write_json(args.output,result);print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__":main()
