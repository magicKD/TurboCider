#!/usr/bin/env python3
"""Read-only CPU recheck of sampler images/requests/source/memory evidence.

Writes only a fresh audit output. Does not rerun inference, rewrite cohort
artifacts, measure engines, or grant instruction-quality/exclusive-load status.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

import numpy as np
from PIL import Image

from flux9_bf16_framework_screen import write_json
from qwen21_dit_cache_report import image_metrics
import qwen_student_sampler_screen as screen
from runtime_ane_common import sha256_file

ROOT=Path(__file__).resolve().parents[2]


def pixels(path):
    with Image.open(path) as im:
        im.load()
        if im.size!=(512,512): raise ValueError("complete 512 output required")
        return np.asarray(im.convert("RGBA"))


def memory(root,summary):
    data=summary["memory"]
    for key in ("evidence","report"):
        if sha256_file(root/data[key])!=data[key+"_sha256"]:
            raise ValueError("memory artifact changed")
    actual=json.loads((root/data["report"]).read_text())
    if actual!=data["verified"] or not actual["complete"] or actual["command_exit_code"]!=0:
        raise ValueError("memory completion differs")
    return {key:actual[key] for key in ("tree_peak_phys_footprint_bytes","swap_in_bytes","swap_out_bytes",
                                        "max_gap_ns","sample_count")}


def cohort(root,library,mode):
    summary=json.loads((root/"summary.json").read_text())
    report=json.loads((root/"trial/report.json").read_text())
    if summary["status"]!="complete" or summary["mode"]!=mode or summary["qualification_passed"] is not False:
        raise ValueError("completed unqualified cohort required")
    if summary["asset_sha256"]["library"]!=sha256_file(library):
        raise ValueError("library differs")
    snapshots=json.loads((root/"source.json").read_text())
    for relative,digest in summary["source_sha256"].items():
        if sha256_file(ROOT/relative)!=digest or hashlib.sha256(snapshots[relative].encode()).hexdigest()!=digest:
            raise ValueError("source/snapshot differs: "+relative)
    screen.validate(report,summary["seed"],mode)
    sequence=screen.points(summary["seed"],mode)
    if sequence!=summary["frozen_sequence"]:
        raise ValueError("frozen sequence differs")
    # Derive the expected adapter/reference from recorded requests, then verify
    # their pins and all other request fields, not just result labels.
    recorded=[json.loads((root/"trial"/f"request-{i}.json").read_text()) for i in range(len(sequence))]
    lora=Path(recorded[0]["loras"][0]["path"])
    reference=Path(next(r for r in recorded if r["operation"]=="image.edit")["inputs"][0]["path"])
    if any(sha256_file(path)!=summary["asset_sha256"][key] for key,path in (("lora",lora),("reference",reference))):
        raise ValueError("adapter/reference changed")
    args=SimpleNamespace(seed=summary["seed"],mode=mode,lora=lora,reference=reference,output=root/"trial")
    if recorded!=screen.requests(args): raise ValueError("request workload changed")
    events=[json.loads(line) for line in (root/"stdout.log").read_text().splitlines() if line.strip()]
    if events!=[dict(index=i,**point) for i,point in enumerate(sequence)]:
        raise ValueError("configuration event sequence differs")
    images=[pixels(root/"trial"/f"{i}.png") for i in range(len(sequence))]
    pairs=((2,3),(5,4),(9,11),(14,12)) if mode=="timing" else tuple(
        (i+a,i+b) for i in range(0,len(sequence),4) for a,b in ((0,1),(3,2)))
    quality=[]
    for a,b in pairs:
        x,y=images[a],images[b]
        quality.append(dict(case_id=sequence[a]["case_id"],seed=sequence[a]["seed"],six_index=a,five_index=b,
            rgb=image_metrics(x[:,:,:3],y[:,:,:3]),alpha_max_abs=int(np.abs(x[:,:,3].astype(int)-y[:,:,3]).max()),
            alpha_ranges=[[int(z[:,:,3].min()),int(z[:,:,3].max())] for z in (x,y)],
            rgba_equal=bool(np.array_equal(x,y))))
    if quality!=summary["analysis"]["quality_differences"]:
        raise ValueError("image difference could not be reproduced")
    timing=[]
    if mode=="timing":
        for case in ("generate","edit"):
            values={step:[row["request_wall_including_export"] for row,point in zip(report["runs"],sequence)
                          if point["case"]==case and point["score"] and point["steps"]==step] for step in (6,5)}
            a,b=statistics.median(values[6]),statistics.median(values[5])
            timing.append(dict(case=case,six_seconds=values[6],five_seconds=values[5],six_median=a,five_median=b,
                               reduction_percent=100*(1-b/a)))
        if timing!=summary["analysis"]["timing"]: raise ValueError("timing denominator differs")
        for a,b in ((2,6),(3,7),(9,15),(9,19),(11,17)):
            if not np.array_equal(images[a],images[b]): raise ValueError("sampler restoration differs")
    loads=[json.loads(line) for line in (root/"load.jsonl").read_text().splitlines()]
    busy=sum(bool(row["competing_processes"]) for row in loads)
    return dict(mode=mode,full_requests=len(sequence),summary_sha256=sha256_file(root/"summary.json"),
        request_file_sha256=[sha256_file(root/"trial"/f"request-{i}.json") for i in range(len(sequence))],
        png_sha256=[sha256_file(root/"trial"/f"{i}.png") for i in range(len(sequence))],
        timing=timing,quality_differences=quality,memory=memory(root,summary),
        competing_cpu_load_samples=busy,load_samples=len(loads),load_error=summary.get("load_error"),
        load_sha256=sha256_file(root/"load.jsonl"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--timing",type=Path,required=True)
    p.add_argument("--quality",type=Path,required=True)
    p.add_argument("--raw",type=Path)
    p.add_argument("--baseline",type=Path,default=ROOT/"outputs/qwen512-vae-norm-fused-full-v3-20261018/trial")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    for key in ("library","timing","quality","baseline"):
        setattr(args,key,getattr(args,key).resolve(strict=True))
    if args.output.exists(): p.error("fresh audit output required")
    if args.raw is not None: args.raw=args.raw.resolve(strict=True)
    manifest_path=args.library.parent/"runtime-build/runtime-build-manifest.json"
    manifest=json.loads(manifest_path.read_text())
    for item in manifest["inputs"]["sources"]:
        if sha256_file(ROOT/item["path"])!=item["sha256"]:
            raise ValueError("native source changed after build: "+item["path"])
    rows=[cohort(args.timing,args.library,"timing"),cohort(args.quality,args.library,"quality")]
    baseline_summary=json.loads((args.baseline.parent/"summary.json").read_text())
    if baseline_summary["status"]!="complete" or baseline_summary["library_sha256"]!=\
            "8260a6ec032c193b1727d46b93ce185130293cd8f5bf11f173479d717f7363f3":
        raise ValueError("historical baseline library identity differs")
    defaults=[]
    for current,old in ((2,2),(5,5),(9,10),(14,13)):
        before=json.loads((args.baseline/f"request-{old}.json").read_text())
        after=json.loads((args.timing/"trial"/f"request-{current}.json").read_text())
        normalize=lambda request:{k:v for k,v in request.items() if k!="output"}
        if normalize(before)!=normalize(after): raise ValueError("historical baseline task differs")
        if not np.array_equal(pixels(args.timing/"trial"/f"{current}.png"),pixels(args.baseline/f"{old}.png")):
            raise ValueError("default six-step baseline changed")
        defaults.append(dict(current_index=current,baseline_index=old,rgba_equal=True,
                             baseline_png_sha256=sha256_file(args.baseline/f"{old}.png")))
    raw=None
    if args.raw is not None:
        from qwen_student_sampler_raw import analyze as raw_analyze
        summary=json.loads((args.raw/"summary.json").read_text())
        if summary["status"]!="complete" or summary["asset_sha256"]["library"]!=sha256_file(args.library) or\
                summary["performance_denominator"] is not False:
            raise ValueError("complete parity-only raw capture required")
        snapshots=json.loads((args.raw/"source.json").read_text())
        for relative,digest in summary["source_sha256"].items():
            if sha256_file(ROOT/relative)!=digest or hashlib.sha256(snapshots[relative].encode()).hexdigest()!=digest:
                raise ValueError("raw source/snapshot changed")
        actual=raw_analyze(args.raw/"trial")
        if actual!=summary["analysis"]: raise ValueError("raw input audit could not be reproduced")
        raw=dict(analysis=actual,memory=memory(args.raw,summary),summary_sha256=sha256_file(args.raw/"summary.json"))
        # The dump route must reproduce its own fixed no-prefix quality arm.
        for capture,quality_index in ((0,0),(1,1),(2,16),(3,17)):
            if not np.array_equal(pixels(args.raw/"trial"/f"{capture}.png"),pixels(args.quality/"trial"/f"{quality_index}.png")):
                raise ValueError("raw capture changed corresponding ordinary quality image")
        raw["capture_png_matches_quality_arm"]=True
    result=dict(schema="tc-qwen-student-sampler-audit-v1",status="complete",qualification_passed=False,
        instruction_quality_qualified=False,library_sha256=sha256_file(args.library),
        runtime_build_id=manifest["runtime_build_id"],manifest_sha256=sha256_file(manifest_path),
        native_source_hashes_verified=len(manifest["inputs"]["sources"]),cohorts=rows,
        six_step_default_restoration=defaults,raw=raw,script_sha256=sha256_file(Path(__file__)),
        scope="CPU/source audit only; historical baseline PNG parity; no new inference or physical placement claim")
    write_json(args.output,result)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__": main()
