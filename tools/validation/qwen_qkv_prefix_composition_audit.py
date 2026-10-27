#!/usr/bin/env python3
"""CPU PNG/source/request/denominator audit; cached image latents not captured."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image

from flux9_bf16_framework_screen import write_json
import qwen_qkv_prefix_composition_screen as screen
from runtime_ane_common import sha256_file

ROOT=Path(__file__).resolve().parents[2]


def read_window(root,library_sha,runtime):
    summary=json.loads((root/"summary.json").read_text());report=json.loads((root/"trial/report.json").read_text())
    if summary["status"]!="complete" or summary["asset_sha256"]["library"]!=library_sha or report["loaded_runtime_id"]!=runtime:
        raise ValueError("complete same-library window required")
    source=json.loads((root/"source.json").read_text())
    for name,digest in summary["source_sha256"].items():
        if hashlib.sha256(source[name].encode()).hexdigest()!=digest or sha256_file(ROOT/name)!=digest:raise ValueError("bound source differs")
    sequence=screen.points(summary["mode"],summary["steps"],summary["arm"])
    if sequence!=summary["frozen_sequence"]:raise ValueError("frozen workload changed")
    analysis=screen.analyze(root/"trial",report,sequence,summary["mode"])
    if analysis!=summary["analysis"]:raise ValueError("full PNG/denominator recheck differs")
    for key in ("evidence","report"):
        if sha256_file(root/summary["memory"][key])!=summary["memory"][key+"_sha256"]:raise ValueError("memory evidence differs")
    memory=summary["memory"]["verified"]
    if not memory["complete"] or memory["command_exit_code"]!=0:raise ValueError("incomplete process window")
    requests=[]
    for i in range(len(sequence)):
        request=json.loads((root/f"trial/request-{i}.json").read_text());request.pop("output",None);requests.append(request)
    return summary,report,sequence,screen.images(root/"trial",len(sequence)),requests


def key(point):return point["case"],point["prompt"],point["steps"],point["arm"],point["prefix_hit"],point["seed"]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--library",type=Path,required=True)
    p.add_argument("--native-source-archive",type=Path,required=True);p.add_argument("--restoration",type=Path,required=True)
    p.add_argument("--quality",type=Path,required=True);p.add_argument("--cohort",type=Path,action="append",required=True)
    p.add_argument("--previous-quality",type=Path,required=True);p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error("fresh audit output required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text());sources=manifest["inputs"]["sources"]
    library_sha=sha256_file(args.library);runtime=manifest["runtime_build_id"]
    with tarfile.open(args.native_source_archive,"r:gz") as archive:
        members={member.name:member for member in archive.getmembers()}
        if set(members)!={source["path"] for source in sources}:raise ValueError("complete native archive required")
        for source in sources:
            stream=archive.extractfile(members[source["path"]])
            if stream is None or hashlib.sha256(stream.read()).hexdigest()!=source["sha256"] or sha256_file(ROOT/source["path"])!=source["sha256"]:
                raise ValueError("archive/current source differs from linked build")
    restoration=read_window(args.restoration,library_sha,runtime);quality=read_window(args.quality,library_sha,runtime)
    if restoration[0]["mode"]!="restoration" or quality[0]["mode"]!="quality" or not quality[0]["analysis"]["prefix_pixel_gate_passed"]:
        raise ValueError("separate complete restoration and qualified local-pixel quality matrix required")
    lookup={key(point):i for i,point in enumerate(quality[2])}
    ownership=[]
    for start in range(0,len(quality[2]),10):
        active=[row["metrics"]["memory"]["mlx_active_bytes"] for row in quality[1]["runs"][start:start+10]]
        if active[0]!=active[1] or active[2]!=active[3] or any(value!=active[4] for value in active[5:]):
            raise ValueError("direct QKV introduced extra active ownership within a prefix tier")
        ownership.append(dict(steps=quality[2][start]["steps"],case_id=quality[2][start]["case_id"],mlx_active_bytes=active,
            prefix_active_increment=active[8]-active[1],compact_kv_bytes=quality[1]["runs"][start+8]["metrics"]["qwen_corrected_qkv"]["prefix_snapshot_bytes"],
            within_prefix_tier_direct_extra_active_bytes=0))
    previous=json.loads((args.previous_quality/"summary.json").read_text())
    if previous["status"]!="complete" or not previous["analysis"]["all_within_tier_latents_rgba_exact"]:
        raise ValueError("previous sealed four-arm sampler quality control required")
    old_lookup={(point["case"],point["prompt"],point["steps"],point["on"],point["seed"]):i for i,point in enumerate(previous["frozen_sequence"])}
    old_pairs=0
    def check_previous(window):
        nonlocal old_pairs
        for i,point in enumerate(window[2]):
            if screen.ARMS[point["arm"]][1]:continue
            j=old_lookup[point["case"],point["prompt"],point["steps"],screen.ARMS[point["arm"]][0],point["seed"]]
            with Image.open(args.previous_quality/f"trial/{j}.png") as image:old=np.asarray(image.convert("RGBA"))
            request=json.loads((args.previous_quality/f"trial/request-{j}.json").read_text())
            for name in ("output","dump_tensors"):request.pop(name,None)
            if request!=window[4][i] or not np.array_equal(old,window[3][i]):raise ValueError("uncached request or PNG regressed across builds")
            old_pairs+=1
    def check_quality(window):
        for i,point in enumerate(window[2]):
            if point["case"]!="edit":continue
            j=lookup[key(point)]
            if window[4][i]!=quality[4][j] or not np.array_equal(window[3][i],quality[3][j]):raise ValueError("matrix/restoration/timing request or PNG differs")
    check_quality(restoration);check_previous(restoration);check_previous(quality)
    cohorts={};observations=[];windows=[restoration,quality]
    for root in args.cohort:
        window=read_window(root,library_sha,runtime);s=window[0];k=s["steps"],s["arm"]
        if s["mode"]!="timing" or k in cohorts:raise ValueError("one fixed process per tier/arm required")
        check_quality(window);check_previous(window);cohorts[k]=s["analysis"];windows.append(window)
        loads=[json.loads(line) for line in (root/"load.jsonl").read_text().splitlines()]
        observations.append(dict(cohort=root.name,steps=k[0],arm=k[1],analysis=s["analysis"],memory=s["memory"]["verified"],
            load_error=s.get("load_error"),competing_load_samples=sum(bool(row["competing_processes"]) for row in loads),summary_sha256=sha256_file(root/"summary.json")))
    if set(cohorts)!={(steps,arm) for steps in (5,6) for arm in screen.ARMS}:raise ValueError("both complete four-arm sampler cohorts required")
    timings=[]
    for steps in (6,5):
        m={arm:cohorts[steps,arm]["median_seconds"] for arm in screen.ARMS}
        timings.append(dict(steps=steps,medians=m,kernel_prefix_off_reduction_percent=100*(1-m["direct"]/m["original"]),
            prefix_original_kernel_reduction_percent=100*(1-m["prefix"]/m["original"]),
            kernel_over_strong_prefix_reduction_percent=100*(1-m["combined"]/m["prefix"]),
            combined_vs_original_reduction_percent=100*(1-m["combined"]/m["original"])))
    runs=[row for window in windows for row in window[1]["runs"]];direct=[row["metrics"]["qwen_corrected_qkv"] for row in runs if row["metrics"].get("qwen_corrected_qkv")]
    result=dict(schema="tc-qwen-direct-qkv-prefix-composition-audit-v1",status="complete",qualification_passed=False,library_sha256=library_sha,runtime_build_id=runtime,
        native_source_hashes_verified=len(sources),native_source_archive_sha256=sha256_file(args.native_source_archive),
        restoration_analysis=restoration[0]["analysis"],quality_analysis=quality[0]["analysis"],active_ownership=ownership,timings=timings,observations=observations,
        previous_quality_sha256=sha256_file(args.previous_quality/"summary.json"),uncached_previous_exact_pairs=old_pairs,
        completed_full_requests=len(runs),direct_requests=len(direct),completed_direct_decode_blocks=sum(q["completed_decode_blocks"] for q in direct),
        actual_denoise_steps=sum(row["metrics"]["actual_denoise_steps"] for row in runs),script_sha256=sha256_file(Path(__file__)),
        scope="full PNG/request/source/denominator audit; native fixture covers cached tensors at actual QKV shape, not full-image cached latent captures; no multiplied historical gains or GPU exclusivity")
    write_json(args.output,result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
