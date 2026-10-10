#!/usr/bin/env python3
"""Strict same-binary/component local ConvRot/BF16 screen; never qualification."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics

from screen_quantized_against_bf16 import collect,content_identity,workload
from screen_gguf_memory_budgets import positive,sha

PRIVATE={"TURBOCIDER_Z_IMAGE_TRANSFORMER","TURBOCIDER_Z_CONVROT_GPU_RECIPE","TURBOCIDER_Z_CONVROT_CACHE_BYTES","TURBOCIDER_Z_CONVROT_CACHE_RETAIN"}


def identity(report):
    normalized=copy.deepcopy(report)
    normalized["runtime_environment"]={k:v for k,v in report["runtime_environment"].items() if k not in PRIVATE}
    return content_identity(normalized)


def screen(bt,bm,ct,cm,min_samples=4):
    if type(min_samples) is not int or min_samples<2:raise ValueError("insufficient required samples")
    _,bw,b=collect(bt,bm,True,min_samples)
    bi=identity(bt)
    if identity(bm)!=bi or bi["device"] is None:raise ValueError("baseline device/identity missing or changed")
    shared=None;current_workload=None;recipe=None;png=None;times=[];peak=0;gap=0
    for mode,report in (("timing",ct),("memory",cm)):
        if report.get("status")!="completed" or report.get("measurement")!=mode or report.get("route")!="gpu" or not report.get("runs"):
            raise ValueError("requires completed separate pure-GPU ConvRot reports")
        ci=identity(report)
        if shared is not None and ci!=shared:raise ValueError("candidate identity changed")
        shared=ci
        source=sha(report.get("source_sha256"))
        if source!=sha(report.get("dit_source_sha256")):raise ValueError("source binding differs")
        for row in report["runs"]:
            if row.get("status")!=0 or row.get("error") or row.get("measurement")!=mode or type(row.get("warmup")) is not bool:
                raise ValueError("failed or mislabeled sample")
            w=workload(row)
            if current_workload is not None and w!=current_workload:raise ValueError("candidate workload changed")
            current_workload=w;m=row["metrics"];q=m.get("convrot_experiment")
            if not q or q.get("source_profile")!="convrot-legacy-packed-bf16-scale-v1" or q.get("dense_scope")!="none" or q.get("activation_quantization")!="none":
                raise ValueError("unbound ConvRot source/math/representation")
            if m.get("runtime_backend")!="mlx_cpp_metal_convrot_compiled_experimental" or m.get("plan",{}).get("gpu_graph")!="convrot_parameterized_packed_bf16_blocks":
                raise ValueError("wrong actual ConvRot graph")
            if m.get("runtime_precision")!=q.get("execution_recipe") or report.get("gpu_recipe")!="compiled_dense" or not q["execution_recipe"].endswith(":compiled_dense"):
                raise ValueError("unqualified or changed candidate recipe")
            current=(source,q["execution_recipe"],q.get("allocator_cache_limit_bytes"),q.get("allocator_cache_retention"))
            cache=current[2];configured=report.get("controls",{}).get("TURBOCIDER_Z_CONVROT_CACHE_BYTES")
            if cache is not None and (type(cache) is not int or not 0<=cache<=4<<30):raise ValueError("invalid observed cache hint")
            if (configured is None)!=(cache is None) or (configured is not None and str(cache)!=configured):raise ValueError("cache hint control/receipt mismatch")
            retention=current[3];retained=report.get("controls",{}).get("TURBOCIDER_Z_CONVROT_CACHE_RETAIN","0")
            if retained not in ("0","1") or retention not in ("request-cleanup-v1","bounded-global-bins-between-requests-v1") or (retained=="1")!=(retention=="bounded-global-bins-between-requests-v1"):
                raise ValueError("cache retention control/receipt mismatch")
            if recipe is not None and current!=recipe:raise ValueError("source/cache/math changed between samples")
            recipe=current
            h=sha(row.get("png_sha256"))
            if png is not None and h!=png:raise ValueError("candidate PNG changed")
            png=h
            if not row["warmup"] and m.get("prompt_cache_hit") is not True:raise ValueError("warm hit sample did not hit")
            if mode=="timing":
                if row.get("memory_samples"):raise ValueError("timing observer contamination")
                if not row["warmup"]:times.append(positive(row["wall_seconds"],"wall seconds"))
            else:
                samples=row.get("memory_samples")
                if not samples or row.get("sampling_errors") or row.get("vm_deltas",{}).get("Swapouts")!=0:raise ValueError("memory evidence missing/failed/swapped")
                intervals=[b["time"]-a["time"] for a,b in zip(samples,samples[1:])]
                if not intervals or any(not math.isfinite(v) or v<=0 for v in intervals):raise ValueError("invalid sample timestamps")
                actual=max(intervals);recorded=positive(row.get("memory_sample_max_gap_seconds"),"memory gap")
                if actual>.05 or abs(actual-recorded)>1e-6:raise ValueError("inadequate or false memory gap")
                gap=max(gap,actual)
                peak=max(peak,*(positive(s[k],"footprint") for s in samples for k in ("phys_footprint_bytes","lifetime_max_phys_footprint_bytes")))
    if shared!=bi or current_workload!=bw or len(times)<min_samples:raise ValueError("BF16/candidate components/device/workload/tuning differ or samples insufficient")
    median=statistics.median(times);ratio=median/b["median_seconds"];memory_ratio=peak/b["observed_peak_bytes"]
    return {"schema":"tc-convrot-bf16-local-screen-v1","production_qualified":False,"whole_request_memory":"unknown","formal_performance":"not_run","quality":"not_run",
        "identity":bi,"workload":bw,"baseline":b,"candidate":{"source_sha256":recipe[0],"recipe":recipe[1],"allocator_cache_limit_bytes":recipe[2],
            "allocator_cache_retention":recipe[3],
            "samples_seconds":times,"median_seconds":median,"observed_peak_bytes":peak,"maximum_memory_gap_seconds":gap,"png_sha256":png},
        "warm_wall_ratio":ratio,"observed_process_memory_ratio":memory_ratio,"hard_20pct_latency_pass":ratio<=1.2,"preferred_10pct_latency_pass":ratio<=1.1,
        "empirical_speed_memory_target_pass":ratio<=1.2 and memory_ratio<1,"scope":"local matched warm screen; no formal performance/quality/cold/whole-request qualification"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-timing","baseline-memory","candidate-timing","candidate-memory"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists() or a.output.is_symlink():p.error("output already exists")
    paths=[a.baseline_timing,a.baseline_memory,a.candidate_timing,a.candidate_memory]
    result=screen(*(json.loads(path.read_text()) for path in paths))
    result["raw_reports"]=[{"id":path.parent.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
    result["verifier_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
    print("wall ratio",result["warm_wall_ratio"],"memory ratio",result["observed_process_memory_ratio"],"screen target",result["empirical_speed_memory_target_pass"])


if __name__=="__main__":main()
