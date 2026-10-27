#!/usr/bin/env python3
"""Six-arm FFN->dependent-QKV submission-frontier fixture, NOT full40."""
import argparse
from collections import Counter
import ctypes as C
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys

import analyze
import collect
from collect import ROOT,Observer,model_snapshot,sha256_file,write_json
from runtime_ane_common import benchmark_environment
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ARMS=("original-post","dual-post","original-pre","dual-pre","original-hidden","dual-hidden")
SPANS=("ffn_caller_seconds","ffn_async_wait_seconds","qkv_weight_wait_seconds","qkv_predict_restore_join_seconds")
SOURCES=tuple("tools/profiling/qwen_base/"+n for n in ("frontier_collect.py","frontier_probe.mm","collect.py","analyze.py","observer.mm"))+(
    "tools/native/build_qwen_frontier_probe.sh","tools/validation/runtime_ane_memory.py","tools/validation/runtime_ane_load.py",
    "tools/native/process_tree_sampler.py","tools/native/verify_process_tree_samples.py")


def scopes(repeats):
    return list(range(1,25))+list(range(100,100+6*(repeats+2)))


def validate(r,repeats):
    if (type(repeats) is not int or repeats not in (16,32) or r.get("schema")!="tc-qwen-frontier-fixture-v2" or r.get("status")!="complete" or
        any(r.get(k) is not False for k in ("complete_request","image_export","natural_two_block_forward")) or
        any(r.get(k) is not True for k in ("source_unchanged","all_outputs_bits_exact","single_bank")) or
        r.get("next_norm")!="reused-layer0-modulation-fixture" or r.get("current_block_qkv")!="gpu" or
        (r.get("completed_full_dit_steps"),r.get("stopped_step"),r.get("stopped_layer"))!=(1,1,0) or
        r.get("repeats")!=repeats or r.get("warmups")!=2 or r.get("extra_owned_weight_bytes")!=0 or
        r.get("timing_checksum_policy")!="post-loop-last-per-arm" or r.get("last_owned_output_bytes")!=201326592):
        raise ValueError("actual bounded real block + synthetic dependent edge, not natural two-block/full40 qualification")
    source=r.get("actual_source_request") or {}
    if ((source.get("model"),source.get("width"),source.get("height"),source.get("steps"),source.get("seed"),source.get("audio"),source.get("lora_count"))!=
        ("qwen-image-2.1",512,512,40,73,False,0) or not ((source.get("operation")=="image.generate" and source.get("reference_count")==0) or
        (source.get("operation")=="image.edit" and source.get("reference_count")==1 and source.get("reference_size")==1024))):
        raise ValueError("original base40 generation or standard-reference edit source required")
    calls=6*(repeats+6);m=r["metrics"];w=m["runtime_weight"];q=r["qkv"];p=r["prefetch"]
    if (m.get("runtime_failed") is not False or m.get("runtime_calls_session_total")!=calls or w.get("hybrid_blocks_session_total")!=calls or
        w.get("ane_channels")!=6144 or w.get("gpu_channels")!=6144 or w.get("fp32_channel_join_enabled") is not True or
        w.get("executor_backend")!="private_ane" or w.get("data_path")!="w8a8_hadamard" or w.get("prefetch_enabled") is not False or
        w.get("weight_code_cache",{}).get("enabled") is not False):raise ValueError("one healthy fixed FFN, unchanged W8/F32/6144 recipe required")
    collect.validate_fixed_program([dict(hybrid=m)]);collect.validate_activation_receipt([dict(hybrid=m)],"sigmoid-complement")
    if (any(type(q.get(k)) is not int or q[k]!=calls for k in ("predictions","weight_stages","completed_blocks")) or
        q.get("validation_calls")!=1 or q.get("validation_passed") is not True or q.get("failed") is not False or q.get("fallback_blocks")!=0 or
        not isinstance(q.get("program_key"),str) or len(q["program_key"])!=64 or q.get("slot_bytes")!=65814528 or q.get("estimated_bytes")!=301989888):
        raise ValueError("one fixed QKV graph with all successful workload calls required")
    expected=dict.fromkeys(("attempts","submissions","claims","weight_completions","completed_hits"),calls)
    expected.update(producer_submissions=3*calls,discards=0,retired_claims=0,failures=0)
    if (any(type(p.get(k)) is not int or p[k]!=v for k,v in expected.items()) or p.get("requested") is not True or
        p.get("enabled") is not True or p.get("future_pending") is not False):raise ValueError("complete single-bank one-hop producers/claims/completions required")
    samples=r["samples"];order=[(ARMS[(j+rep+6)%6],rep) for rep in range(-2,repeats) for j in range(6)]
    if [(s["arm"],s["rep"]) for s in samples]!=order:raise ValueError("six rotated arms with full warm/measured coverage required")
    for s in samples:
        if (s.get("warmup") is not (s["rep"]<0) or s.get("input_variant") is not bool(s["rep"]&1) or
            type(s.get("seconds")) not in (int,float) or not math.isfinite(s["seconds"]) or s["seconds"]<=0 or
            any(type(s.get(k)) not in (int,float) or not math.isfinite(s[k]) or s[k]<0 for k in SPANS)):
            raise ValueError("actual positive completed consumer wall and finite diagnostic deltas required")
    medians={a:statistics.median(s["seconds"] for s in samples if s["arm"]==a and not s["warmup"]) for a in ARMS}
    return dict(median_seconds=medians,reduction_vs_original_post={a:1-v/medians["original-post"] for a,v in medians.items()},
        median_spans={a:{k:statistics.median(s[k] for s in samples if s["arm"]==a and not s["warmup"]) for k in SPANS} for a in ARMS},
        native_calls_per_operator=calls,measured_consumers=6*repeats,
        scope="real complete layer0 block plus dependent projection fixture with reused layer0 modulation; not natural two-block or complete40 speed")


def observer_contract(raw,repeats):
    rows=analyze.observer_evidence(raw);hot=[x for x in rows if x["request"]>=1]
    calls=[x for x in hot if x["name"]=="ane.evaluate_api"];models={x["model"] for x in calls}
    if len(models)!=2 or Counter((x["request"],x["model"]) for x in calls)!=Counter({(n,m):1 for n in scopes(repeats) for m in models}):
        raise ValueError("two actual fixed model identities each predict once per completed fixture")
    if any(x["name"] in ("ane.compile_api","ane.load_api") for x in hot):raise ValueError("hot frontier changes compiled/loaded an ANE model")
    return dict(runtime_models=2,actual_workload_evaluations=len(calls),hot_compile_load_zero=True,
        scope="observed Private selectors and completed Metal records; no physical INT8/overlap qualification")


def child(args):
    observer=Observer(args.observer) if args.mode=="timeline" else None
    if observer:observer.lib.tc_qwen_profile_scope(-1)
    C.CDLL(str(args.native_library),mode=C.RTLD_GLOBAL);lib=C.CDLL(str(args.library));scope_type=C.CFUNCTYPE(None,C.c_int)
    scope=scope_type(lambda n:observer.lib.tc_qwen_profile_scope(n)) if observer else scope_type()
    fn=lib.tc_qwen_frontier_probe;fn.argtypes=[C.c_char_p]*4+[C.c_int,scope_type];fn.restype=C.c_int
    try:
        if fn(str(args.model).encode(),str(args.reference).encode() if args.case=="edit" else b"-",str(args.manifest).encode(),str(args.output).encode(),args.repeats,scope):
            raise RuntimeError("frontier fixture failed; inspect native report and stderr")
    finally:
        if observer:write_json(args.output/"observer.json",observer.take("tc_qwen_profile_finish"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("library","native-library","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--case",choices=("generate","edit"),default="generate");p.add_argument("--mode",choices=("stock","timeline"),default="stock")
    p.add_argument("--repeats",choices=(16,32),type=int,default=16);p.add_argument("--child",action="store_true")
    p.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--reference",type=Path,default=ROOT/"results/qwen21/teapot-blue-edit-native-512.png")
    p.add_argument("--manifest",type=Path,default=ROOT/"outputs/qwen-base-runtime-c1024-h4096-f12288-20261018.json")
    p.add_argument("--observer",type=Path,default=ROOT/"build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib");args=p.parse_args()
    for name in ("library","native_library","model","reference","manifest","observer"):setattr(args,name,getattr(args,name).resolve(strict=True))
    args.output=args.output.resolve()
    if args.child:child(args);return
    manifest=json.loads((args.native_library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    def native_sources():
        if any(sha256_file(ROOT/e["path"])!=e["sha256"] for e in manifest["inputs"]["sources"]):raise ValueError("native source differs from linked C++ ABI")
    native_sources()
    if shutil.disk_usage(ROOT).free<(2<<30)+(640<<20):raise ValueError("2GiB reserve plus640MiB complete arrays required")
    hashes={name:sha256_file(ROOT/name) for name in SOURCES};assets={name:sha256_file(getattr(args,name)) for name in ("library","native_library","manifest","reference","observer")}
    identity=model_snapshot(args.model);args.output.mkdir(parents=True,exist_ok=False);trial=args.output/"trial";trial.mkdir()
    summary=dict(schema="tc-qwen-frontier-screen-v1",status="running",qualification_passed=False,case=args.case,mode=args.mode,repeats=args.repeats,
        source_sha256=hashes,assets=assets,model_snapshot=identity,runtime_build_id=manifest["runtime_build_id"])
    write_json(args.output/"summary.json",summary);write_json(args.output/"source.json",{name:(ROOT/name).read_text() for name in SOURCES});load=LoadObservation(args.output/"load.jsonl")
    try:
        command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child","--library",str(args.library),"--native-library",str(args.native_library),
            "--case",args.case,"--mode",args.mode,"--repeats",str(args.repeats),"--model",str(args.model),"--reference",str(args.reference),"--manifest",str(args.manifest),
            "--observer",str(args.observer),"--output",str(trial)]
        env={k:v for k,v in benchmark_environment().items() if not k.startswith("TURBOCIDER_") and k!="MLX_DISABLE_COMPILE"}
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            summary["memory"]=run_sampled(command,repo=ROOT,output=args.output,stem="trial",env=env,stdout=stdout,stderr=stderr,timeout=600,interval_ms=100,max_gap_ms=500,observer=load)
        report=json.loads((trial/"report.json").read_text())
        if (report["cpp_sha256"]!=hashes["tools/profiling/qwen_base/frontier_probe.mm"] or
            report["compiled_native_id"]!=manifest["runtime_build_id"] or report["loaded_runtime_id"]!=manifest["runtime_build_id"]):
            raise ValueError("compiled frontier helper/native ABI identity differs")
        summary["analysis"]=validate(report,args.repeats);summary["arrays_sha256"]=sha256_file(trial/"arrays.safetensors")
        if args.mode=="timeline":
            raw=json.loads((trial/"observer.json").read_text())
            if raw["source_sha256"]!=hashes["tools/profiling/qwen_base/observer.mm"]:raise ValueError("compiled observer source differs")
            summary["observer"]=observer_contract(raw,args.repeats)
        native_sources();summary["status"]="complete"
    except Exception as error:summary.update(status="failed",error=str(error))
    try:summary["load"]=load.verify()
    except ValueError as error:summary["load_error"]=str(error)
    if (model_snapshot(args.model)!=identity or any(sha256_file(ROOT/n)!=d for n,d in hashes.items()) or
        any(sha256_file(getattr(args,n))!=d for n,d in assets.items())):summary.update(status="failed",error="source/model/assets changed")
    write_json(args.output/"summary.json",summary);print(json.dumps(summary,indent=2))
    if summary["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
