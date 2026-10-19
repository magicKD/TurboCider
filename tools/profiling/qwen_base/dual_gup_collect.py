#!/usr/bin/env python3
"""Real full FFN block paired original/dual GPU complement on ONE ANE graph."""
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

ARMS=("gpu-original","gpu-staged","private-original","private-fused")
SOURCES=tuple("tools/profiling/qwen_base/"+s for s in ("dual_gup_collect.py","dual_gup.hpp","ffn_projection_probe.mm","collect.py","analyze.py","observer.mm"))+("tools/native/build_qwen_dual_gup_ffn_probe.sh",)


def validate(report,repeats,channels,shared):
    if type(repeats) is not int or repeats not in (8,16) or type(channels) is not int or channels not in (5632,6144) or type(shared) is not bool:
        raise ValueError("explicit bounded dual-GUP experiment required")
    if (report.get("status")!="complete" or report.get("complete_request") is not False or report.get("image_export") is not False or
        report.get("repeats")!=repeats or report.get("warmups")!=2 or report.get("projection")!="descriptor" or report.get("activation")!="sigmoid-complement" or
        report.get("neighbor_share_screen") is not True or report.get("handoff_screen") is not True or report.get("gpu_complement_strategy")!=("dual-shared" if shared else "dual-register") or
        report.get("gpu_packed_weight_bytes")!=0 or report.get("gpu_packed_setup_seconds")!=0 or
        (report.get("completed_full_dit_steps"),report.get("stopped_step"),report.get("stopped_layer"))!=(1,1,0) or
        any(report.get(k) is not True for k in ("source_unchanged","fresh_weights_inputs_outputs_verified","gpu_staged_exact"))):
        raise ValueError("actual original-base partial capture and paired complete-block experiment required")
    d=report.get("dual_gup") or {}
    if ({k:d.get(k) for k in ("shared","bm","bn","first","count","input_dtype","weight_copies")}!=
        dict(shared=shared,bm=32,bn=64,first=0,count=12288-channels,input_dtype="bfloat16",weight_copies=0)):
        raise ValueError("actual original-buffer no-copy typed dual tile differs")
    source=report.get("actual_source_request") or {}
    if ((source.get("model"),source.get("width"),source.get("height"),source.get("steps"),source.get("seed"),source.get("audio"),source.get("lora_count"))!=
        ("qwen-image-2.1",512,512,40,73,False,0) or not ((source.get("operation")=="image.generate" and source.get("reference_count")==0) or
        (source.get("operation")=="image.edit" and source.get("reference_count")==1 and source.get("reference_size")==1024))):raise ValueError("original full-base40 source request required")
    m=report["metrics"];w=m["runtime_weight"];calls=2*repeats+12
    if (m.get("runtime_failed") is not False or m["runtime_calls_session_total"]!=calls or w["hybrid_blocks_session_total"]!=calls or
        w["ane_channels"]!=channels or w["gpu_channels"]!=12288-channels or w["partition_axis"]!="intermediate_channels" or
        w["executor_backend"]!="private_ane" or w["data_path"]!="w8a8_hadamard" or w["fp32_channel_join_enabled"] is not True or
        w["prefetch_enabled"] is not False or w["weight_code_cache"]["enabled"] is not False):raise ValueError("one healthy fixed ANE graph and actual private calls required")
    collect.validate_fixed_program([dict(hybrid=m)]);collect.validate_activation_receipt([dict(hybrid=m)],"sigmoid-complement")
    for name,bound,elements in [(n,.03,4194304) for n in ("complete_block_quality","alternate_quality","other_weight_quality")]+[("dual_hidden_quality_"+n,.005,1024*(12288-channels)) for n in ("a","b")]+[("dual_incremental_quality_"+n,.003,4194304) for n in ("a","b","w")]:
        q=report.get(name) or {}
        if type(q.get("relative_l2")) not in (int,float) or not math.isfinite(q["relative_l2"]) or not 0<=q["relative_l2"]<bound or q.get("elements")!=elements:
            raise ValueError("full real hidden/incremental/block numerical screening bound differs: "+name)
    samples=report["samples"];expected=[(ARMS[(j+r+4)%4],r) for r in range(-2,repeats) for j in range(4)]
    if [(s["arm"],s["rep"]) for s in samples]!=expected:raise ValueError("all four rotated full-block warm/measured samples required")
    for s in samples:
        if (s["warmup"] is not (s["rep"]<0) or s["input_variant"] is not bool(s["rep"]&1) or
            type(s["seconds"]) not in (int,float) or not math.isfinite(s["seconds"]) or s["seconds"]<=0):raise ValueError("positive actual complete-block samples required")
        if s["arm"].startswith("private-"):
            d=s.get("completion_deltas") or {}
            if set(d)!={"prediction_wait_seconds","output_wait_seconds","caller_wait_seconds"} or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in d.values()):raise ValueError("both private arms need actual finite completion differences")
    comp=report["gpu_complement_alone_samples"];expected_comp=[(("original","fused")[(j+r+2)%2],r) for r in range(-2,repeats) for j in range(2)]
    if [(s["arm"],s["rep"]) for s in comp]!=expected_comp:raise ValueError("both complete ready-input GPU-complement controls required")
    for s in comp:
        if s["warmup"] is not (s["rep"]<0) or s["input_variant"] is not bool(s["rep"]&1) or type(s["seconds"]) not in (int,float) or not math.isfinite(s["seconds"]) or s["seconds"]<=0:raise ValueError("positive actual complement times required")
    medians={a:statistics.median(s["seconds"] for s in samples if s["arm"]==a and not s["warmup"]) for a in ARMS}
    return dict(median_seconds=medians,gpu_complement_alone_medians={a:statistics.median(s["seconds"] for s in comp if s["arm"]==a and not s["warmup"]) for a in ("original","fused")},
        incremental_reduction=1-medians["private-fused"]/medians["private-original"],native_workload_predictions=calls,measured_full_blocks=4*repeats,
        scope="matched real full-block original/fused GPU complement with one fixed ANE Program; QKV GPU; not full40 image speed")


def observer_contract(raw,report,repeats):
    events=analyze.observer_evidence(raw);hot=[r for r in events if r["request"]>=1]
    expected={1,2,3,4,5,101,102,104}|{6+i for i,s in enumerate(report["samples"]) if s["arm"].startswith("private-")}
    evaluations=[r for r in hot if r["name"]=="ane.evaluate_api"]
    if (Counter(r["request"] for r in evaluations)!=Counter({n:1 for n in expected}) or len({r["model"] for r in evaluations})!=1 or
        len(evaluations)!=2*repeats+12 or any(r["name"] in ("ane.compile_api","ane.load_api") for r in hot)):
        raise ValueError("one actual fixed model, exact original/fused evaluations and hot compile/load0 required")
    return dict(runtime_model_count=1,actual_workload_evaluations=len(evaluations),hot_compile_load_zero=True,scope="actual Private selectors; no physical INT8/overlap or kernel duration proof")


def child(args):
    os.environ["TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION"]="sigmoid-complement"
    observer=Observer(args.observer) if args.mode=="timeline" else None
    if observer:observer.lib.tc_qwen_profile_scope(-1)
    C.CDLL(str(args.native_library),mode=C.RTLD_GLOBAL);lib=C.CDLL(str(args.library));scope_type=C.CFUNCTYPE(None,C.c_int)
    scope=scope_type(lambda n:observer.lib.tc_qwen_profile_scope(n)) if observer else scope_type()
    fn=lib.tc_ffn_projection_probe;fn.argtypes=[C.c_char_p]*5+[C.c_int,scope_type];fn.restype=C.c_int
    try:
        if fn(str(args.model).encode(),str(args.reference).encode() if args.case=="edit" else b"-",str(args.manifest).encode(),str(args.output).encode(),b"descriptor",args.repeats,scope):raise RuntimeError("dual-GUP real block failed; inspect report/stderr")
    finally:
        if observer:write_json(args.output/"observer.json",observer.take("tc_qwen_profile_finish"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("library","native-library","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--kind",choices=("register","shared"),required=True);p.add_argument("--channels",type=int,choices=(5632,6144),default=6144)
    p.add_argument("--mode",choices=("stock","timeline"),default="stock");p.add_argument("--case",choices=("generate","edit"),default="generate")
    p.add_argument("--repeats",type=int,choices=(8,16),default=8);p.add_argument("--child",action="store_true")
    p.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--reference",type=Path,default=ROOT/"results/qwen21/teapot-blue-edit-native-512.png")
    p.add_argument("--manifest",type=Path,default=ROOT/"outputs/qwen-base-runtime-c1024-h4096-f12288-20261018.json")
    p.add_argument("--observer",type=Path,default=ROOT/"build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib");args=p.parse_args()
    for name in ("library","native_library","model","reference","manifest","observer"):setattr(args,name,getattr(args,name).resolve(strict=True))
    args.output=args.output.resolve()
    if args.child:child(args);return
    native=json.loads((args.native_library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    def native_sources():
        if any(sha256_file(ROOT/s["path"])!=s["sha256"] for s in native["inputs"]["sources"]):raise ValueError("native sources differ")
    native_sources()
    if shutil.disk_usage(ROOT).free<(2<<30)+(384<<20):raise ValueError("2GiB reserve plus384MiB full block arrays required")
    hashes={s:sha256_file(ROOT/s) for s in SOURCES};assets={s:sha256_file(getattr(args,s)) for s in ("library","native_library","manifest","reference","observer")}
    identity=model_snapshot(args.model);args.output.mkdir(parents=True,exist_ok=False);trial=args.output/"trial";trial.mkdir()
    summary=dict(schema="tc-dual-gup-real-block-screen-v1",status="running",qualification_passed=False,kind=args.kind,channels=args.channels,mode=args.mode,case=args.case,
        repeats=args.repeats,source_sha256=hashes,assets=assets,model_snapshot=identity,runtime_build_id=native["runtime_build_id"])
    write_json(args.output/"summary.json",summary);write_json(args.output/"source.json",{s:(ROOT/s).read_text() for s in SOURCES});load=LoadObservation(args.output/"load.jsonl")
    try:
        command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child","--library",str(args.library),"--native-library",str(args.native_library),"--kind",args.kind,
            "--channels",str(args.channels),"--mode",args.mode,"--case",args.case,"--repeats",str(args.repeats),"--model",str(args.model),"--reference",str(args.reference),
            "--manifest",str(args.manifest),"--observer",str(args.observer),"--output",str(trial)]
        env={k:v for k,v in benchmark_environment().items() if not k.startswith("TURBOCIDER_") and k!="MLX_DISABLE_COMPILE"}
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            summary["memory"]=run_sampled(command,repo=ROOT,output=args.output,stem="trial",env=env,stdout=stdout,stderr=stderr,timeout=600,interval_ms=100,max_gap_ms=500,observer=load)
        report=json.loads((trial/"report.json").read_text())
        if (report["cpp_sha256"]!=hashes["tools/profiling/qwen_base/ffn_projection_probe.mm"] or report.get("compiled_native_id")!=native["runtime_build_id"] or
            report["loaded_runtime_id"]!=native["runtime_build_id"] or report.get("dual_gup",{}).get("header_sha256")!=hashes["tools/profiling/qwen_base/dual_gup.hpp"]):raise ValueError("actual dual helper/header/native ABI identity differs")
        summary["analysis"]=validate(report,args.repeats,args.channels,args.kind=="shared")
        summary["arrays_sha256"]={name:sha256_file(trial/name) for name in ("arrays.safetensors","gpu-complement-arrays.safetensors")}
        if args.mode=="timeline":
            raw=json.loads((trial/"observer.json").read_text())
            if raw["source_sha256"]!=hashes["tools/profiling/qwen_base/observer.mm"]:raise ValueError("actual observer source differs")
            summary["observer"]=observer_contract(raw,report,args.repeats)
        native_sources();summary["status"]="complete"
    except Exception as error:summary.update(status="failed",error=str(error))
    try:summary["load"]=load.verify()
    except ValueError as error:summary["load_error"]=str(error)
    if (model_snapshot(args.model)!=identity or any(sha256_file(ROOT/s)!=d for s,d in hashes.items()) or
        any(sha256_file(getattr(args,s))!=d for s,d in assets.items())):summary.update(status="failed",error="bound source/model/assets changed")
    write_json(args.output/"summary.json",summary);print(json.dumps(summary,indent=2))
    if summary["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
