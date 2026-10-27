#!/usr/bin/env python3
"""Original QKV W8 staging alone: no ANE predictions or full-image claims."""
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
from collect import ROOT,Observer,model_snapshot,sha256_file,write_json
from runtime_ane_common import benchmark_environment
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ARMS=("native-three-cb","batch-three-codes","fused-three-dispatch","fused-one-dispatch")
SOURCES=tuple("tools/profiling/qwen_base/"+x for x in ("qkv_stage_collect.py","qkv_stage_probe.mm","qkv_stage_batch.hpp","collect.py","analyze.py","observer.mm"))+(
    "tools/native/build_qwen_qkv_stage_probe.sh","tools/native/process_tree_sampler.py","tools/native/verify_process_tree_samples.py",
    "tools/validation/runtime_ane_memory.py","tools/validation/runtime_ane_load.py")


def validate(r,n):
    if (type(n) is not int or n not in (32,64) or r.get("schema")!="tc-qkv-w-stage-v1" or r.get("status")!="complete" or
        r.get("complete_request") is not False or r.get("image_export") is not False or r.get("ane_predictions")!=0 or
        r.get("real_source_layers")!=[0,1] or r.get("repeats")!=n or r.get("warmups")!=2 or r.get("extra_owned_weight_bytes")!=0 or
        r.get("scale_cache_capacity")!=2 or r.get("metadata_bytes")!=49152 or r.get("checksum_policy")!="preflight-and-post-loop-only" or
        any(r.get(k) is not True for k in ("source_unchanged","all_stage_bytes_exact"))):raise ValueError("actual original W-only fixed-bank cached-metadata fixture required")
    for key,want in dict(calls=4*n+23,native_calls=n+7,custom_calls=3*n+16,command_buffers=6*n+37,compute_dispatches=16*n+99,
        native_scale_misses=6,native_scale_hits=3*(n+5),rejected=5).items():
        if type(r.get(key)) is not int or r[key]!=want:raise ValueError("actual code/metadata/call/negative-control count differs: "+key)
    samples=r["samples"];expected=[(ARMS[(j+rep+4)%4],rep) for rep in range(-2,n) for j in range(4)]
    if [(s["arm"],s["rep"]) for s in samples]!=expected:raise ValueError("four rotated arms, complete warm/measured samples required")
    for s in samples:
        arm=ARMS.index(s["arm"])
        if (s.get("warmup") is not (s["rep"]<0) or type(s.get("layer")) is not int or s["layer"]!=int(bool(s["rep"]&1)) or
            type(s.get("seconds")) not in (int,float) or not math.isfinite(s["seconds"]) or s["seconds"]<=0 or
            s.get("command_buffers")!=(3 if arm==0 else 1) or s.get("compute_dispatches")!=(6,6,3,1)[arm]):
            raise ValueError("actual completed producer wall, source and dispatch policy differ")
    arrays=r["arrays"]
    if Counter((x["layer"],x["arm"]) for x in arrays)!=Counter({(l,a):1 for l in (0,1) for a in ARMS}):raise ValueError("all eight complete code/scale pairs required")
    for k in ("setup_seconds","prime_seconds"):
        if type(r.get(k)) not in (int,float) or not math.isfinite(r[k]) or r[k]<=0:raise ValueError("actual shader/setup and metadata priming must be separate positive spans")
    medians={a:statistics.median(s["seconds"] for s in samples if s["arm"]==a and not s["warmup"]) for a in ARMS}
    return dict(median_seconds=medians,reduction_vs_native={a:1-v/medians[ARMS[0]] for a,v in medians.items()},
        measured_producers=4*n,calls=4*n+23,command_buffers=6*n+37,compute_dispatches=16*n+99,
        scope="completed real QKV W producers alone; compact metadata preprimed; not pipeline/full40/ANE/physical-overlap performance")


def observer_contract(raw,n):
    rows=analyze.observer_evidence(raw)
    if any(x["name"].startswith("ane.") for x in rows):raise ValueError("W-only screen cannot run or compile an ANE model")
    expected={100+(rep+2)*4+j:(3 if (j+rep+4)%4==0 else 1) for rep in range(-2,n) for j in range(4)}
    expected.update({1+phase*4+arm:(3 if arm==0 else 1) for phase in range(3) for arm in range(4)})
    actual=Counter(x["request"] for x in rows if x["name"]=="metal.command_buffer" and x["request"] in expected)
    if actual!=Counter(expected):raise ValueError("actual per-scope native3 vs batch1 command buffers differ")
    if any(x["request"]>=1 and x["request"] not in expected and x["request"]!=50 for x in rows):raise ValueError("unaccounted hot GPU fixture scope")
    return dict(ane_calls=0,matched_scopes=len(expected),actual_measured_and_qualifier_cbs=sum(actual.values()),
        scope="actual Metal completion/clock events; no ANE invocation, physical bandwidth or full-image speed qualification")


def child(args):
    observer=Observer(args.observer) if args.mode=="timeline" else None
    if observer:observer.lib.tc_qwen_profile_scope(-1)
    C.CDLL(str(args.native_library),mode=C.RTLD_GLOBAL);lib=C.CDLL(str(args.library));scope_type=C.CFUNCTYPE(None,C.c_int)
    scope=scope_type(lambda i:observer.lib.tc_qwen_profile_scope(i)) if observer else scope_type()
    fn=lib.tc_qkv_stage_probe;fn.argtypes=[C.c_char_p,C.c_char_p,C.c_int,scope_type];fn.restype=C.c_int
    try:
        if fn(str(args.model).encode(),str(args.output).encode(),args.repeats,scope):raise RuntimeError("QKV W staging failed; inspect report/stderr")
    finally:
        if observer:write_json(args.output/"observer.json",observer.take("tc_qwen_profile_finish"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("library","native-library","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--mode",choices=("stock","timeline"),default="stock");p.add_argument("--repeats",type=int,choices=(32,64),default=32)
    p.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--observer",type=Path,default=ROOT/"build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib");p.add_argument("--child",action="store_true");args=p.parse_args()
    for key in ("library","native_library","model","observer"):setattr(args,key,getattr(args,key).resolve(strict=True))
    args.output=args.output.resolve()
    if args.child:child(args);return
    manifest=json.loads((args.native_library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    def sources():
        if any(sha256_file(ROOT/e["path"])!=e["sha256"] for e in manifest["inputs"]["sources"]):raise ValueError("linked native source differs")
    sources()
    if shutil.disk_usage(ROOT).free<(2<<30)+(448<<20):raise ValueError("2GiB reserve plus448MiB full code payloads required")
    hashes={x:sha256_file(ROOT/x) for x in SOURCES};assets={k:sha256_file(getattr(args,k)) for k in ("library","native_library","observer")};identity=model_snapshot(args.model)
    args.output.mkdir(parents=True,exist_ok=False);trial=args.output/"trial";trial.mkdir();s=dict(schema="tc-qkv-w-stage-screen-v1",status="running",qualification_passed=False,
        mode=args.mode,repeats=args.repeats,source_sha256=hashes,assets=assets,model_snapshot=identity,runtime_build_id=manifest["runtime_build_id"])
    write_json(args.output/"summary.json",s);write_json(args.output/"source.json",{x:(ROOT/x).read_text() for x in SOURCES});load=LoadObservation(args.output/"load.jsonl")
    try:
        command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child","--library",str(args.library),"--native-library",str(args.native_library),"--mode",args.mode,
            "--repeats",str(args.repeats),"--model",str(args.model),"--observer",str(args.observer),"--output",str(trial)]
        env={k:v for k,v in benchmark_environment().items() if not k.startswith("TURBOCIDER_") and k!="MLX_DISABLE_COMPILE"}
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            s["memory"]=run_sampled(command,repo=ROOT,output=args.output,stem="trial",env=env,stdout=stdout,stderr=stderr,timeout=300,interval_ms=100,max_gap_ms=500,observer=load)
        r=json.loads((trial/"report.json").read_text())
        if (r["compiled_native_id"]!=manifest["runtime_build_id"] or r["loaded_runtime_id"]!=manifest["runtime_build_id"] or
            r["cpp_sha256"]!=hashes["tools/profiling/qwen_base/qkv_stage_probe.mm"] or r["header_sha256"]!=hashes["tools/profiling/qwen_base/qkv_stage_batch.hpp"]):raise ValueError("actual helper/header/native ABI differs")
        s["analysis"]=validate(r,args.repeats)
        if args.mode=="timeline":
            raw=json.loads((trial/"observer.json").read_text())
            if raw["source_sha256"]!=hashes["tools/profiling/qwen_base/observer.mm"]:raise ValueError("compiled observer differs")
            s["observer"]=observer_contract(raw,args.repeats)
        sources();s["status"]="complete"
    except Exception as e:s.update(status="failed",error=str(e))
    try:s["load"]=load.verify()
    except ValueError as e:s["load_error"]=str(e)
    if (model_snapshot(args.model)!=identity or any(sha256_file(ROOT/x)!=d for x,d in hashes.items()) or any(sha256_file(getattr(args,k))!=d for k,d in assets.items())):
        s.update(status="failed",error="source/model/assets changed")
    write_json(args.output/"summary.json",s);print(json.dumps(s,indent=2))
    if s["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
