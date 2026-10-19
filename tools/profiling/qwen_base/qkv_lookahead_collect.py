#!/usr/bin/env python3
"""Bounded synthetic real-ANE lookahead state probe; no checkpoint or image."""
import argparse
from collections import Counter
import ctypes as C
import json
from pathlib import Path
import shutil
import sys

import analyze
from collect import ROOT, Observer, sha256_file, write_json
from qkv_collect import environment
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

SOURCES=("tools/profiling/qwen_base/qkv_lookahead_collect.py",
         "tools/profiling/qwen_base/qkv_lookahead_probe.mm",
         "tools/native/build_qwen_qkv_lookahead_probe.sh",
         "tools/profiling/qwen_base/collect.py", "tools/profiling/qwen_base/qkv_collect.py",
         "tools/profiling/qwen_base/analyze.py", "tools/profiling/qwen_base/observer.mm")
COUNTS=dict(submissions=6,producer_submissions=18,claims=3,weight_completions=2,
            completed_hits=2,discards=3,retired_claims=1,failures=0)


def validate(report):
    if (report.get("status")!="complete" or report.get("synthetic_only") is not True or
        report.get("complete_request") is not False or report.get("image_export") is not False or
        report.get("predictions")!=5 or report.get("validation_calls")!=2 or
        any(report.get(k) is not True for k in ("no_fallback","source_unchanged","old_output_unchanged")) or
        not isinstance(report.get("same_program_key"),str) or not report["same_program_key"]):
        raise ValueError("actual bounded state transitions, unchanged source/old output and no fallback required")
    s=report.get("prefetch") or {}
    if any(type(s.get(k)) is not int or s[k]!=v for k,v in COUNTS.items()):
        raise ValueError("actual producer/claim/mismatch/generation/drain/disable accounting differs")
    if (type(s.get("attempts")) is not int or s["attempts"]!=6 or s.get("requested") is not True or
        s.get("enabled") is not False or s.get("future_pending") is not False):
        raise ValueError("exact attempts and disabled drained state required")


def observer_contract(raw):
    rows=analyze.observer_evidence(raw)
    work=[r for r in rows if r["request"]>=1]
    evaluations=[r for r in work if r["name"]=="ane.evaluate_api"]
    if (len({r["model"] for r in evaluations})!=1 or
        Counter(r["request"] for r in evaluations)!=Counter({i:1 for i in range(1,6)}) or
        any(r["name"] in ("ane.compile_api","ane.load_api") for r in work)):
        raise ValueError("one model/five actual evaluations and no hot compile/load required")
    return dict(runtime_model_count=1,evaluate_calls=5,hot_compile_calls=0,hot_load_calls=0,
                scope="actual Private selectors, not physical INT8 or hardware overlap")


def child(args):
    observer=Observer(args.observer) if args.mode=="timeline" else None
    if observer:observer.lib.tc_qwen_profile_scope(-1)  # synthetic preparation is not the single blit calibration
    C.CDLL(str(args.native_library),mode=C.RTLD_GLOBAL)
    lib=C.CDLL(str(args.library));scope_type=C.CFUNCTYPE(None,C.c_int)
    scope=scope_type(lambda i:observer.lib.tc_qwen_profile_scope(i)) if observer else scope_type()
    fn=lib.tc_qkv_lookahead_probe;fn.argtypes=[C.c_char_p,scope_type];fn.restype=C.c_int
    try:
        if fn(str(args.output).encode(),scope):raise RuntimeError("real QKV state probe failed; inspect report/stderr")
    finally:
        if observer:write_json(args.output/"observer.json",observer.take("tc_qwen_profile_finish"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("library","native-library","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--observer",type=Path,default=ROOT/"build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib")
    p.add_argument("--mode",choices=("stock","timeline"),default="timeline");p.add_argument("--child",action="store_true")
    args=p.parse_args()
    for name in ("library","native_library","observer"):setattr(args,name,getattr(args,name).resolve(strict=True))
    args.output=args.output.resolve()
    if args.child:child(args);return
    manifest=json.loads((args.native_library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    def native_sources():
        if any(sha256_file(ROOT/e["path"])!=e["sha256"] for e in manifest["inputs"]["sources"]):
            raise ValueError("native sources changed")
    native_sources()
    parent=next(v for v in args.output.parents if v.exists())
    if shutil.disk_usage(parent).free<(2<<30)+(256<<20):raise ValueError("2GiB reserve plus256MiB state output allowance required")
    sources={s:sha256_file(ROOT/s) for s in SOURCES}
    assets={s:sha256_file(getattr(args,s)) for s in ("library","native_library","observer")}
    args.output.mkdir(parents=True,exist_ok=False);trial=args.output/"trial";trial.mkdir()
    summary=dict(schema="tc-qkv-lookahead-state-screen-v1",status="running",qualification_passed=False,
                 mode=args.mode,source_sha256=sources,assets=assets,runtime_build_id=manifest["runtime_build_id"],environment=environment(),
                 scope="synthetic diagonal original BF16 matrices; actual fixed512 Private QKV state only; no complete request/image/performance claim")
    write_json(args.output/"source.json",{s:(ROOT/s).read_text() for s in SOURCES});write_json(args.output/"summary.json",summary)
    load=LoadObservation(args.output/"load.jsonl")
    try:
        command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child","--library",str(args.library),
                 "--native-library",str(args.native_library),"--observer",str(args.observer),"--mode",args.mode,"--output",str(trial)]
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            summary["memory"]=run_sampled(command,repo=ROOT,output=args.output,stem="trial",env=environment(),stdout=stdout,
                                         stderr=stderr,timeout=300,interval_ms=100,max_gap_ms=500,observer=load)
        report=json.loads((trial/"report.json").read_text());validate(report)
        if (report["loaded_runtime_id"]!=manifest["runtime_build_id"] or report["compiled_native_id"]!=manifest["runtime_build_id"] or
            report["cpp_sha256"]!=sources["tools/profiling/qwen_base/qkv_lookahead_probe.mm"]):raise ValueError("helper/native/source ABI identity differs")
        if args.mode=="timeline":
            raw=json.loads((trial/"observer.json").read_text())
            if raw["source_sha256"]!=sources["tools/profiling/qwen_base/observer.mm"]:raise ValueError("observer compiled source differs")
            summary["observer"]=observer_contract(raw)
        summary["arrays_sha256"]=sha256_file(trial/"arrays.safetensors");native_sources();summary["status"]="complete"
    except Exception as error:summary.update(status="failed",error=str(error))
    try:summary["load"]=load.verify()
    except ValueError as error:summary["load_error"]=str(error)
    if any(sha256_file(ROOT/k)!=v for k,v in sources.items()) or any(sha256_file(getattr(args,k))!=v for k,v in assets.items()):
        summary.update(status="failed",error="bound source/assets changed")
    write_json(args.output/"summary.json",summary);print(json.dumps(summary,indent=2))
    if summary["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
