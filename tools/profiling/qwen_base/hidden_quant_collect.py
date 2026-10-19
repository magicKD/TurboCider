#!/usr/bin/env python3
"""Fixed hidden rotation/normalization/quantization screen; no model/image."""
import argparse
from collections import Counter
import ctypes as C
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

import analyze
from collect import ROOT,Observer,sha256_file,write_json
from qkv_collect import environment
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

POLICIES=("divide","reciprocal-divide","reciprocal-inverse")
PATTERNS=("random-a","zero","tiny","wide","overflow","random-a-restored")
GROUPS=tuple((m,c,r) for m,c in ((32,512),(1024,6144)) for r in (False,True))
SOURCES=tuple("tools/profiling/qwen_base/"+x for x in ("hidden_quant_collect.py","hidden_quant_mil.hpp","hidden_quant_probe.mm",
    "rotation_mil.hpp","collect.py","qkv_collect.py","analyze.py","observer.mm"))+("tools/native/build_qwen_hidden_quant_probe.sh",)


def validate(report,repeats):
    if type(repeats) is not int or repeats not in (12,24):raise ValueError("bounded hidden quant repeat count required")
    if (report.get("status")!="complete" or report.get("synthetic_only") is not True or report.get("model_weights_loaded") is not False or
        report.get("complete_request") is not False or report.get("image_export") is not False or report.get("repeats")!=repeats or report.get("warmups")!=2 or
        report.get("dtype")!="float16-le" or report.get("layout")!="channels_tokens_packed_scale" or
        any(report.get(k) is not True for k in ("source_unchanged","same_program_restoration_exact","overflow_negative_checked"))):
        raise ValueError("complete typed hidden normalization/quantization and source/negative controls required")
    programs=report.get("programs",[]);expected=[(g*3+p,m,c,r,policy) for g,(m,c,r) in enumerate(GROUPS) for p,policy in enumerate(POLICIES)]
    if [tuple(p[k] for k in ("ordinal","rows","channels","rotate","policy")) for p in programs]!=expected:
        raise ValueError("all twelve bounded policy/shape/rotation attempts required")
    for p in programs:
        if p.get("status") not in ("ready","compile-rejected") or (p["status"]=="compile-rejected" and (not p.get("error") or p["ordinal"]%3==0)):
            raise ValueError("successful original controls and explicit optional compiler rejections required")
    ready=[p for p in programs if p["status"]=="ready"]
    if len({p["key"] for p in ready})!=len(ready):raise ValueError("unique independent diagnostic model keys required")
    expected_arrays=[(g*3+p,(g*3+p)*1000+i+1,policy,r,pattern,m,c) for g,(m,c,r) in enumerate(GROUPS)
                     for i,pattern in enumerate(PATTERNS) for p,policy in enumerate(POLICIES) if programs[g*3+p]["status"]=="ready"]
    arrays=report.get("arrays",[])
    if [tuple(x[k] for k in ("ordinal","scope","policy","rotate","pattern","rows","channels")) for x in arrays]!=expected_arrays:
        raise ValueError("complete actual zero/tiny/wide/finite-overflow/restoration arrays required")
    for row in arrays:
        rejected=row["rotate"] and row["pattern"]=="overflow"
        if (row.get("expected_rejection") is not rejected or type(row.get("validation_flags")) is not int or
            row["validation_flags"]<0 or (row["validation_flags"]!=0)!=rejected):raise ValueError("actual finite-output rejection contract differs")
    expected_times=[]
    for g in range(4):
        selected=[p for p in range(3) if programs[g*3+p]["status"]=="ready"]
        for rep in range(-2,repeats):
            for j in range(len(selected)):
                p=selected[(j+rep+len(selected))%len(selected)];expected_times.append((g*3+p,rep,POLICIES[p]))
    samples=report.get("samples",[])
    if [(s["ordinal"],s["rep"],s["policy"]) for s in samples]!=expected_times:raise ValueError("exact rotated warm/measured complete operator sequence required")
    for row in samples:
        p=programs[row["ordinal"]]
        if (row.get("warmup") is not (row["rep"]<0) or row.get("input_variant") is not bool(row["rep"]&1) or
            row.get("scope")!=row["ordinal"]*1000+102+row["rep"] or row.get("validation_flags")!=0 or
            type(row.get("seconds")) not in (int,float) or not math.isfinite(row["seconds"]) or row["seconds"]<=0 or
            (row["rows"],row["channels"],row["rotate"])!=(p["rows"],p["channels"],p["rotate"])):
            raise ValueError("actual finite complete timing, warmup isolation and fixed shape required")
    return dict(median_seconds={f"m{m}-c{c}-{'rotate' if r else 'raw'}":{p["policy"]:statistics.median(s["seconds"] for s in samples if s["ordinal"]==p["ordinal"] and not s["warmup"])
        for p in ready if (p["rows"],p["channels"],p["rotate"])==(m,c,r)} for m,c,r in GROUPS},
        measured_operator_calls=len(ready)*repeats,total_operator_calls=len(ready)*(repeats+8),
        compile_rejected=[p for p in programs if p["status"]=="compile-rejected"],
        scope="synthetic fixed FP16 hidden->packed(q/128,hscale), physical-complete/finite validation; not full FFN/image or hardware kernel time")


def observer_contract(raw,report,repeats):
    if Counter(e["name"] for e in raw["events"])!=Counter(raw["counts"]):raise ValueError("actual raw observer event counts differ")
    rejected={p["ordinal"]*1000 for p in report["programs"] if p["status"]=="compile-rejected"}
    failed=[e for e in raw["events"] if e.get("ok") is not True]
    if (Counter(e["request"] for e in failed)!=Counter({n:1 for n in rejected}) or
        any(e.get("ok") is not False or e["name"]!="ane.compile_api" or type(e.get("begin_ns")) is not int or
            type(e.get("end_ns")) is not int or not 0<e["begin_ns"]<=e["end_ns"] for e in failed)):
        raise ValueError("only exactly declared setup compiler failures may fail")
    good={**raw,"events":[e for e in raw["events"] if e.get("ok") is True]};good["counts"]=dict(Counter(e["name"] for e in good["events"]))
    events=analyze.observer_evidence(good);ids=set();receipts=[]
    for p in report["programs"]:
        if p["status"]!="ready":continue
        n=p["ordinal"];scopes={n*1000+i+1 for i in range(6)}|{n*1000+102+rep for rep in range(-2,repeats)}
        rows=[e for e in events if e["request"] in scopes];calls=[e for e in rows if e["name"]=="ane.evaluate_api"]
        models={e["model"] for e in calls}
        if (len(models)!=1 or Counter(e["request"] for e in calls)!=Counter({i:1 for i in scopes}) or
            any(e["name"] in ("ane.compile_api","ane.load_api") for e in rows)):raise ValueError("one actual fixed model/exact predictions and no hot compile/load required")
        ids|=models;receipts.append(dict(ordinal=n,model_ids=sorted(models),evaluations=len(calls),hot_compile_load_zero=True))
    if len(ids)!=len(receipts) or sum(e["name"]=="ane.evaluate_api" for e in events)!=len(receipts)*(repeats+8):raise ValueError("unexpected/conflated actual diagnostic predictions")
    setup={p["ordinal"]*1000 for p in report["programs"]}
    if any(e["name"] in ("ane.compile_api","ane.load_api") and e["request"] not in setup for e in events):raise ValueError("unexpected compiler/load scope")
    return dict(programs=receipts,evaluations=len(receipts)*(repeats+8),setup_compile_rejections=len(failed),
                scope="actual Private selectors; expected logical overflow still completes physically; not hardware placement or arithmetic proof")


def child(args):
    observer=Observer(args.observer) if args.mode=="timeline" else None
    if observer:observer.lib.tc_qwen_profile_scope(-1)
    C.CDLL(str(args.native_library),mode=C.RTLD_GLOBAL);lib=C.CDLL(str(args.library));scope_type=C.CFUNCTYPE(None,C.c_int)
    scope=scope_type(lambda i:observer.lib.tc_qwen_profile_scope(i)) if observer else scope_type()
    fn=lib.tc_hidden_quant_probe;fn.argtypes=[C.c_char_p,C.c_int,scope_type];fn.restype=C.c_int
    try:
        if fn(str(args.output).encode(),args.repeats,scope):raise RuntimeError("hidden quant probe failed; inspect native report/stderr")
    finally:
        if observer:write_json(args.output/"observer.json",observer.take("tc_qwen_profile_finish"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("library","native-library","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--observer",type=Path,default=ROOT/"build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib")
    p.add_argument("--mode",choices=("stock","timeline"),default="stock");p.add_argument("--repeats",type=int,choices=(12,24),default=12)
    p.add_argument("--child",action="store_true");args=p.parse_args()
    for name in ("library","native_library","observer"):setattr(args,name,getattr(args,name).resolve(strict=True))
    args.output=args.output.resolve()
    if args.child:child(args);return
    manifest=json.loads((args.native_library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    def native_sources():
        if any(sha256_file(ROOT/e["path"])!=e["sha256"] for e in manifest["inputs"]["sources"]):raise ValueError("native source differs")
    native_sources()
    if shutil.disk_usage(ROOT).free<(2<<30)+(768<<20):raise ValueError("2GiB reserve plus768MiB arrays/cache allowance required")
    hashes={s:sha256_file(ROOT/s) for s in SOURCES};assets={s:sha256_file(getattr(args,s)) for s in ("library","native_library","observer")}
    args.output.mkdir(parents=True,exist_ok=False);trial=args.output/"trial";trial.mkdir()
    summary=dict(schema="tc-hidden-quant-screen-v1",status="running",qualification_passed=False,mode=args.mode,repeats=args.repeats,
        source_sha256=hashes,assets=assets,runtime_build_id=manifest["runtime_build_id"],environment={k:v for k,v in environment().items() if k.startswith("TURBOCIDER_")})
    write_json(args.output/"summary.json",summary);write_json(args.output/"source.json",{s:(ROOT/s).read_text() for s in SOURCES});load=LoadObservation(args.output/"load.jsonl")
    try:
        command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child","--library",str(args.library),"--native-library",str(args.native_library),
            "--observer",str(args.observer),"--mode",args.mode,"--repeats",str(args.repeats),"--output",str(trial)]
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            summary["memory"]=run_sampled(command,repo=ROOT,output=args.output,stem="trial",env=environment(),stdout=stdout,stderr=stderr,
                timeout=600,interval_ms=100,max_gap_ms=500,observer=load)
        report=json.loads((trial/"report.json").read_text())
        if (report["cpp_sha256"]!=hashes["tools/profiling/qwen_base/hidden_quant_probe.mm"] or report["header_sha256"]!=hashes["tools/profiling/qwen_base/hidden_quant_mil.hpp"] or
            report.get("rotation_header_sha256")!=hashes["tools/profiling/qwen_base/rotation_mil.hpp"] or
            report["compiled_native_id"]!=manifest["runtime_build_id"] or report["loaded_runtime_id"]!=manifest["runtime_build_id"]):raise ValueError("compiled helper/header/native ABI identity differs")
        summary["analysis"]=validate(report,args.repeats)
        if args.mode=="timeline":
            raw=json.loads((trial/"observer.json").read_text())
            if raw["source_sha256"]!=hashes["tools/profiling/qwen_base/observer.mm"]:raise ValueError("actual compiled observer differs")
            summary["observer"]=observer_contract(raw,report,args.repeats)
        summary["report_sha256"]=sha256_file(trial/"report.json");native_sources();summary["status"]="complete"
    except Exception as error:summary.update(status="failed",error=str(error))
    try:summary["load"]=load.verify()
    except ValueError as error:summary["load_error"]=str(error)
    if any(sha256_file(ROOT/k)!=v for k,v in hashes.items()) or any(sha256_file(getattr(args,k))!=v for k,v in assets.items()):summary.update(status="failed",error="bound source/assets changed")
    write_json(args.output/"summary.json",summary);print(json.dumps(summary,indent=2))
    if summary["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
