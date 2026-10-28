#!/usr/bin/env python3
"""Actual H512/factored rotation complete operator boundaries, not image speed."""
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

ARMS=("identity","conv512","conv64x8","conv32x16","conv16x32","conv128x4","matmul64x8","matmul32x16","flat64x8","flat32x16")
BLOCKS=(0,512,64,32,16,128,-64,-32,10064,10032)
K=len(ARMS)
PATTERNS=("random-a","signed-basis","wide","random-a-restored")
SOURCES=tuple("tools/profiling/qwen_base/"+v for v in ("rotation_collect.py","rotation_probe.mm","rotation_mil.hpp",
    "collect.py","qkv_collect.py","analyze.py","observer.mm"))+("tools/native/build_qwen_rotation_probe.sh",)


def validate(report,repeats,*,legacy=False):
    if type(repeats) is not int or repeats not in (12,24) or type(legacy) is not bool:
        raise ValueError("bounded repeats and explicit historical schema required")
    # Legacy eight-arm operator data can only be admitted explicitly by the
    # CPU archive auditor, never implicitly by a new device launch.
    arms=ARMS[:8] if legacy else ARMS;blocks=BLOCKS[:8] if legacy else BLOCKS;K=len(arms)
    if (report.get("status")!="complete" or report.get("synthetic_only") is not True or report.get("model_weights_loaded") is not False or
        report.get("complete_request") is not False or report.get("image_export") is not False or report.get("repeats")!=repeats or
        report.get("warmups")!=2 or report.get("dtype")!="float16-le" or report.get("layout")!="channels_tokens" or
        any(report.get(k) is not True for k in ("source_unchanged","identity_bits_exact","same_program_restoration_exact"))):
        raise ValueError("complete bounded original/factored rotation with actual restoration controls required")
    programs=report.get("programs",[]);expected=[(m,c,a,b) for m,c in ((32,512),(1024,6144)) for a,b in zip(arms,blocks)]
    if len(programs)!=2*K or [(p["rows"],p["channels"],p["arm"],p["block"]) for p in programs]!=expected:
        raise ValueError("all attempted fixed diagnostic programs, including compile rejections, required")
    ready=[p for p in programs if p.get("status")=="ready"]
    if len({p["key"] for p in ready})!=len(ready) or [p["ordinal"] for p in programs]!=list(range(2*K)):raise ValueError("unique program keys/exact diagnostic ordinal required")
    for p in programs:
        if p.get("status") not in ("ready","compile-rejected") or (p["status"]=="compile-rejected" and (not p.get("error") or p["ordinal"]%K<2)):
            raise ValueError("honest optional compile rejection or successful original controls required")
    expected_arrays=[(g*K+a,(g*K+a)*1000+p+1,arm,pattern,m,c) for g,(m,c) in enumerate(((32,512),(1024,6144)))
                     for p,pattern in enumerate(PATTERNS) for a,arm in enumerate(arms) if programs[g*K+a]["status"]=="ready"]
    actual=[tuple(r[k] for k in ("ordinal","scope","arm","pattern","rows","channels")) for r in report.get("arrays",[])]
    if actual!=expected_arrays:raise ValueError("complete actual source/identity/rotation output patterns required")
    expected_times=[]
    for g in range(2):
        selected=[i for i in range(K) if programs[g*K+i]["status"]=="ready"]
        for r in range(-2,repeats):
            for j in range(len(selected)):
                arm=selected[(j+r+len(selected))%len(selected)];expected_times.append((g*K+arm,r,arms[arm]))
    samples=report.get("samples",[])
    if [(r["ordinal"],r["rep"],r["arm"]) for r in samples]!=expected_times:raise ValueError("balanced warm/measured complete timing sequence required")
    for row in samples:
        if (row["warmup"] is not (row["rep"]<0) or row["input_variant"] is not bool(row["rep"]&1) or
            row["scope"]!=row["ordinal"]*1000+12+row["rep"] or type(row["seconds"]) not in (int,float) or
            not math.isfinite(row["seconds"]) or row["seconds"]<=0):raise ValueError("actual typed positive complete-boundary samples required")
        p=programs[row["ordinal"]]
        if (row["rows"],row["channels"])!=(p["rows"],p["channels"]):raise ValueError("sample fixed shape differs")
    return dict(median_seconds={f"m{m}-c{c}":{p["arm"]:statistics.median(r["seconds"] for r in samples if r["ordinal"]==p["ordinal"] and not r["warmup"])
        for p in ready if p["rows"]==m} for m,c in ((32,512),(1024,6144))},measured_operator_calls=len(ready)*repeats,
        total_operator_calls=len(ready)*(repeats+6),compile_rejected=[p for p in programs if p["status"]=="compile-rejected"],
        scope="synthetic fixed-FP16 complete operator/finite-transfer boundary; not FFN/image or physical ANE hardware time")


def observer_contract(raw,report,repeats):
    # A capability-screen compiler rejection is NOT a timed workload failure.
    # Preserve it in raw evidence and admit only one failed compile in each
    # explicitly rejected setup. All successful workload/GPU gates stay strict.
    if Counter(r["name"] for r in raw["events"])!=Counter(raw["counts"]):raise ValueError("raw selector counters differ")
    rejected={p["ordinal"]*1000 for p in report["programs"] if p["status"]=="compile-rejected"}
    failed=[r for r in raw["events"] if r.get("ok") is not True]
    if (Counter(r["request"] for r in failed)!=Counter({i:1 for i in rejected}) or
        any(r["name"]!="ane.compile_api" or r.get("ok") is not False or type(r.get("begin_ns")) is not int or
            type(r.get("end_ns")) is not int or not 0<r["begin_ns"]<=r["end_ns"] for r in failed)):
        raise ValueError("only actual declared one-time setup compiler rejections may fail")
    value={**raw,"events":[r for r in raw["events"] if r.get("ok") is True]}
    value["counts"]=dict(Counter(r["name"] for r in value["events"]))
    events=analyze.observer_evidence(value);models=set();receipts=[]
    for p in report["programs"]:
        if p["status"]!="ready":continue
        n=p["ordinal"];scopes={n*1000+i for i in range(1,5)}|{n*1000+12+i for i in range(-2,repeats)}
        rows=[r for r in events if r["request"] in scopes];evals=[r for r in rows if r["name"]=="ane.evaluate_api"]
        ids={r["model"] for r in evals};counts=Counter(r["request"] for r in evals)
        if len(ids)!=1 or set(counts)!=scopes or any(v!=1 for v in counts.values()) or any(r["name"] in ("ane.compile_api","ane.load_api") for r in rows):
            raise ValueError("one stable model/exact evaluations and zero hot compiler/load per rotation operator required")
        models|=ids;receipts.append(dict(ordinal=n,model_ids=sorted(ids),hot_compile_load_zero=True,evaluations=len(evals)))
    count=sum(p["status"]=="ready" for p in report["programs"])
    if len(models)!=count:raise ValueError("separate diagnostic programs were conflated")
    setup={p["ordinal"]*1000 for p in report["programs"]}
    if any(r["name"] in ("ane.compile_api","ane.load_api") and r["request"] not in setup for r in events):raise ValueError("unexpected compile/load scope")
    return dict(programs=receipts,evaluations=count*(repeats+6),setup_compile_rejections=len(failed),
                scope="actual Private selectors; explicit rejected setup compiles separate; not physical arithmetic/overlap")


def child(args):
    observer=Observer(args.observer) if args.mode=="timeline" else None
    if observer:observer.lib.tc_qwen_profile_scope(-1)
    C.CDLL(str(args.native_library),mode=C.RTLD_GLOBAL);lib=C.CDLL(str(args.library))
    scope_type=C.CFUNCTYPE(None,C.c_int);scope=scope_type(lambda i:observer.lib.tc_qwen_profile_scope(i)) if observer else scope_type()
    fn=lib.tc_rotation_probe;fn.argtypes=[C.c_char_p,C.c_int,scope_type];fn.restype=C.c_int
    try:
        if fn(str(args.output).encode(),args.repeats,scope):raise RuntimeError("rotation probe failed; inspect report/stderr")
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
        if any(sha256_file(ROOT/e["path"])!=e["sha256"] for e in manifest["inputs"]["sources"]):raise ValueError("native sources differ")
    native_sources()
    if shutil.disk_usage(ROOT).free<(2<<30)+(640<<20):raise ValueError("2GiB reserve plus640MiB operator arrays/cache required")
    hashes={s:sha256_file(ROOT/s) for s in SOURCES};assets={s:sha256_file(getattr(args,s)) for s in ("library","native_library","observer")}
    args.output.mkdir(parents=True,exist_ok=False);trial=args.output/"trial";trial.mkdir()
    summary=dict(schema="tc-h512-rotation-screen-v1",status="running",qualification_passed=False,mode=args.mode,repeats=args.repeats,
        source_sha256=hashes,assets=assets,runtime_build_id=manifest["runtime_build_id"],
        environment={k:v for k,v in environment().items() if k.startswith("TURBOCIDER_")})
    write_json(args.output/"summary.json",summary);write_json(args.output/"source.json",{s:(ROOT/s).read_text() for s in SOURCES})
    load=LoadObservation(args.output/"load.jsonl")
    try:
        command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child","--library",str(args.library),"--native-library",str(args.native_library),
            "--observer",str(args.observer),"--mode",args.mode,"--repeats",str(args.repeats),"--output",str(trial)]
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            summary["memory"]=run_sampled(command,repo=ROOT,output=args.output,stem="trial",env=environment(),stdout=stdout,stderr=stderr,
                timeout=600,interval_ms=100,max_gap_ms=500,observer=load)
        report=json.loads((trial/"report.json").read_text())
        if (report["cpp_sha256"]!=hashes["tools/profiling/qwen_base/rotation_probe.mm"] or report["header_sha256"]!=hashes["tools/profiling/qwen_base/rotation_mil.hpp"] or
            report["loaded_runtime_id"]!=manifest["runtime_build_id"] or report["compiled_native_id"]!=manifest["runtime_build_id"]):raise ValueError("compiled rotation helper/header/native ABI identity differs")
        summary["analysis"]=validate(report,args.repeats)
        if args.mode=="timeline":
            raw=json.loads((trial/"observer.json").read_text())
            if raw["source_sha256"]!=hashes["tools/profiling/qwen_base/observer.mm"]:raise ValueError("observer compiled source differs")
            summary["observer"]=observer_contract(raw,report,args.repeats)
        summary["report_sha256"]=sha256_file(trial/"report.json");native_sources();summary["status"]="complete"
    except Exception as error:summary.update(status="failed",error=str(error))
    try:summary["load"]=load.verify()
    except ValueError as error:summary["load_error"]=str(error)
    if any(sha256_file(ROOT/k)!=v for k,v in hashes.items()) or any(sha256_file(getattr(args,k))!=v for k,v in assets.items()):
        summary.update(status="failed",error="bound operator source/assets changed")
    write_json(args.output/"summary.json",summary);print(json.dumps(summary,indent=2))
    if summary["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
