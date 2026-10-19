#!/usr/bin/env python3
"""Actual fixed FP16 activation operator diagnostics, not model throughput."""
import argparse
from collections import Counter
import ctypes as C
import json
from pathlib import Path
import shutil
import sys

import analyze
from collect import ROOT,Observer,sha256_file,write_json
from runtime_ane_common import benchmark_environment
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

SOURCES=tuple("tools/profiling/qwen_base/"+x for x in
    ("activation_op_collect.py","activation_op_probe.mm","observer.mm","analyze.py","collect.py"))+("tools/native/build_qwen_activation_op_probe.sh",)
MODES=("identity","exp-divide","sigmoid","silu","sigmoid-mul","sigmoid-complement")
PATTERNS=("small0125","negative025","wide","small0125-restored")


def validate(report,observed):
    if (report.get("status")!="complete" or report.get("synthetic_only") is not True or report.get("model_weights_loaded") is not False or
            report.get("complete_request") is not False or report.get("image_export") is not False or
            report.get("dtype")!="float16-le" or report.get("layout")!="channels_tokens" or
            any(report.get(k) is not True for k in ("source_unchanged","identity_bits_exact","same_program_restoration_exact"))):
        raise ValueError("actual complete FP16 operator/identity/owned-output diagnostics required")
    programs=report["programs"];samples=report["samples"]
    expected=[(rows,channels,mode) for rows,channels in ((32,128),(1024,6144)) for mode in MODES]
    count=2*len(MODES)
    if (len(programs)!=count or len(samples)!=count*4 or len({p["cache_key"] for p in programs})!=count or
            [(p["rows"],p["channels"],p["operator"]) for p in programs]!=expected or
            [p["ordinal"] for p in programs]!=list(range(count))):raise ValueError("all fixed typed operator programs required")
    events=analyze.observer_evidence(observed);setup={i*10 for i in range(count)}
    receipts=[]
    for ordinal,(rows,channels,mode) in enumerate(expected):
        scopes={ordinal*10+i+1 for i in range(4)};actual=[r for r in events if r["request"] in scopes]
        models={r["model"] for r in actual if r["name"]=="ane.evaluate_api"}
        if len(models)!=1 or any(r["name"] in ("ane.compile_api","ane.load_api") for r in actual):
            raise ValueError("one stable operator model, no compile/load during changed-input calls required")
        for pattern in range(4):
            row=samples[ordinal*4+pattern];scope=ordinal*10+pattern+1
            if ((row["program"],row["scope"],row["operator"],row["pattern"],row["rows"],row["channels"])!=
                    (ordinal,scope,mode,PATTERNS[pattern],rows,channels) or
                    row["quality"]["elements"]!=rows*channels or
                    sum(r["name"]=="ane.evaluate_api" and r["request"]==scope for r in actual)!=1):
                raise ValueError("all actual array shapes/inputs/evaluations must match fixed programs")
        receipts.append(dict(program=ordinal,model_ids=sorted(models),actual_evaluations=4,hot_compile_load_zero=True,
            setup_api_calls=dict(Counter(r["name"] for r in events if r["request"]==ordinal*10))))
    if any(r["name"] in ("ane.compile_api","ane.load_api") and r["request"] not in setup for r in events):
        raise ValueError("unexpected operator compiler/load scope")
    return dict(programs=receipts,actual_evaluations=count*4,
        scope="raw FP16 operator accuracy/identity diagnostic; no checkpoint, FFN, image speed or physical placement qualification")


def child(args):
    observer=Observer(args.observer);observer.lib.tc_qwen_profile_scope(-1)
    native=C.CDLL(str(args.native_library),mode=C.RTLD_GLOBAL);lib=C.CDLL(str(args.library))
    callback_type=C.CFUNCTYPE(None,C.c_int);callback=callback_type(lambda phase:observer.lib.tc_qwen_profile_scope(phase))
    fn=lib.tc_activation_op_probe;fn.argtypes=[C.c_char_p,callback_type];fn.restype=C.c_int
    try:
        if fn(str(args.output).encode(),callback):raise RuntimeError("activation operator diagnostic failed; inspect report")
    finally:write_json(args.output/"observer.json",observer.take("tc_qwen_profile_finish"))
    _=(native,lib)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("library","native-library","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--observer",type=Path,default=ROOT/"build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib")
    parser.add_argument("--child",action="store_true");args=parser.parse_args()
    for name in ("library","native_library","observer"):setattr(args,name,getattr(args,name).resolve(strict=True))
    args.output=args.output.resolve()
    if args.child:child(args);return
    native=json.loads((args.native_library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    def native_sources():
        if any(sha256_file(ROOT/e["path"])!=e["sha256"] for e in native["inputs"]["sources"]):raise ValueError("native sources differ from the loaded library")
    native_sources()
    if shutil.disk_usage(ROOT).free<(2688<<20):parser.error("reserve2GiB plus640MiB full operator arrays before running")
    hashes={x:sha256_file(ROOT/x) for x in SOURCES};assets={x:sha256_file(getattr(args,x)) for x in ("library","native_library","observer")}
    args.output.mkdir(parents=True,exist_ok=False);trial=args.output/"trial";trial.mkdir()
    summary=dict(schema="tc-activation-operator-screen-v1",status="running",qualification_passed=False,
        source_sha256=hashes,assets=assets,runtime_build_id=native["runtime_build_id"])
    write_json(args.output/"summary.json",summary);write_json(args.output/"source.json",{x:(ROOT/x).read_text() for x in SOURCES})
    load=LoadObservation(args.output/"load.jsonl")
    command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child","--library",str(args.library),"--native-library",str(args.native_library),
        "--observer",str(args.observer),"--output",str(trial)]
    try:
        env={k:v for k,v in benchmark_environment().items() if not k.startswith("TURBOCIDER_") and k!="MLX_DISABLE_COMPILE"}
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            summary["memory"]=run_sampled(command,repo=ROOT,output=args.output,stem="trial",env=env,stdout=stdout,stderr=stderr,
                timeout=300,interval_ms=100,max_gap_ms=500,observer=load)
        report=json.loads((trial/"report.json").read_text());observed=json.loads((trial/"observer.json").read_text())
        if (report["cpp_sha256"]!=hashes["tools/profiling/qwen_base/activation_op_probe.mm"] or
                report["loaded_runtime_id"]!=native["runtime_build_id"] or observed["source_sha256"]!=hashes["tools/profiling/qwen_base/observer.mm"]):
            raise ValueError("actual operator/observer/native source differs")
        summary["analysis"]=validate(report,observed);summary["report_sha256"]=sha256_file(trial/"report.json");native_sources();summary["status"]="complete"
    except Exception as error:
        summary.update(status="failed",error=str(error))
        try:summary["probe_error"]=json.loads((trial/"report.json").read_text()).get("error")
        except (OSError,ValueError):pass
    try:summary["load"]=load.verify()
    except ValueError as error:summary["load_error"]=str(error)
    if any(sha256_file(ROOT/x)!=d for x,d in hashes.items()) or any(sha256_file(getattr(args,x))!=d for x,d in assets.items()):
        summary.update(status="failed",error="bound diagnostic source/asset changed")
    write_json(args.output/"summary.json",summary);print(json.dumps(summary,indent=2))
    if summary["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
