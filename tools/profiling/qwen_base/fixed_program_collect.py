#!/usr/bin/env python3
"""Actual synthetic overflow branch + GPU fallback, not an image benchmark."""
import argparse
from collections import Counter
import ctypes as C
import json
import os
from pathlib import Path
import sys

import analyze
import collect
from collect import ROOT, Observer, sha256_file, write_json
from runtime_ane_common import benchmark_environment
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

SOURCES=tuple("tools/profiling/qwen_base/"+x for x in
    ("fixed_program_collect.py","fixed_program_probe.mm","observer.mm","analyze.py","collect.py"))+("tools/native/build_qwen_fixed_program_probe.sh",)


def validate(report,observed,strict,activation="exp"):
    if (report.get("status")!="complete" or report.get("synthetic_only") is not True or
            report.get("complete_request") is not False or report.get("strict") is not strict or report.get("image_export") is not False or
            report.get("activation","exp")!=activation or (activation!="exp" and not strict) or
            (report.get("rows"),report.get("hidden"),report.get("ane_channels"),report.get("full_width"))!=(32,128,512,1024)):
        raise ValueError("complete bounded synthetic branch control required")
    direct,hybrid=report["direct"],report["hybrid"]
    a,b=hybrid["first_metrics"]["runtime_weight"],hybrid["second_metrics"]["runtime_weight"]
    first=hybrid["first_metrics"];second=hybrid["second_metrics"]
    if activation!="exp":
        if direct.get("captured_activation_survived_env_change") is not True:
            raise ValueError("non-default activation owner did not retain its captured policy")
        collect.validate_activation_receipt([dict(hybrid=first),dict(hybrid=second)],activation)
    if not direct["captured_policy_survived_env_change"] or not direct["old_output_unchanged"] or not hybrid["source_unchanged"] or not hybrid["disabled_peer_no_model"]:
        raise ValueError("captured policy/source/output/disabled-peer contract differs")
    if strict:
        if (direct["overflow"]["ok"] or direct["overflow"]["calls"]!=1 or direct["overflow"]["overflow_retries"] or
                direct["after"]!={"fixed":True,"instances":1,"blocked_rebuilds":1,"disabled":True} or
                not direct["repeat_stage_rejected"] or direct["after_repeat"]!=direct["after"] or
                not hybrid["complete_gpu_bits_exact"] or not hybrid["sticky_failure"] or hybrid["first_retained_resources"] or hybrid["full_gpu_calls"]!=2):
            raise ValueError("strict overflow must disable without rebuilding and return complete GPU results")
        for metrics in (a,b):
            if (metrics["fixed_program_requested"] is not True or metrics["fixed_program_enabled"] is not True or
                    metrics["program_instances_session_total"]!=1 or metrics["blocked_rebuilds_session_total"]!=1 or
                    metrics["program_disabled"] is not True or metrics["overflow_retries_session_total"] or
                    metrics["headroom_scale"]!=1 or metrics["fallback_blocks_session_total"]!=1):
                raise ValueError("sticky fixed-program failure receipt lost after owner retirement")
        if (first["runtime_calls_session_total"]!=1 or second["runtime_calls_session_total"]!=1):
            raise ValueError("subsequent fixed request must not resubmit ANE")
    else:
        if (not direct["overflow"]["ok"] or direct["overflow"]["overflow_retries"]<1 or
                direct["after"]["instances"]!=1+direct["overflow"]["overflow_retries"] or
                hybrid["sticky_failure"] or hybrid["full_gpu_calls"] or not hybrid["first_retained_resources"] or
                a["overflow_retries_session_total"]<1 or b["overflow_retries_session_total"]!=a["overflow_retries_session_total"] or
                a["program_instances_session_total"]!=1+a["overflow_retries_session_total"] or
                b["program_instances_session_total"]!=a["program_instances_session_total"] or a["headroom_scale"]<=1):
            raise ValueError("legacy positive control must actually recover through a new Program")
    rows=analyze.observer_evidence(observed)
    expected={0:3,1:direct["safe"]["calls"],2:direct["overflow"]["calls"],3:0 if strict else direct["repeat"]["calls"],
              4:3,5:first["runtime_calls_session_total"],6:second["runtime_calls_session_total"]-first["runtime_calls_session_total"],7:0}
    for phase,calls in expected.items():
        actual=sum(r["name"]=="ane.evaluate_api" and r["request"]==phase for r in rows)
        if actual!=calls:raise ValueError("actual overflow/fallback selector submissions differ from native counts")
    hot=[r for r in rows if r["request"] in (1,2,3,5,6,7)]
    hot_counts=Counter(r["name"] for r in hot)
    if strict and (hot_counts["ane.compile_api"] or hot_counts["ane.load_api"]):
        raise ValueError("strict overflow/fallback performed actual hot compile/load")
    if not strict and hot_counts["ane.load_api"]<2:
        raise ValueError("legacy positive control did not load replacement programs in both owners")
    if any(r["name"] in ("ane.load_api","ane.compile_api","ane.evaluate_api") and r["request"]==7 for r in rows):
        raise ValueError("disabled peer allocated a program")
    return dict(actual_evaluations_by_phase=expected,hot_api_calls=dict(hot_counts),
        hot_no_compile=not hot_counts["ane.compile_api"],hot_no_load=not hot_counts["ane.load_api"],
        scope="actual Private client selectors and GPU completion; not hardware INT8, kernel timing or full image request")


def child(args):
    os.environ["TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION"]=args.activation
    observer=Observer(args.observer);observer.lib.tc_qwen_profile_scope(-1)
    native=C.CDLL(str(args.native_library),mode=C.RTLD_GLOBAL)
    lib=C.CDLL(str(args.library))
    callback_type=C.CFUNCTYPE(None,C.c_int)
    callback=callback_type(lambda phase:observer.lib.tc_qwen_profile_scope(phase))
    fn=lib.tc_fixed_program_probe;fn.argtypes=[C.c_char_p,C.c_char_p,C.c_int,callback_type];fn.restype=C.c_int
    try:
        if fn(str(args.output).encode(),str(args.descriptor).encode(),int(args.policy=="fixed"),callback):
            raise RuntimeError("synthetic fixed-program probe failed; inspect report/stderr")
    finally:
        write_json(args.output/"observer.json",observer.take("tc_qwen_profile_finish"))
    # Hold both actual loaded modules through observer retirement.
    _=(native,lib)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--native-library",type=Path,required=True)
    parser.add_argument("--observer",type=Path,default=ROOT/"build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib")
    parser.add_argument("--descriptor",type=Path,default=ROOT/"outputs/ane-fixed-program-toy-m32-h128-f1024-v1-20261018.json")
    parser.add_argument("--policy",choices=("fixed","legacy"),required=True)
    parser.add_argument("--activation",choices=collect.FFN_ACTIVATIONS,default="exp")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--child",action="store_true")
    args=parser.parse_args()
    if args.activation!="exp" and args.policy!="fixed":parser.error("non-default activation requires fixed program")
    for name in ("library","native_library","observer","descriptor"):setattr(args,name,getattr(args,name).resolve(strict=True))
    args.output=args.output.resolve()
    if args.child:child(args);return
    manifest=json.loads((args.native_library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    def native_sources():
        if any(sha256_file(ROOT/e["path"])!=e["sha256"] for e in manifest["inputs"]["sources"]):raise ValueError("native source differs from bound library")
    native_sources()
    hashes={x:sha256_file(ROOT/x) for x in SOURCES}
    assets={x:sha256_file(getattr(args,x)) for x in ("library","native_library","observer","descriptor")}
    args.output.mkdir(parents=True,exist_ok=False);trial=args.output/"trial";trial.mkdir()
    summary=dict(schema="tc-fixed-program-negative-screen-v1",status="running",policy=args.policy,activation=args.activation,qualification_passed=False,
        runtime_build_id=manifest["runtime_build_id"],source_sha256=hashes,assets=assets)
    write_json(args.output/"summary.json",summary);write_json(args.output/"source.json",{x:(ROOT/x).read_text() for x in SOURCES})
    load=LoadObservation(args.output/"load.jsonl")
    command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child","--library",str(args.library),"--native-library",str(args.native_library),
        "--observer",str(args.observer),"--descriptor",str(args.descriptor),"--policy",args.policy,"--activation",args.activation,"--output",str(trial)]
    try:
        env={k:v for k,v in benchmark_environment().items() if not k.startswith("TURBOCIDER_") and k!="MLX_DISABLE_COMPILE"}
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            summary["memory"]=run_sampled(command,repo=ROOT,output=args.output,stem="trial",env=env,
                stdout=stdout,stderr=stderr,timeout=180,interval_ms=100,max_gap_ms=500,observer=load)
        report=json.loads((trial/"report.json").read_text());observed=json.loads((trial/"observer.json").read_text())
        if report["cpp_sha256"]!=hashes["tools/profiling/qwen_base/fixed_program_probe.mm"] or report["loaded_runtime_id"]!=manifest["runtime_build_id"]:
            raise ValueError("actual loaded probe/native source differs")
        if observed["source_sha256"]!=hashes["tools/profiling/qwen_base/observer.mm"]:raise ValueError("actual observer source differs")
        summary["analysis"]=validate(report,observed,args.policy=="fixed",args.activation)
        summary["arrays_sha256"]=sha256_file(trial/"arrays.safetensors")
        native_sources();summary["status"]="complete"
    except Exception as error:summary.update(status="failed",error=str(error))
    try:summary["load"]=load.verify()
    except ValueError as error:summary["load_error"]=str(error)
    if any(sha256_file(ROOT/x)!=digest for x,digest in hashes.items()) or any(sha256_file(getattr(args,x))!=digest for x,digest in assets.items()):
        summary.update(status="failed",error="bound source/asset changed")
    write_json(args.output/"summary.json",summary);print(json.dumps(summary,indent=2))
    if summary["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
