#!/usr/bin/env python3
"""Real base40 full-block FFN projection screen; not full-image timing."""
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

SOURCES=tuple("tools/profiling/qwen_base/"+x for x in
    ("ffn_projection_collect.py","ffn_projection_probe.mm","collect.py","analyze.py","observer.mm"))+("tools/native/build_qwen_ffn_projection_probe.sh","tools/native/build_qwen_ffn_share_probe.sh")
ARMS=("gpu-original","gpu-staged","private-restaged")


def validate(report,repeats,projection,activation="exp",*,channels=6144):
    if type(channels) is not int or channels not in (5120,5632,6144,6656):
        raise ValueError("explicit supported fixed FFN share required")
    if (report.get("status")!="complete" or report.get("complete_request") is not False or report.get("image_export") is not False or
            report.get("projection")!=projection or report.get("activation","exp")!=activation or
            activation not in collect.FFN_ACTIVATIONS or (activation!="exp" and projection!="descriptor") or
            report.get("repeats")!=repeats or report.get("warmups")!=2 or
            (report.get("completed_full_dit_steps"),report.get("stopped_step"),report.get("stopped_layer"))!=(1,1,0) or
            report.get("gpu_staged_exact") is not True or report.get("source_unchanged") is not True or report.get("fresh_weights_inputs_outputs_verified") is not True):
        raise ValueError("complete real base40 block capture/control required")
    source=report["actual_source_request"]
    if ((source["model"],source["width"],source["height"],source["steps"],source["seed"],source["audio"],source["lora_count"])!=
            ("qwen-image-2.1",512,512,40,73,False,0) or
            not ((source["operation"]=="image.generate" and source["reference_count"]==0) or
                 (source["operation"]=="image.edit" and source["reference_count"]==1 and source["reference_size"]==1024))):
        raise ValueError("actual original-base gen/standard-reference source required")
    m=report["metrics"];w=m["runtime_weight"]
    if (m["runtime_failed"] or m["runtime_calls_session_total"]!=repeats+7 or w["hybrid_blocks_session_total"]!=repeats+7 or
            w["partition_axis"]!="intermediate_channels" or w["ane_channels"]!=channels or w["gpu_channels"]!=12288-channels or
            w["executor_backend"]!="private_ane" or w["data_path"]!="w8a8_hadamard" or not w["fp32_channel_join_enabled"] or
            w["prefetch_enabled"] or w["weight_code_cache"]["enabled"]):
        raise ValueError("actual declared fixed runtime share required, no silent fallback/cache")
    collect.validate_fixed_program([dict(hybrid=m)]);collect.validate_projection_receipt([dict(hybrid=m)],projection)
    collect.validate_activation_receipt([dict(hybrid=m)],activation)
    for name in ("complete_block_quality","alternate_quality","other_weight_quality"):
        if not 0<=report[name]["relative_l2"]<.03 or report[name]["elements"]!=4194304:
            raise ValueError("complete finite real block/source variants required")
    rows=report["samples"]
    if len(rows)!=3*(repeats+2):raise ValueError("all rotated whole-block samples required")
    expected=[(ARMS[(j+rep+3)%3],rep) for rep in range(-2,repeats) for j in range(3)]
    if [(row["arm"],row["rep"]) for row in rows]!=expected:raise ValueError("actual balanced visit order differs")
    seen=Counter()
    for row in rows:
        if (row["arm"] not in ARMS or row["rep"] not in range(-2,repeats) or row["warmup"] is not (row["rep"]<0) or
                row.get("input_variant") is not bool(row["rep"]&1) or
                type(row["seconds"]) not in (int,float) or not math.isfinite(row["seconds"]) or row["seconds"]<=0):
            raise ValueError("actual finite warm/measured block intervals required")
        seen[row["arm"],row["rep"]]+=1
    if set(seen)!={(arm,rep) for arm in ARMS for rep in range(-2,repeats)} or any(v!=1 for v in seen.values()):
        raise ValueError("one sample for every arm/repetition required")
    medians={arm:statistics.median(r["seconds"] for r in rows if r["arm"]==arm and not r["warmup"]) for arm in ARMS}
    return dict(median_seconds=medians,restaged_reduction_vs_original=1-medians["private-restaged"]/medians["gpu-original"],
                measured_blocks=3*repeats,warm_blocks=6,native_workload_predictions=repeats+7,
                scope="real decode layer0 complete blocks incl stage; QKV GPU; not forty-step image speed")


def observer_contract(value,calls):
    rows=analyze.observer_evidence(value);hot=[r for r in rows if r["request"]>=1]
    models={r["model"] for r in hot if r["name"]=="ane.evaluate_api"}
    if len(models)!=1 or sum(r["name"]=="ane.evaluate_api" for r in hot)!=calls:
        raise ValueError("one stable FFN model must serve changed weights/inputs and all timed consumers")
    if any(r["name"] in ("ane.compile_api","ane.load_api") for r in hot):raise ValueError("hot FFN changed or reloaded a program")
    return dict(model_ids=sorted(models),actual_workload_evaluations=calls,hot_compile_load_zero=True)


def child(args):
    os.environ["TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION"]=args.activation
    observer=Observer(args.observer) if args.mode=="timeline" else None
    if observer:observer.lib.tc_qwen_profile_scope(-1)
    native=C.CDLL(str(args.native_library),mode=C.RTLD_GLOBAL);lib=C.CDLL(str(args.library))
    callback_type=C.CFUNCTYPE(None,C.c_int)
    callback=callback_type(lambda phase:observer.lib.tc_qwen_profile_scope(phase)) if observer else callback_type()
    fn=lib.tc_ffn_projection_probe
    fn.argtypes=[C.c_char_p,C.c_char_p,C.c_char_p,C.c_char_p,C.c_char_p,C.c_int,callback_type];fn.restype=C.c_int
    try:
        if fn(str(args.model).encode(),str(args.reference).encode() if args.case=="edit" else b"-",str(args.manifest).encode(),
              str(args.output).encode(),args.projection.encode(),args.repeats,callback):raise RuntimeError("real FFN projection capture failed; inspect report")
    finally:
        if observer:write_json(args.output/"observer.json",observer.take("tc_qwen_profile_finish"))
    _=(native,lib)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library",type=Path,required=True);parser.add_argument("--native-library",type=Path,required=True)
    parser.add_argument("--projection",choices=collect.FFN_PROJECTIONS,required=True)
    parser.add_argument("--activation",choices=collect.FFN_ACTIVATIONS,default="exp")
    parser.add_argument("--case",choices=("generate","edit"),default="generate")
    parser.add_argument("--mode",choices=("stock","timeline"),default="stock")
    parser.add_argument("--repeats",type=int,choices=(8,16),default=8)
    parser.add_argument("--channels",type=int,choices=(5120,5632,6144,6656),default=6144,help="explicit diagnostic helper's fixed FFN share, never a hot policy switch")
    parser.add_argument("--observer",type=Path,default=ROOT/"build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib")
    parser.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    parser.add_argument("--reference",type=Path,default=ROOT/"results/qwen21/teapot-blue-edit-native-512.png")
    parser.add_argument("--manifest",type=Path,default=ROOT/"outputs/qwen-base-runtime-c1024-h4096-f12288-20261018.json")
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--child",action="store_true")
    args=parser.parse_args()
    if args.activation!="exp" and args.projection!="descriptor":parser.error("activation experiment requires descriptor projection")
    for name in ("library","native_library","observer","model","reference","manifest"):setattr(args,name,getattr(args,name).resolve(strict=True))
    args.output=args.output.resolve()
    if args.child:child(args);return
    native=json.loads((args.native_library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    def native_sources():
        if any(sha256_file(ROOT/e["path"])!=e["sha256"] for e in native["inputs"]["sources"]):raise ValueError("native source changed")
    native_sources()
    if shutil.disk_usage(ROOT).free<(2304<<20):parser.error("reserve2GiB plus256MiB outputs before model capture")
    hashes={x:sha256_file(ROOT/x) for x in SOURCES};assets={x:sha256_file(getattr(args,x)) for x in ("library","native_library","observer","manifest","reference")}
    identity=model_snapshot(args.model);args.output.mkdir(parents=True,exist_ok=False);trial=args.output/"trial";trial.mkdir()
    summary=dict(schema="tc-ffn-projection-block-screen-v1",status="running",qualification_passed=False,projection=args.projection,activation=args.activation,
        case=args.case,mode=args.mode,repeats=args.repeats,channels=args.channels,source_sha256=hashes,assets=assets,model_snapshot=identity,runtime_build_id=native["runtime_build_id"])
    write_json(args.output/"summary.json",summary);write_json(args.output/"source.json",{x:(ROOT/x).read_text() for x in SOURCES})
    load=LoadObservation(args.output/"load.jsonl")
    command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child","--library",str(args.library),"--native-library",str(args.native_library),
        "--projection",args.projection,"--activation",args.activation,"--case",args.case,"--mode",args.mode,"--repeats",str(args.repeats),"--observer",str(args.observer),
        "--model",str(args.model),"--reference",str(args.reference),"--manifest",str(args.manifest),"--channels",str(args.channels),"--output",str(trial)]
    try:
        env={k:v for k,v in benchmark_environment().items() if not k.startswith("TURBOCIDER_") and k!="MLX_DISABLE_COMPILE"}
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            summary["memory"]=run_sampled(command,repo=ROOT,output=args.output,stem="trial",env=env,stdout=stdout,stderr=stderr,
                timeout=300,interval_ms=100,max_gap_ms=500,observer=load)
        report=json.loads((trial/"report.json").read_text())
        if report["cpp_sha256"]!=hashes["tools/profiling/qwen_base/ffn_projection_probe.mm"] or report["loaded_runtime_id"]!=native["runtime_build_id"]:
            raise ValueError("actual probe/native source differs")
        if args.channels in (5632,6656) and (report.get("compiled_native_id")!=native["runtime_build_id"] or report.get("neighbor_share_screen") is not True):
            raise ValueError("neighbor shares require the bound actual diagnostic helper")
        summary["analysis"]=validate(report,args.repeats,args.projection,args.activation,channels=args.channels);summary["arrays_sha256"]=sha256_file(trial/"arrays.safetensors")
        if args.mode=="timeline":
            observed=json.loads((trial/"observer.json").read_text())
            if observed["source_sha256"]!=hashes["tools/profiling/qwen_base/observer.mm"]:raise ValueError("observer source differs")
            summary["observer"]=observer_contract(observed,args.repeats+7)
        native_sources();summary["status"]="complete"
    except Exception as error:
        summary.update(status="failed",error=str(error))
        try:summary["probe_error"]=json.loads((trial/"report.json").read_text()).get("error")
        except (OSError,ValueError):pass
    try:summary["load"]=load.verify()
    except ValueError as error:summary["load_error"]=str(error)
    if model_snapshot(args.model)!=identity or any(sha256_file(ROOT/x)!=d for x,d in hashes.items()) or any(sha256_file(getattr(args,x))!=d for x,d in assets.items()):
        summary.update(status="failed",error="bound source/model/asset changed")
    write_json(args.output/"summary.json",summary);print(json.dumps(summary,indent=2))
    if summary["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
