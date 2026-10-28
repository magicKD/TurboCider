#!/usr/bin/env python3
"""Full-base Session negative: deny graph admission, require sticky exact GPU."""
import argparse
from collections import Counter
import ctypes as C
import json
import os
from pathlib import Path
import shutil
import sys
import time

import analyze
import collect
from collect import ROOT,Observer,model_snapshot,sha256_file,write_json
from runtime_ane_common import benchmark_environment
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

SOURCES=collect.SOURCES+("tools/profiling/qwen_base/fallback_collect.py",)


def environment():
    flags=collect.environment("runtime","timeline",6144,False,False,True,True)
    # Existing optional cap stays2GiB; the graph/scratch estimate cannot fit
    # IN ADDITION to this requested cache. No cache payload is allocated.
    flags["TURBOCIDER_RUNTIME_ANE_WEIGHT_CODE_CACHE_BYTES"]=str(2<<30)
    return flags


def validate(rows):
    for index,row in enumerate(rows):
        m=row["metrics"];h=m.get("hybrid") or {};w=h.get("runtime_weight") or {};q=m.get("qwen_private_qkv") or {}
        if ((m.get("model"),m.get("operation"),m.get("width"),m.get("height"),m.get("steps"),m.get("actual_denoise_steps"),m.get("seed"))!=
                ("qwen-image-2.1","image.generate",512,512,40,40,(73,74,73)[index]) or m.get("lora_applied_projections",0) or m.get("qwen21_dbcache") or
                h.get("runtime_failed") is not True or h.get("runtime_failures_session_total")!=1 or h.get("runtime_calls_session_total")!=0 or
                w.get("fixed_program_requested") is not True or w.get("fixed_program_enabled") is not False or
                w.get("program_instances_session_total")!=0 or w.get("blocked_rebuilds_session_total")!=0 or
                w.get("gpu_blocks_session_total")!=1280*(index+1) or w.get("hybrid_blocks_session_total")!=0 or
                w.get("overflow_retries_session_total") or "exceed memory budget" not in w.get("failure_reason","") or
                q.get("failed") is not True or q.get("predictions_session_total")!=0 or q.get("slot_bytes")!=0 or
                q.get("gpu_blocks_session_total")!=1280*(index+1) or not q.get("failure_reason","").startswith("fixed FFN disabled")):
            raise ValueError("actual sticky same-owner Session failure, no program and all1280 GPU blocks required")
        if m.get("qwen_runtime_prefix_snapshot") or "private_ane" in m.get("backend",""):
            raise ValueError("fallback cannot masquerade as executed Private or publish runtime prefix")


def child(args):
    os.environ.update(environment())
    observer=Observer(args.observer);observer.lib.tc_qwen_profile_scope(-1)
    lib=C.CDLL(str(args.library))
    lib.tc_string_free.argtypes=[C.c_void_p];lib.tc_runtime_build_identity.restype=C.c_void_p
    lib.tc_engine_create_model.argtypes=[C.c_char_p,C.c_char_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
    callback_type=C.CFUNCTYPE(None,C.c_char_p,C.c_void_p)
    lib.tc_engine_generate.argtypes=[C.c_void_p,C.c_char_p,callback_type,C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_void_p)]
    lib.tc_engine_free.argtypes=[C.c_void_p]
    def take(pointer):
        if not pointer:return ""
        try:return C.string_at(pointer).decode()
        finally:lib.tc_string_free(pointer)
    callback=callback_type(lambda _data,_context:None)
    report=dict(schema="tc-fixed-program-session-fallback-report-v1",status="running",runs=[],
        loaded_runtime_id=take(lib.tc_runtime_build_identity()),intent="initial admission denial, not injected ANE overflow")
    engine,error=C.c_void_p(),C.c_void_p()
    requests=collect.base.requests(argparse.Namespace(route="runtime-off",case="generate",output=args.output,manifest=args.manifest))
    try:
        if lib.tc_engine_create_model(b"qwen-image-2.1",str(args.model).encode(),C.byref(engine),C.byref(error)):
            raise RuntimeError(take(error.value))
        for index,request in enumerate(requests):
            write_json(args.output/f"request-{index}.json",request)
            observer.lib.tc_qwen_profile_scope(index)
            result,error=C.c_void_p(),C.c_void_p();start=time.perf_counter();begin=observer.now()
            code=lib.tc_engine_generate(engine,json.dumps(request).encode(),callback,None,C.byref(result),C.byref(error))
            wall=time.perf_counter()-start;end=observer.now();payload,failure=take(result.value),take(error.value)
            if code:raise RuntimeError(failure)
            if observer.lib.tc_qwen_profile_wait(2000):raise RuntimeError("observer failed to drain")
            report["runs"].append(dict(index=index,begin_ns=begin,end_ns=end,wall_seconds=wall,metrics=json.loads(payload)))
            write_json(args.output/"report.json",report);validate(report["runs"])
        report["status"]="complete"
    except Exception as error:report.update(status="failed",error=str(error));raise
    finally:
        observer.lib.tc_qwen_profile_scope(-3)
        if engine.value:lib.tc_engine_free(engine)
        write_json(args.output/"observer.json",observer.take("tc_qwen_profile_finish"))
        write_json(args.output/"report.json",report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--observer",type=Path,default=ROOT/"build/qwen-base-profile-observer-v2-20261018/libqwen_base_profile.dylib")
    parser.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    parser.add_argument("--manifest",type=Path,default=ROOT/"outputs/qwen-base-runtime-c1024-h4096-f12288-20261018.json")
    parser.add_argument("--child",action="store_true")
    args=parser.parse_args()
    for name in ("library","observer","model","manifest"):setattr(args,name,getattr(args,name).resolve(strict=True))
    args.output=args.output.resolve()
    if args.child:child(args);return
    native=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    def native_sources():
        if any(sha256_file(ROOT/e["path"])!=e["sha256"] for e in native["inputs"]["sources"]):raise ValueError("native source changed")
    native_sources()
    if shutil.disk_usage(ROOT).free<2304<<20:parser.error("reserve2GiB plus256MiB output allowance")
    hashes={x:sha256_file(ROOT/x) for x in SOURCES};assets={x:sha256_file(getattr(args,x)) for x in ("library","observer","manifest")}
    identity=model_snapshot(args.model)
    args.output.mkdir(parents=True,exist_ok=False);trial=args.output/"trial";trial.mkdir()
    summary=dict(schema="tc-fixed-program-session-fallback-v1",status="running",qualification_passed=False,
        runtime_build_id=native["runtime_build_id"],source_sha256=hashes,assets=assets,model_snapshot=identity,environment=environment(),
        scope="intentional graph-admission denial; all-GPU negative, not successful ANE timing")
    write_json(args.output/"summary.json",summary);write_json(args.output/"source.json",{x:(ROOT/x).read_text() for x in SOURCES})
    load=LoadObservation(args.output/"load.jsonl")
    command=[sys.executable,"-B",str(Path(__file__).resolve()),"--child","--library",str(args.library),"--output",str(trial),
        "--observer",str(args.observer),"--model",str(args.model),"--manifest",str(args.manifest)]
    try:
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            summary["memory"]=run_sampled(command,repo=ROOT,output=args.output,stem="trial",env=benchmark_environment(),
                stdout=stdout,stderr=stderr,timeout=600,interval_ms=100,max_gap_ms=500,observer=load)
        report=json.loads((trial/"report.json").read_text());validate(report["runs"])
        if report["status"]!="complete" or len(report["runs"])!=3 or report["loaded_runtime_id"]!=native["runtime_build_id"]:
            raise ValueError("complete actual three-request fallback window required")
        observed=json.loads((trial/"observer.json").read_text());rows=analyze.observer_evidence(observed)
        if observed["source_sha256"]!=hashes["tools/profiling/qwen_base/observer.mm"]:raise ValueError("actual observer differs")
        counts=Counter(r["name"] for r in rows)
        if any(counts[name] for name in ("ane.evaluate_api","ane.compile_api","ane.load_api")):
            raise ValueError("failed Session retried or loaded an ANE model")
        summary["actual_api_counts"]=dict(counts);native_sources();summary["status"]="complete"
    except Exception as error:summary.update(status="failed",error=str(error))
    try:summary["load"]=load.verify()
    except ValueError as error:summary["load_error"]=str(error)
    if model_snapshot(args.model)!=identity or any(sha256_file(ROOT/x)!=digest for x,digest in hashes.items()) or any(sha256_file(getattr(args,x))!=digest for x,digest in assets.items()):
        summary.update(status="failed",error="bound source/model/asset changed")
    write_json(args.output/"summary.json",summary);print(json.dumps(summary,indent=2))
    if summary["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
