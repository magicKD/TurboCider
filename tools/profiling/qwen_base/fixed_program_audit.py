#!/usr/bin/env python3
"""CPU independent overflow-oracle/source/selector and complete-request audit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile

import numpy as np

from audit import normalized
from collect import ROOT,sha256_file,write_json
import fixed_program_collect as negative
import joint_audit
from qkv_audit import arrays
from vpipe_audit import trace
import fallback_collect
from collect import model_snapshot


def negative_window(root,native_library,probe_library,manifest,historical):
    summary=json.loads((root/"summary.json").read_text())
    report=json.loads((root/"trial/report.json").read_text())
    if (summary["status"]!="complete" or report["loaded_runtime_id"]!=manifest["runtime_build_id"] or
            summary["assets"]["native_library"]!=sha256_file(native_library) or summary["assets"]["library"]!=sha256_file(probe_library)):
        raise ValueError("complete bound negative probe/library required")
    snapshot=json.loads((root/"source.json").read_text())
    if set(snapshot)!=set(summary["source_sha256"]):raise ValueError("complete negative source snapshot required")
    for name,digest in summary["source_sha256"].items():
        raw=snapshot[name].encode()
        if hashlib.sha256(raw).hexdigest()!=digest or not (raw==(ROOT/name).read_bytes() or raw in historical.get(name,set())):
            raise ValueError("negative current/archive source changed")
    observed=json.loads((root/"trial/observer.json").read_text())
    if observed["source_sha256"]!=summary["source_sha256"]["tools/profiling/qwen_base/observer.mm"]:
        raise ValueError("actual observer source differs")
    analysis=negative.validate(report,observed,summary["policy"]=="fixed",summary.get("activation","exp"))
    # JSON object keys are strings even when the in-memory scope map uses ints.
    if json.loads(json.dumps(analysis))!=summary["analysis"] or sha256_file(root/"trial/arrays.safetensors")!=summary["arrays_sha256"]:
        raise ValueError("actual negative selector analysis/arrays changed")
    values=arrays(root/"trial/arrays.safetensors")
    if set(values)!={"input","first","second","gpu_reference","safe_direct"} or any(v.shape!=(1,32,128) or not np.isfinite(v).all() for v in values.values()):
        raise ValueError("complete finite synthetic BF16 arrays required")
    # Independent algebra: diagonal gate/up32 with X8 ->256, SiLU256~256,
    # BF16 hidden65536. Two selected down coefficients .25 yield32768.
    # All are exactly representable; a stale low-range safe output cannot pass.
    if not np.all(values["input"]==8) or not np.all(values["gpu_reference"]==32768) or np.any(values["safe_direct"]==32768):
        raise ValueError("independent original-BF16 synthetic oracle disagrees")
    strict=summary["policy"]=="fixed"
    for name in ("first","second"):
        error=float(np.linalg.norm(values[name].astype(np.float64)-32768)/math.sqrt(values[name].size)/32768)
        if (strict and error!=0) or error>=.06 or not math.isclose(error,report["hybrid"][name+"_relative_l2"],rel_tol=1e-4,abs_tol=1e-6):
            raise ValueError("full-array fallback/legacy error differs from independent oracle")
    for key in ("report","evidence"):
        if sha256_file(root/summary["memory"][key])!=summary["memory"][key+"_sha256"]:raise ValueError("negative memory digest changed")
    if not summary["memory"]["verified"]["complete"] or summary["memory"]["verified"]["command_exit_code"]!=0:
        raise ValueError("complete successful process memory evidence required")
    return dict(cohort=root.name,policy=summary["policy"],activation=summary.get("activation","exp"),analysis=analysis,full_array_cpu_oracle=True,
        direct=report["direct"],hybrid=report["hybrid"],memory=summary["memory"]["verified"],
        summary_sha256=sha256_file(root/"summary.json")),trace(observed)


def matched(a,b):
    x,y=a["summary"]["environment"],b["summary"]["environment"]
    if set(x)!=set(y) or {k for k in x if x[k]!=y[k]}!={"TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM"}:
        raise ValueError("only fixed-program policy may differ in complete-request controls")
    if any(a["summary"]["assets"][k]!=b["summary"]["assets"][k] for k in ("library","manifest","reference")):
        raise ValueError("actual native/reference/descriptor changed")
    for i in range(3):
        if normalized(a["root"]/f"trial/request-{i}.json")!=normalized(b["root"]/f"trial/request-{i}.json"):
            raise ValueError("policy control changed actual input request")


def fallback_window(root,library,manifest,historical,gpu_images):
    summary=json.loads((root/"summary.json").read_text());report=json.loads((root/"trial/report.json").read_text())
    if (summary["status"]!="complete" or report["status"]!="complete" or len(report["runs"])!=3 or
            summary["assets"]["library"]!=sha256_file(library) or report["loaded_runtime_id"]!=manifest["runtime_build_id"] or
            summary["model_snapshot"]!=model_snapshot(ROOT/"models/Comfy-Org-Qwen-Image-2.1")):
        raise ValueError("complete source-bound all-GPU Session fallback required")
    snapshot=json.loads((root/"source.json").read_text())
    if set(snapshot)!=set(summary["source_sha256"]):raise ValueError("complete fallback source snapshot required")
    for name,digest in summary["source_sha256"].items():
        raw=snapshot[name].encode()
        if hashlib.sha256(raw).hexdigest()!=digest or not (raw==(ROOT/name).read_bytes() or raw in historical.get(name,set())):
            raise ValueError("fallback source current/archive mismatch")
    fallback_collect.validate(report["runs"])
    observed=json.loads((root/"trial/observer.json").read_text());rows=fallback_collect.analyze.observer_evidence(observed)
    from collections import Counter
    counts=dict(Counter(r["name"] for r in rows))
    if (counts!=summary["actual_api_counts"] or any(counts.get(n,0) for n in ("ane.compile_api","ane.load_api","ane.evaluate_api")) or
            observed["source_sha256"]!=summary["source_sha256"]["tools/profiling/qwen_base/observer.mm"]):
        raise ValueError("sticky failed owner still submitted Private model work")
    from audit import pixels
    images=[pixels(root/f"trial/{i}.png") for i in range(3)]
    if not np.array_equal(images[0],images[2]) or np.array_equal(images[0],images[1]):raise ValueError("fallback seed change/restoration failed")
    if gpu_images:
        for i,image in enumerate(images):
            left=normalized(root/f"trial/request-{i}.json");right=normalized(gpu_images/f"request-{i}.json")
            for value in (left,right):
                for key in ("execution","ane_manifest","hybrid_mlp_mode"):value.pop(key,None)
            if left!=right or not np.array_equal(image,pixels(gpu_images/f"{i}.png")):
                raise ValueError("full Session fallback differs from actual frozen GPU inputs/PNG")
    for key in ("report","evidence"):
        if sha256_file(root/summary["memory"][key])!=summary["memory"][key+"_sha256"]:raise ValueError("fallback memory digest changed")
    if not summary["memory"]["verified"]["complete"] or summary["memory"]["verified"]["command_exit_code"]!=0:
        raise ValueError("incomplete owned fallback process")
    return dict(cohort=root.name,actual_api_counts=counts,full_gpu_rgba_exact=bool(gpu_images),
        ffn_gpu_blocks=[r["metrics"]["hybrid"]["runtime_weight"]["gpu_blocks_session_total"] for r in report["runs"]],
        qkv_gpu_blocks=[r["metrics"]["qwen_private_qkv"]["gpu_blocks_session_total"] for r in report["runs"]],
        memory=summary["memory"]["verified"],scope="initial admission denied before any ANE Program;3 actual complete base40 requests"),trace(observed)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("library","native-source-archive","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--probe-library",type=Path)
    parser.add_argument("--negative",type=Path,action="append",default=[])
    parser.add_argument("--off",type=Path)
    parser.add_argument("--on",type=Path)
    parser.add_argument("--timeline",type=Path)
    parser.add_argument("--reverse-off",type=Path)
    parser.add_argument("--case",choices=("generate","edit"))
    parser.add_argument("--historical-images",type=Path,help="optional old same-arithmetic PNG control only; never old timing denominator")
    parser.add_argument("--fallback",type=Path)
    parser.add_argument("--fallback-gpu-images",type=Path,help="frozen complete GPU PNG/request directory, quality only")
    parser.add_argument("--tool-source-archive",type=Path,action="append",default=[])
    args=parser.parse_args()
    if args.output.exists():parser.error("fresh audit directory required")
    if bool(args.off)!=bool(args.on) or (args.on and not args.case) or (args.negative and not args.probe_library):
        parser.error("request off/on need case; negatives need matching probe library")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    joint_audit.archive_contract(args.native_source_archive,manifest)
    historical={}
    for path in args.tool_source_archive:
        with tarfile.open(path,"r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():historical.setdefault(member.name,set()).add(archive.extractfile(member).read())
    negatives=[];traces=[]
    for root in args.negative:
        value,events=negative_window(root,args.library,args.probe_library,manifest,historical);negatives.append(value);traces.append(events)
    controls=[];reduction=None
    fallback=None
    if args.fallback:
        fallback,events=fallback_window(args.fallback,args.library,manifest,historical,args.fallback_gpu_images)
        traces.append(events)
    if args.on:
        def load(root,fixed,mode="stock"):
            return joint_audit.window(root,args.library,manifest,True,mode,args.case,args.case=="edit",historical,fixed_program=fixed)
        off,on=load(args.off,False),load(args.on,True);matched(off,on)
        if any(not np.array_equal(a,b) for a,b in zip(off["images"],on["images"])):
            raise ValueError("successful fixed policy changed full request arithmetic")
        controls=[off["receipt"],on["receipt"]]
        for root,fixed,mode,base in ((args.timeline,True,"timeline",on),(args.reverse_off,False,"stock",off)):
            if root is None:continue
            value=load(root,fixed,mode)
            if any(not np.array_equal(a,b) for a,b in zip(value["images"],base["images"])):
                raise ValueError("timeline/reverse changed original successful arithmetic")
            controls.append(value["receipt"])
        if args.historical_images:
            from audit import pixels
            if any(not np.array_equal(pixels(args.historical_images/f"{i}.png"),off["images"][i]) for i in range(3)):
                raise ValueError("disabled new library changed historical same-arithmetic PNGs")
        reduction=1-on["receipt"]["hot_median_seconds"]/off["receipt"]["hot_median_seconds"]
    result=dict(schema="tc-fixed-program-audit-v1",status="complete",qualification_passed=False,
        library_sha256=sha256_file(args.library),runtime_build_id=manifest["runtime_build_id"],script_sha256=sha256_file(Path(__file__)),
        native_source_archive_sha256=sha256_file(args.native_source_archive),negative_controls=negatives,request_controls=controls,
        session_fallback=fallback,
        successful_request_rgba_exact=True if controls else None,stock_hot_wall_reduction=reduction,
        scope="actual overflow synthetic component and optional full-base requests; no hardware INT8/opaque-driver no-compile guarantee")
    args.output.mkdir(parents=True)
    for i,events in enumerate(traces):write_json(args.output/f"negative-{i}-trace.json",events)
    write_json(args.output/"audit.json",result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
