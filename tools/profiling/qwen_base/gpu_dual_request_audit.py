#!/usr/bin/env python3
"""Independent full40 GPU-dual request, exact RGBA, source and selector audit."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tarfile

import numpy as np
import analyze
import collect
import joint_audit
import sequence_audit
from audit import normalized,pixels,quality
from collect import ROOT,model_snapshot,sha256_file,write_json


sys.path.insert(0,str(ROOT/"tools/native"))
from verify_process_tree_samples import verify as verify_memory


def validation_once(metrics,enabled):
    collect.validate_gpu_dual_swiglu(metrics,enabled)
    spans=[m["qwen_gpu_dual_swiglu"]["validation_seconds_session_total"] for m in metrics]
    if any(value!=spans[0] for value in spans[1:]):
        raise ValueError("source-bound GPU validation must not repeat or accrue time on subsequent requests")


def memory_contract(root,summary):
    memory=summary["memory"]
    for name in ("report","evidence"):
        if sha256_file(root/memory[name])!=memory[name+"_sha256"]:raise ValueError("owned memory evidence changed")
    for name,digest in memory["tool_sha256"].items():
        if sha256_file(ROOT/name)!=digest:raise ValueError("actual memory sampler/verifier source differs")
    actual=verify_memory(root/memory["evidence"])
    recorded=json.loads((root/memory["report"]).read_text())
    if (actual!=recorded or actual!=memory["verified"] or actual["correlation_id"]!=memory["correlation_id"] or
        not actual["complete"] or actual["command_exit_code"]!=0):
        raise ValueError("raw process-tree evidence differs from complete successful memory receipt")
    return actual


def window(root,library,manifest,enabled,historical):
    s=json.loads((root/"summary.json").read_text());r=json.loads((root/"trial/report.json").read_text())
    scenario=s["scenario"];mode=s["mode"];text=s["runtime_text_prefix_snapshot"];prefix=s["runtime_prefix_snapshot"]
    policy=s["text_prefix_qkv_step0"];count=3 if scenario=="repeat" else 4
    if (s["status"]!="complete" or r["status"]!="complete" or s.get("gpu_dual_swiglu") is not enabled or r.get("gpu_dual_swiglu") is not enabled or
        s["route"]!="runtime" or mode not in ("stock","timeline") or s["channels"]!=6144 or s["ffn_activation"]!="sigmoid-complement" or
        s["ffn_projection"]!="descriptor" or len(r["runs"])!=count or s.get("resident_prefix") is not False or
        any(s.get(k) is not True or r.get(k) is not True for k in ("private_qkv","private_qkv_prefetch","fixed_program")) or
        s["assets"]["library"]!=sha256_file(library) or s["runtime_build_id"]!=manifest["runtime_build_id"] or r["loaded_runtime_id"]!=manifest["runtime_build_id"] or
        s["model_snapshot"]!=model_snapshot(ROOT/"models/Comfy-Org-Qwen-Image-2.1")):
        raise ValueError("complete current base40 fixed6144 joint/prefetch strong-baseline GPU-dual window required")
    expected=collect.environment("runtime",mode,6144,prefix,False,True,True,"descriptor","sigmoid-complement",
        runtime_text_prefix=text,text_prefix_qkv_step0=policy,private_qkv_prefetch=True,gpu_dual_swiglu=enabled)
    if s["environment"]!=expected:raise ValueError("actual explicit GPU-dual environment differs")
    snapshot=json.loads((root/"source.json").read_text())
    if set(snapshot)!=set(s["source_sha256"]):raise ValueError("complete source snapshot required")
    for name,digest in s["source_sha256"].items():
        raw=snapshot[name].encode()
        if hashlib.sha256(raw).hexdigest()!=digest or not (raw==(ROOT/name).read_bytes() or raw in historical.get(name,set())):raise ValueError("actual current/archive tool source differs")
    metrics=[x["metrics"] for x in r["runs"]]
    collect.validate(metrics,"runtime",mode,6144,prefix,False,True,True,"descriptor","sigmoid-complement",runtime_text_prefix=text,
        text_prefix_matrix=scenario=="text-prefix-matrix",text_prefix_qkv_step0=policy,private_qkv_prefetch=True,gpu_dual_swiglu=enabled)
    validation_once(metrics,enabled)
    requests=[json.loads((root/f"trial/request-{i}.json").read_text()) for i in range(count)]
    geometry=sequence_audit.geometry(requests,metrics) if scenario=="sequence-matrix" else None
    for i,(request,row,m) in enumerate(zip(requests,r["runs"],metrics)):
        if (row["index"]!=i or (request.get("width"),request.get("height"),request.get("steps"),request.get("frames"),request.get("audio"))!=(512,512,40,1,False) or
            request.get("seed")!=(73 if scenario=="sequence-matrix" else (73,74,73,73)[i]) or request.get("loras") or
            request.get("operation")!=m.get("operation") or request.get("execution")!="gpu_ane" or request.get("hybrid_mlp_mode")!="runtime" or
            request.get("allow_approximation") is not True or request.get("qwen21_dit_cache")!="off" or
            sha256_file(Path(request["ane_manifest"]))!=s["assets"]["manifest"]):raise ValueError("actual source operation/geometry/seed/descriptor differs")
        if request.get("inputs"):
            if request.get("qwen21_reference_size")!=1024 or any(x.get("kind")!="image" or sha256_file(Path(x["path"]))!=s["assets"]["reference"] for x in request["inputs"]):
                raise ValueError("actual standard1024 reference bytes differ")
        if scenario=="repeat" and (request.get("prompt")!=s["prompt"] or request.get("operation")!="image."+s["case"]):
            raise ValueError("actual repeated prompt/task differs from bound process workload")
    images=[pixels(root/f"trial/{i}.png") for i in range(count)]
    if any(x.shape!=(512,512,4) for x in images) or np.array_equal(images[0],images[1]):raise ValueError("complete nonstale seed/reference outputs required")
    if scenario=="sequence-matrix" and not np.array_equal(images[0],images[3]):raise ValueError("variable-length request did not restore original image")
    if scenario=="text-prefix-matrix":
        controls=[normalized(root/f"trial/request-{i}.json") for i in range(count)]
        if controls[0]!=controls[3] or not np.array_equal(images[0],images[3]) or controls[0]["prompt"]!=s["prompt"]:
            raise ValueError("prompt matrix did not restore actual source request and complete PNG")
        if set(controls[0])!=set(controls[2]) or {k for k in controls[0] if controls[0][k]!=controls[2][k]}!={"prompt"}:
            raise ValueError("only prompt may change at conditioning miss")
    memory=memory_contract(root,s)
    observed=None
    if mode=="timeline":
        raw=json.loads((root/"trial/observer.json").read_text())
        if raw["source_sha256"]!=s["source_sha256"]["tools/profiling/qwen_base/observer.mm"]:raise ValueError("actual observer source differs")
        observed=analyze.observer_contract(raw,r["runs"],True,expected_models=2)
        if observed!=s["analysis"]["observer"] or not observed["hot_no_compile_observed"] or not observed["hot_no_model_load_observed"]:raise ValueError("actual two-model hot compile/load contract differs")
    hot=[x["request_wall_including_export"] for x in r["runs"][1:]];median=statistics.median(hot) if scenario=="repeat" else None
    if median!=s["analysis"]["hot_median_seconds"] or analyze.native_deltas(r["runs"])!=s["analysis"]["native_runtime_deltas"]:raise ValueError("complete request wall/native counter analysis differs")
    return dict(root=root,summary=s,report=r,images=images,receipt=dict(cohort=root.name,enabled=enabled,case=s["case"],scenario=scenario,mode=mode,
        first_seconds=r["runs"][0]["request_wall_including_export"],hot_seconds=hot,hot_median_seconds=median,
        gpu_receipts=[m["qwen_gpu_dual_swiglu"] for m in metrics],qkv_receipts=[m["qwen_private_qkv"] for m in metrics],
        prefetch_receipts=[m["qwen_private_qkv_prefetch"] for m in metrics],prefix_receipts=[m.get("qwen_runtime_prefix_snapshot") for m in metrics],
        observer=observed,geometry=geometry,memory=memory,load_error=s.get("load_error"),summary_sha256=sha256_file(root/"summary.json")))


def pair(off,on):
    a,b=off["summary"],on["summary"]
    if (a["mode"]!="stock" or b["mode"]!="stock" or set(a["environment"])!=set(b["environment"]) or
        {k for k in a["environment"] if a["environment"][k]!=b["environment"][k]}!={"TURBOCIDER_QWEN21_GPU_DUAL_SWIGLU"} or
        any(a["assets"][k]!=b["assets"][k] for k in ("library","manifest","reference","observer")) or len(off["images"])!=3 or len(on["images"])!=3):
        raise ValueError("only GPU-dual switch may differ in the matched complete stock pair")
    checks=[]
    for i,(x,y) in enumerate(zip(off["images"],on["images"])):
        if normalized(off["root"]/f"trial/request-{i}.json")!=normalized(on["root"]/f"trial/request-{i}.json") or not np.array_equal(x,y):raise ValueError("GPU-dual changed actual request or complete RGBA output")
        checks.append(quality(x,y))
    left=off["receipt"]["prefix_receipts"];right=on["receipt"]["prefix_receipts"]
    if left[0] and right[0] and left[0]["recipe_sha256"]==right[0]["recipe_sha256"]:raise ValueError("different GPU strategies reused the same prefix arithmetic namespace")
    return dict(off=off["root"].name,on=on["root"].name,quality=checks,rgba_exact=True,
                hot_reduction=1-on["receipt"]["hot_median_seconds"]/off["receipt"]["hot_median_seconds"])


def replay(base,other):
    keys=("gpu_dual_swiglu","case","scenario","runtime_prefix_snapshot","runtime_text_prefix_snapshot","text_prefix_qkv_step0")
    if any(base["summary"][k]!=other["summary"][k] for k in keys):return 0
    if base["summary"].get("prompt")!=other["summary"].get("prompt"):return 0
    if len(base["images"])!=len(other["images"]):raise ValueError("incomplete same-recipe replay")
    for i,(a,b) in enumerate(zip(base["images"],other["images"])):
        if normalized(base["root"]/f"trial/request-{i}.json")!=normalized(other["root"]/f"trial/request-{i}.json") or not np.array_equal(a,b):
            raise ValueError("reverse/timeline same-recipe actual request/pixels changed")
    return len(base["images"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("library","native-source-archive","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--pair",nargs=3,metavar=("LABEL","OFF","ON"),action="append",default=[])
    p.add_argument("--window",nargs=2,metavar=("OFF_OR_ON","ROOT"),action="append",default=[])
    p.add_argument("--frozen",nargs=3,metavar=("LABEL","CURRENT","HISTORICAL"),action="append",default=[])
    p.add_argument("--tool-source-archive",type=Path,action="append",default=[]);args=p.parse_args()
    if args.output.exists():p.error("fresh full request audit output required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text());joint_audit.archive_contract(args.native_source_archive,manifest)
    historical={}
    for path in args.tool_source_archive:
        with tarfile.open(path,"r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():historical.setdefault(member.name,set()).add(archive.extractfile(member).read())
    cohorts={};pairs=[]
    for label,left,right in args.pair:
        off=window(Path(left),args.library,manifest,False,historical);on=window(Path(right),args.library,manifest,True,historical)
        cohorts[left]=off;cohorts[right]=on;pairs.append(dict(label=label,**pair(off,on)))
    for mode,path in args.window:
        if mode not in ("off","on"):raise ValueError("explicit window mode required")
        cohorts[path]=window(Path(path),args.library,manifest,mode=="on",historical)
    exact=sum(replay(a,b) for a in cohorts.values() for b in cohorts.values() if a is not b)
    frozen={}
    for label,current,old in args.frozen:
        if current not in cohorts or cohorts[current]["summary"]["gpu_dual_swiglu"]:raise ValueError("frozen regression needs an audited off control")
        v=cohorts[current];old=Path(old);checks=[]
        for i,x in enumerate(v["images"]):
            if normalized(v["root"]/f"trial/request-{i}.json")!=normalized(old/f"trial/request-{i}.json"):raise ValueError("frozen source request differs")
            checks.append(bool(np.array_equal(x,pixels(old/f"trial/{i}.png"))))
        if not all(checks):raise ValueError("new native off control changed frozen original complete PNG")
        frozen[label]=dict(cohort=old.name,rgba_exact=checks,summary_sha256=sha256_file(old/"summary.json"),scope="historical pixels only, not a current timing denominator")
    result=dict(schema="tc-gpu-dual-full40-audit-v1",status="complete",qualification_passed=False,runtime_build_id=manifest["runtime_build_id"],
        native_library_sha256=sha256_file(args.library),native_source_archive_sha256=sha256_file(args.native_source_archive),script_sha256=sha256_file(Path(__file__)),
        pairs=pairs,cohorts=[v["receipt"] for v in cohorts.values()],same_recipe_exact_replays=exact,frozen_regressions=frozen,
        scope="Matched actual base40 requests/full RGBA and fixed-ANE selectors. Not universal semantic, physical INT8/overlap, exclusive-device or driver-reset qualification.")
    args.output.mkdir(parents=True);write_json(args.output/"audit.json",result);print(json.dumps(dict(status="complete",cohorts=len(cohorts),pairs=pairs),indent=2))


if __name__=="__main__":main()
