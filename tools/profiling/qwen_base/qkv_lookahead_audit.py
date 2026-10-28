#!/usr/bin/env python3
"""CPU-only independent state/oracle and matched full40 lookahead audit."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import analyze
import collect
import joint_audit
import qkv_lookahead_collect as state
import sequence_audit
from audit import normalized,pixels,quality
from collect import ROOT,sha256_file,model_snapshot,write_json
from qkv_audit import arrays,quality as tensor_quality


def sources(root,summary):
    snapshot=json.loads((root/"source.json").read_text())
    if set(snapshot)!=set(summary["source_sha256"]):raise ValueError("complete tool source snapshot required")
    for name,digest in summary["source_sha256"].items():
        payload=snapshot[name].encode()
        if hashlib.sha256(payload).hexdigest()!=digest or payload!=(ROOT/name).read_bytes():
            raise ValueError("bound tool source differs")


def memory(root,summary):
    for name in ("report","evidence"):
        if sha256_file(root/summary["memory"][name])!=summary["memory"][name+"_sha256"]:raise ValueError("owned memory evidence changed")
    if not summary["memory"]["verified"]["complete"] or summary["memory"]["verified"]["command_exit_code"]!=0:
        raise ValueError("complete successful owned process required")


def synthetic_arrays(values):
    expected_names={"input_a","input_b","output_a","output_b","restored_a","mismatch_a","generation_a"}
    if set(values)!=expected_names:raise ValueError("all actual source/projection/restoration arrays required")
    for name,constant in (("input_a",.5),("input_b",.375)):
        if values[name].shape!=(1,1024,4096) or not np.all(values[name]==constant):raise ValueError("complete unchanged synthetic BF16 input required")
    result={}
    for name,gains,input_name in (("output_a",(.125,.25,.5),"input_a"),("output_b",(.1875,.375,.75),"input_b")):
        oracle=np.concatenate([values[input_name]*gain for gain in gains],axis=-1)
        result[name]=tensor_quality(oracle,values[name])
        if result[name]["relative_l2"]>=.02 or result[name]["max_abs"]>.01:
            raise ValueError("full independent diagonal projection oracle differs")
        if not np.array_equal(oracle[:,:512],values[name][:,:512]):raise ValueError("ordinary GPU head differs from exact diagonal oracle")
    for name in ("restored_a","mismatch_a","generation_a"):
        if not np.array_equal(values[name],values["output_a"]):raise ValueError("A/B/A, Tensor-ID mismatch or generation drain returned different/stale output")
    if np.array_equal(values["output_a"],values["output_b"]):raise ValueError("weight/input switch returned stale projection")
    return result


def synthetic(root,library,manifest):
    summary=json.loads((root/"summary.json").read_text());report=json.loads((root/"trial/report.json").read_text())
    if summary["status"]!="complete" or summary["assets"]["native_library"]!=sha256_file(library):raise ValueError("complete matching state probe required")
    sources(root,summary);memory(root,summary);state.validate(report)
    if (report["loaded_runtime_id"]!=manifest["runtime_build_id"] or report["compiled_native_id"]!=manifest["runtime_build_id"] or
        report["cpp_sha256"]!=summary["source_sha256"]["tools/profiling/qwen_base/qkv_lookahead_probe.mm"]):raise ValueError("compiled state helper/native ABI identity differs")
    if sha256_file(root/"trial/arrays.safetensors")!=summary["arrays_sha256"]:raise ValueError("raw state arrays changed")
    observed=None
    if summary["mode"]=="timeline":
        raw=json.loads((root/"trial/observer.json").read_text())
        if raw["source_sha256"]!=summary["source_sha256"]["tools/profiling/qwen_base/observer.mm"]:raise ValueError("compiled observer source differs")
        observed=state.observer_contract(raw)
        if observed!=summary["observer"]:raise ValueError("actual state selector evidence differs")
    return dict(cohort=root.name,prefetch=report["prefetch"],oracle=synthetic_arrays(arrays(root/"trial/arrays.safetensors")),
                observer=observed,memory=summary["memory"]["verified"],summary_sha256=sha256_file(root/"summary.json"))


def window(root,library,manifest,enabled):
    summary=json.loads((root/"summary.json").read_text());report=json.loads((root/"trial/report.json").read_text())
    mode=summary["mode"];scenario=summary["scenario"];text=summary["runtime_text_prefix_snapshot"];prefix=summary["runtime_prefix_snapshot"]
    policy=summary["text_prefix_qkv_step0"];count=4 if scenario!="repeat" else 3
    if (summary["status"]!="complete" or report["status"]!="complete" or summary["route"]!="runtime" or mode not in ("stock","timeline") or
        scenario not in ("repeat","sequence-matrix","text-prefix-matrix") or len(report["runs"])!=count or
        summary.get("private_qkv_prefetch") is not enabled or report.get("private_qkv_prefetch") is not enabled or
        any(summary.get(k) is not True or report.get(k) is not True for k in ("private_qkv","fixed_program")) or
        summary.get("resident_prefix") is not False or summary["channels"]!=6144 or summary["ffn_projection"]!="descriptor" or
        summary["ffn_activation"]!="sigmoid-complement" or summary["assets"]["library"]!=sha256_file(library) or
        report["loaded_runtime_id"]!=manifest["runtime_build_id"] or summary["runtime_build_id"]!=manifest["runtime_build_id"] or
        summary["model_snapshot"]!=model_snapshot(ROOT/"models/Comfy-Org-Qwen-Image-2.1")):
        raise ValueError("complete current original-base fixed6144 joint lookahead window required")
    expected=collect.environment("runtime",mode,6144,prefix,False,True,True,"descriptor","sigmoid-complement",
        runtime_text_prefix=text,text_prefix_qkv_step0=policy,private_qkv_prefetch=enabled)
    if summary["environment"]!=expected:raise ValueError("actual explicit lookahead environment differs")
    sources(root,summary);memory(root,summary)
    metrics=[r["metrics"] for r in report["runs"]]
    collect.validate(metrics,"runtime",mode,6144,prefix,False,True,True,"descriptor","sigmoid-complement",
        runtime_text_prefix=text,text_prefix_matrix=scenario=="text-prefix-matrix",text_prefix_qkv_step0=policy,private_qkv_prefetch=enabled)
    requests=[json.loads((root/f"trial/request-{i}.json").read_text()) for i in range(count)]
    geometry=None
    if scenario=="sequence-matrix":geometry=sequence_audit.geometry(requests,metrics)
    previous_calls=0
    for i,(request,row,metric) in enumerate(zip(requests,report["runs"],metrics)):
        if (row["index"]!=i or request.get("seed")!=(73 if scenario=="sequence-matrix" else (73,74,73,73)[i]) or request.get("loras") or
            request.get("audio") is not False or metric["operation"]!=request["operation"] or
            (request["width"],request["height"],request["steps"],metric["actual_denoise_steps"])!=(512,512,40,40) or
            sha256_file(Path(request["ane_manifest"]))!=summary["assets"]["manifest"]):raise ValueError("actual input/seed/shape/descriptor differs")
        if request.get("inputs") and any(not isinstance(reference,dict) or not isinstance(reference.get("path"),str) or
            sha256_file(Path(reference["path"]))!=summary["assets"]["reference"] for reference in request["inputs"]):
            raise ValueError("actual reference differs")
        calls=metric["hybrid"]["runtime_calls_session_total"]
        phase=metric["qwen_ffn_phases"];want=phase["prefill"]["runtime_calls_this_request"]+phase["decode"]["runtime_calls_this_request"]
        if calls-previous_calls!=want:raise ValueError("full actual FFN calls differ from phase work")
        previous_calls=calls
    images=[pixels(root/f"trial/{i}.png") for i in range(count)]
    if any(v.shape!=(512,512,4) for v in images) or np.array_equal(images[0],images[1]):raise ValueError("complete nonstale PNGs required")
    if scenario=="sequence-matrix" and not np.array_equal(images[0],images[3]):raise ValueError("variable-length sequence did not restore original image")
    observed=None
    if mode=="timeline":
        raw=json.loads((root/"trial/observer.json").read_text())
        if raw["source_sha256"]!=summary["source_sha256"]["tools/profiling/qwen_base/observer.mm"]:raise ValueError("observer compiled source differs")
        observed=analyze.observer_contract(raw,report["runs"],True,expected_models=2)
        if observed!=summary["analysis"]["observer"] or not observed["hot_no_compile_observed"] or not observed["hot_no_model_load_observed"]:
            raise ValueError("actual two-model no-hot-compile/load selector contract differs")
    hot=[r["request_wall_including_export"] for r in report["runs"][1:]]
    median=statistics.median(hot) if scenario=="repeat" else None
    if median!=summary["analysis"]["hot_median_seconds"]:raise ValueError("full request timing denominator differs")
    deltas=[];previous={}
    for metric in metrics:
        values=metric["qwen_private_qkv_prefetch"]
        deltas.append({k:v-previous.get(k,0) for k,v in values.items() if k.endswith("_session_total")})
        previous=values
    return dict(root=root,summary=summary,report=report,images=images,receipt=dict(cohort=root.name,enabled=enabled,case=summary["case"],
        mode=mode,scenario=scenario,first_seconds=report["runs"][0]["request_wall_including_export"],hot_seconds=hot,hot_median_seconds=median,
        per_request_lookahead=deltas,native_deltas=analyze.native_deltas(report["runs"]),observer=observed,geometry=geometry,
        memory=summary["memory"]["verified"],load_error=summary.get("load_error"),summary_sha256=sha256_file(root/"summary.json")))


def pair(off,on):
    a,b=off["summary"],on["summary"]
    if (a["mode"]!="stock" or b["mode"]!="stock" or len(off["images"])!=3 or len(on["images"])!=3 or set(a["environment"])!=set(b["environment"]) or
        {k for k in a["environment"] if a["environment"][k]!=b["environment"][k]}!={"TURBOCIDER_QWEN21_PRIVATE_QKV_PREFETCH"} or
        any(a["assets"][k]!=b["assets"][k] for k in ("library","manifest","reference"))):raise ValueError("only lookahead switch may differ in stock pair")
    checks=[]
    for i,(x,y) in enumerate(zip(off["images"],on["images"])):
        if normalized(off["root"]/f"trial/request-{i}.json")!=normalized(on["root"]/f"trial/request-{i}.json") or not np.array_equal(x,y):
            raise ValueError("matched prefetch changed actual request or full RGBA output")
        checks.append(quality(x,y))
    return dict(off=off["root"].name,on=on["root"].name,rgba_exact=True,quality=checks,
                hot_reduction=1-on["receipt"]["hot_median_seconds"]/off["receipt"]["hot_median_seconds"])


def frozen(value,root):
    if len(value["images"])!=3:raise ValueError("frozen regression requires complete repeat requests")
    checks=[]
    for i,x in enumerate(value["images"]):
        if normalized(value["root"]/f"trial/request-{i}.json")!=normalized(root/f"trial/request-{i}.json"):
            raise ValueError("frozen original actual source request differs")
        checks.append(bool(np.array_equal(x,pixels(root/f"trial/{i}.png"))))
    if not all(checks):raise ValueError("disabled prefetch changed frozen original full RGBA output")
    return dict(cohort=root.name,requests_exact=True,rgba_exact=checks,summary_sha256=sha256_file(root/"summary.json"),
                scope="historical original request/pixels only; not an old timing denominator or current native qualification")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("library","native-source-archive","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--probe",type=Path,action="append",default=[])
    p.add_argument("--pair",nargs=3,metavar=("LABEL","OFF","ON"),action="append",default=[])
    p.add_argument("--window",nargs=2,metavar=("OFF_OR_ON","ROOT"),action="append",default=[])
    p.add_argument("--frozen",nargs=3,metavar=("LABEL","CURRENT","HISTORICAL"),action="append",default=[])
    args=p.parse_args()
    if args.output.exists():p.error("fresh audit output required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text());joint_audit.archive_contract(args.native_source_archive,manifest)
    probes=[synthetic(root,args.library,manifest) for root in args.probe];cohorts={};pairs=[]
    for label,left,right in args.pair:
        off=window(Path(left),args.library,manifest,False);on=window(Path(right),args.library,manifest,True)
        cohorts[left]=off;cohorts[right]=on;pairs.append(dict(label=label,**pair(off,on)))
    for mode,path in args.window:
        if mode not in ("off","on"):raise ValueError("explicit window mode required")
        cohorts[path]=window(Path(path),args.library,manifest,mode=="on")
    # Compare the same recipe and request position. In particular edit prefix
    # miss versus hit changes first-step QKV precision; equal public requests
    # alone do not make those two arithmetic routes equivalent.
    replays=0
    for value in cohorts.values():
        for other in cohorts.values():
            keys=("private_qkv_prefetch","case","scenario","runtime_prefix_snapshot","runtime_text_prefix_snapshot","text_prefix_qkv_step0")
            if value is other or any(value["summary"][k]!=other["summary"][k] for k in keys):continue
            if len(value["images"])!=len(other["images"]):raise ValueError("same-recipe replay is incomplete")
            for i,(x,y) in enumerate(zip(value["images"],other["images"])):
                if normalized(value["root"]/f"trial/request-{i}.json")!=normalized(other["root"]/f"trial/request-{i}.json") or not np.array_equal(x,y):
                    raise ValueError("reverse/timeline replay changed same-recipe actual request pixels")
                replays+=1
    frozen_checks={}
    for label,current,historical in args.frozen:
        if current not in cohorts or cohorts[current]["summary"]["private_qkv_prefetch"]:raise ValueError("frozen control must select an audited current prefetch-off cohort")
        frozen_checks[label]=frozen(cohorts[current],Path(historical))
    args.output.mkdir(parents=True)
    result=dict(schema="tc-qkv-lookahead-audit-v1",status="complete",qualification_passed=False,native_library_sha256=sha256_file(args.library),
        runtime_build_id=manifest["runtime_build_id"],native_source_archive_sha256=sha256_file(args.native_source_archive),script_sha256=sha256_file(Path(__file__)),
        probes=probes,cohorts=[v["receipt"] for v in cohorts.values()],pairs=pairs,exact_replay_comparisons=replays,frozen_regression=frozen_checks,
        scope="Full40 original-base stock wall and exact incremental RGBA; diagnostics separate. Not universal quality, exclusive-device, physical INT8/overlap or driver recovery qualification.")
    write_json(args.output/"audit.json",result);print(json.dumps(dict(status="complete",probes=len(probes),pairs=pairs,cohorts=len(cohorts)),indent=2))


if __name__=="__main__":main()
