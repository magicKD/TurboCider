#!/usr/bin/env python3
"""CPU full-base multi-prompt pixel/provenance audit; no semantic pass claim."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile

import numpy as np
from PIL import Image, ImageDraw

from audit import normalized, pixels, quality
import collect
import joint_audit
from collect import ROOT, model_snapshot, sha256_file, write_json


def matched_request(a,b):
    # These are explicit execution choices, not prompt/seed/geometry changes.
    ignored={"output","execution","ane_manifest","hybrid_mlp_mode"}
    if {k:v for k,v in a.items() if k not in ignored}!={k:v for k,v in b.items() if k not in ignored}:
        raise ValueError("quality controls must retain the exact prompt/seed/reference/geometry")
    if (a.get("execution")!="gpu" or b.get("execution")!="gpu_ane" or
            b.get("hybrid_mlp_mode")!="runtime" or not b.get("ane_manifest")):
        raise ValueError("ordinary GPU versus explicit runtime joint control required")


def window(root,library,manifest,channels,hybrid,historical,activation="exp"):
    root=Path(root)
    summary=json.loads((root/"summary.json").read_text())
    report=json.loads((root/"trial/report.json").read_text())
    route="runtime" if hybrid else "gpu"
    actual_activation=activation if hybrid else "exp"
    fixed=hybrid and activation!="exp"
    if (summary["status"]!="complete" or report["status"]!="complete" or summary["mode"]!="stock" or
            summary["scenario"]!="repeat" or summary["route"]!=route or
            summary.get("private_qkv") is not hybrid or report.get("private_qkv") is not hybrid or
            summary.get("ffn_activation","exp")!=actual_activation or report.get("ffn_activation","exp")!=actual_activation or
            summary.get("fixed_program",False) is not fixed or report.get("fixed_program",False) is not fixed or
            summary["channels"]!=(channels if hybrid else 4096) or summary.get("runtime_prefix_snapshot") or
            summary.get("resident_prefix") or len(report["runs"])!=1 or summary["requests"]!=1 or
            summary["assets"]["library"]!=sha256_file(library) or
            report["loaded_runtime_id"]!=manifest["runtime_build_id"]):
        raise ValueError("complete original-base single-request stock quality window required")
    if summary["model_snapshot"]!=model_snapshot(ROOT/"models/Comfy-Org-Qwen-Image-2.1"):
        raise ValueError("original checkpoint generation changed")
    snapshot=json.loads((root/"source.json").read_text())
    if set(snapshot)!=set(summary["source_sha256"]):raise ValueError("complete source snapshot required")
    for name,digest in summary["source_sha256"].items():
        raw=snapshot[name].encode()
        if hashlib.sha256(raw).hexdigest()!=digest or not (raw==(ROOT/name).read_bytes() or raw in historical.get(name,set())):
            raise ValueError("profiling current/archive source differs")
    metrics=[r["metrics"] for r in report["runs"]]
    collect.validate(metrics,route,"stock",channels if hybrid else 4096,private_qkv=hybrid,fixed_program=fixed,ffn_activation=actual_activation)
    request=normalized(root/"trial/request-0.json")
    metric=metrics[0]
    case=summary["case"]
    if (request.get("operation")!="image."+case or metric.get("operation")!=request["operation"] or
            request.get("prompt")!=summary["prompt"] or not request.get("prompt","").strip() or
            request.get("seed")!=73 or metric.get("seed")!=73 or request.get("loras") or request.get("audio") is not False or
            (case=="generate" and request.get("inputs")) or
            (case=="edit" and (len(request.get("inputs",[]))!=1 or request.get("qwen21_reference_size")!=1024))):
        raise ValueError("actual base prompt/seed/operation/standard-reference contract differs")
    if hybrid:
        phases=metric["qwen_ffn_phases"]
        prefill,decode=phases["prefill"],phases["decode"]
        if (phases["policy"]!="all" or prefill["steps_this_request"]!=1 or decode["steps_this_request"]!=39 or
                prefill["actual_rows"]!=1024+metric["text_tokens"]+metric["reference_tokens"] or
                decode["actual_rows"]!=1024 or metric["hybrid"]["runtime_calls_session_total"]!=
                32*(39+math.ceil(prefill["actual_rows"]/1024))):
            raise ValueError("actual variable-prompt sequence and all FFN chunks required")
    for name in ("report","evidence"):
        if sha256_file(root/summary["memory"][name])!=summary["memory"][name+"_sha256"]:
            raise ValueError("process memory evidence digest changed")
    memory=summary["memory"]["verified"]
    if not memory["complete"] or memory["command_exit_code"]!=0:
        raise ValueError("complete successful owned process evidence required")
    image=pixels(root/"trial/0.png")
    return dict(root=root,summary=summary,request=request,image=image,receipt=dict(cohort=root.name,
        route=route,channels=channels if hybrid else None,first_request_seconds=report["runs"][0]["request_wall_including_export"],
        summary_sha256=sha256_file(root/"summary.json"),png_sha256=sha256_file(root/"trial/0.png"),
        actual_denoise_steps=metric["actual_denoise_steps"],text_tokens=metric["text_tokens"],reference_tokens=metric["reference_tokens"],
        private_qkv=metric.get("qwen_private_qkv"),runtime_calls=(metric.get("hybrid") or {}).get("runtime_calls_session_total"),
        ffn_activation=actual_activation,fixed_program=fixed,
        memory=memory,load_error=summary.get("load_error")))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair",nargs=3,action="append",required=True,metavar=("LABEL","GPU_DIR","JOINT_DIR"))
    parser.add_argument("--channels",type=int,choices=(5120,6144),required=True)
    parser.add_argument("--ffn-activation",choices=collect.FFN_ACTIVATIONS,default="exp")
    for name in ("library","native-source-archive","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--tool-source-archive",type=Path,action="append",default=[])
    args=parser.parse_args()
    if args.output.exists():parser.error("fresh audit directory required")
    labels=[p[0] for p in args.pair]
    if len(set(labels))!=len(labels) or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,64}",label) for label in labels):
        parser.error("unique short safe labels required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    joint_audit.archive_contract(args.native_source_archive,manifest)
    historical={}
    for path in args.tool_source_archive:
        with tarfile.open(path,"r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():historical.setdefault(member.name,set()).add(archive.extractfile(member).read())
    records=[];sheets=[]
    for label,gpu_path,joint_path in args.pair:
        a=window(gpu_path,args.library,manifest,args.channels,False,historical,args.ffn_activation)
        b=window(joint_path,args.library,manifest,args.channels,True,historical,args.ffn_activation)
        matched_request(a["request"],b["request"])
        if any(a["summary"]["assets"][key]!=b["summary"]["assets"][key] for key in ("library","manifest","reference")):
            raise ValueError("actual native/reference assets differ")
        comparison=quality(a["image"],b["image"])
        records.append(dict(label=label,request=a["request"],gpu=a["receipt"],joint=b["receipt"],quality=comparison,
                            semantic_qualification="not established by pixel metrics; inspect the original prompt and both images"))
        sheet=Image.new("RGB",(1024,544),"white");draw=ImageDraw.Draw(sheet)
        for j,(title,value) in enumerate((("Full GPU",a),(f"Joint FFN{args.channels}+QKV512/{args.ffn_activation}",b))):
            draw.text((j*512+8,8),f"{label}: {title}, seed73",fill="black")
            sheet.paste(Image.fromarray(value["image"][:,:,:3]),(j*512,32))
        sheets.append((label,sheet))
    keys={r["joint"]["private_qkv"]["program_key"] for r in records}
    if len(keys)!=1:raise ValueError("fixed QKV program key changed across prompts")
    args.output.mkdir(parents=True)
    for label,sheet in sheets:sheet.save(args.output/f"{label}.png")
    result=dict(schema="tc-qwen-base-joint-multiprompt-quality-v1",status="complete",qualification_passed=False,
        channels=args.channels,ffn_activation=args.ffn_activation,library_sha256=sha256_file(args.library),runtime_build_id=manifest["runtime_build_id"],
        script_sha256=sha256_file(Path(__file__)),native_source_archive_sha256=sha256_file(args.native_source_archive),
        records=records,limits=["One first request per process/prompt; no hot median or steady-state speed claim.",
            "Pixel similarity does not prove text accuracy, identity/material preservation or overall image quality.",
            "Private graph/input types do not prove physical INT8 or GPU/ANE hardware overlap."])
    write_json(args.output/"audit.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
