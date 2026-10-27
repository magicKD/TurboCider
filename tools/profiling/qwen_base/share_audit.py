#!/usr/bin/env python3
"""CPU-only fixed FFN share comparison while QKV, steps and source stay fixed."""
import argparse
import json
import math
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image, ImageDraw

from audit import normalized, pixels, quality
import collect
import joint_audit
from collect import ROOT, sha256_file, write_json


COMPLETION_COUNTERS = (
    ("ffn_prediction_wait_seconds", "hybrid", "model_prediction_seconds_session_total", False),
    ("ffn_output_completion_wait_seconds", "hybrid", "output_handling_seconds_session_total", False),
    ("ffn_output_bytes", "hybrid", "output_copy_bytes_session_total", True),
    ("ffn_predictions", "hybrid", "runtime_calls_session_total", True),
    ("ffn_caller_finish_wait_seconds", "runtime_weight", "async_ane_wait_seconds_session_total", False),
    ("ffn_activation_wait_seconds", "runtime_weight", "a8_wait_seconds_session_total", False),
    ("ffn_pre_ffn_seconds", "runtime_weight", "pre_ffn_seconds_session_total", False),
    ("ffn_post_join_seconds", "runtime_weight", "post_join_seconds_session_total", False),
    ("qkv_join_seconds", "qwen_private_qkv", "join_seconds_session_total", False),
    ("qkv_stage_lifetime_seconds", "qwen_private_qkv", "stage_lifetime_seconds_session_total", False),
)
COMPLETION_SCOPE = (
    "Per-request differences of one graph-owner's cumulative counters, not cumulative values treated as requests. "
    "FFN prediction wait brackets request.finish(); output wait brackets the following io.finish() on the worker; "
    "caller finish waits on the broader async worker path. These overlapping host wait spans must not be summed "
    "or interpreted as physical ANE compute, GPU kernel time, transfer bandwidth, or a unique queue-contention cause."
)


def completion_deltas(runs):
    """Expose the existing worker/caller completion split without changing capture."""
    if not runs:
        raise ValueError("complete ordered requests required for completion differences")
    previous = {name: 0 for name, *_ in COMPLETION_COUNTERS}
    results = []
    for index, row in enumerate(runs):
        if type(row.get("index")) is not int or row["index"] != index:
            raise ValueError("contiguous actual request indices required")
        try:
            hybrid = row["metrics"]["hybrid"]
            groups = dict(hybrid=hybrid, runtime_weight=hybrid["runtime_weight"],
                          qwen_private_qkv=row["metrics"]["qwen_private_qkv"])
            current = {name: groups[group][field] for name, group, field, _ in COMPLETION_COUNTERS}
        except (KeyError, TypeError) as error:
            raise ValueError("complete native completion counters required") from error
        for name, _, _, integer in COMPLETION_COUNTERS:
            value = current[name]
            if (type(value) not in (int, float) or (integer and type(value) is not int) or
                    not math.isfinite(value) or value < previous[name]):
                raise ValueError("completion counters reset/regressed, changed type, or are nonfinite")
        results.append(dict(index=index, **{name: value - previous[name] for name, value in current.items()}))
        previous = current
    return results


def recipe_contract(summary,channels,mode,*,fixed_program=False,activation="exp"):
    if (activation not in collect.FFN_ACTIVATIONS or (activation!="exp" and not fixed_program) or
            summary.get("fixed_program",False) is not fixed_program or summary.get("ffn_activation","exp")!=activation or
            summary.get("ffn_projection","descriptor")!="descriptor" or summary["channels"]!=channels or
            summary.get("mode","stock")!=mode):
        raise ValueError("explicit matched fixed-program/activation/descriptor recipe required")
    if fixed_program:
        expected=collect.environment("runtime",mode,channels,summary["case"]=="edit",False,True,True,"descriptor",activation)
        if summary["environment"]!=expected:
            raise ValueError("complete fixed share environment differs from the declared recipe")


def matched(a,b,*,fixed_program=False,activation="exp"):
    left,right=a["summary"],b["summary"]
    recipe_contract(left,5120,"stock",fixed_program=fixed_program,activation=activation)
    recipe_contract(right,6144,"stock",fixed_program=fixed_program,activation=activation)
    if (left["case"]!=right["case"] or left["runtime_prefix_snapshot"] is not right["runtime_prefix_snapshot"] or
            left.get("private_qkv") is not True or right.get("private_qkv") is not True or
            left["channels"]!=5120 or right["channels"]!=6144):
        raise ValueError("ordered joint5120/6144 matched cases required")
    x,y=left["environment"],right["environment"]
    if set(x)!=set(y) or {k for k in x if x[k]!=y[k]}!={"TURBOCIDER_PRIVATE_ANE_CHANNELS"}:
        raise ValueError("only the fixed FFN channel share may differ")
    if any(left["assets"][k]!=right["assets"][k] for k in ("library","manifest","reference")):
        raise ValueError("matched native/checkpoint recipe/reference assets differ")
    for i in range(3):
        if normalized(a["root"]/f"trial/request-{i}.json")!=normalized(b["root"]/f"trial/request-{i}.json"):
            raise ValueError("share comparison changed an actual request")


def replay(base,value,mode,*,fixed_program=False,activation="exp"):
    a,b=base["summary"],value["summary"];channels=a["channels"]
    recipe_contract(a,channels,"stock",fixed_program=fixed_program,activation=activation)
    recipe_contract(b,channels,mode,fixed_program=fixed_program,activation=activation)
    if a["case"]!=b["case"] or any(a["assets"][k]!=b["assets"][k] for k in ("library","manifest","reference")):
        raise ValueError("same-case same-recipe replay assets required")
    for i in range(3):
        if (normalized(base["root"]/f"trial/request-{i}.json")!=normalized(value["root"]/f"trial/request-{i}.json") or
                not np.array_equal(base["images"][i],value["images"][i])):
            raise ValueError("same share reverse/timeline did not restore actual requests and complete PNGs")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("c5120","c6144","library","native-source-archive","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--case",choices=("generate","edit"),required=True)
    parser.add_argument("--fixed-program",action="store_true")
    parser.add_argument("--ffn-activation",choices=collect.FFN_ACTIVATIONS,default="exp")
    parser.add_argument("--timeline5120",type=Path)
    parser.add_argument("--reverse5120",type=Path)
    parser.add_argument("--reverse6144",type=Path)
    parser.add_argument("--gpu-reference",type=Path)
    parser.add_argument("--tool-source-archive",type=Path,action="append",default=[])
    args=parser.parse_args()
    if args.ffn_activation!="exp" and not args.fixed_program:parser.error("non-default activation requires fixed-program")
    if args.output.exists():parser.error("fresh audit directory required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    joint_audit.archive_contract(args.native_source_archive,manifest)
    historical={}
    for path in args.tool_source_archive:
        with tarfile.open(path,"r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():historical.setdefault(member.name,set()).add(archive.extractfile(member).read())
    def load(root,channels,mode="stock"):
        value=joint_audit.window(root,args.library,manifest,True,mode,args.case,args.case=="edit",historical,channels=channels,
            fixed_program=args.fixed_program,ffn_activation=args.ffn_activation)
        value["receipt"]["completion_deltas"]=completion_deltas(value["report"]["runs"])
        return value
    a,b=load(args.c5120,5120),load(args.c6144,6144)
    matched(a,b,fixed_program=args.fixed_program,activation=args.ffn_activation)
    cohorts=[a["receipt"],b["receipt"]]
    for root,channels,base in ((args.reverse5120,5120,a),(args.reverse6144,6144,b),(args.timeline5120,5120,a)):
        if root is None:continue
        mode="timeline" if root==args.timeline5120 else "stock"
        value=load(root,channels,mode)
        replay(base,value,mode,fixed_program=args.fixed_program,activation=args.ffn_activation)
        if channels==5120 and mode=="stock":matched(value,b,fixed_program=args.fixed_program,activation=args.ffn_activation)
        if channels==6144:matched(a,value,fixed_program=args.fixed_program,activation=args.ffn_activation)
        cohorts.append(value["receipt"])
    gpu=None
    if args.gpu_reference:
        gpu={str(channels):[quality(pixels(args.gpu_reference/f"{i}.png"),value["images"][i]) for i in range(3)]
             for channels,value in ((5120,a),(6144,b))}
    qa=[quality(b["images"][i],a["images"][i]) for i in range(3)]
    slower=b["receipt"]["hot_median_seconds"]
    candidate=a["receipt"]["hot_median_seconds"]
    result=dict(schema="tc-qwen-joint-fixed-share-audit-v2",status="complete",qualification_passed=False,
        case=args.case,fixed_program=args.fixed_program,ffn_activation=args.ffn_activation,library_sha256=sha256_file(args.library),runtime_build_id=manifest["runtime_build_id"],
        script_sha256=sha256_file(Path(__file__)),native_source_archive_sha256=sha256_file(args.native_source_archive),
        cohorts=cohorts,stock_5120_hot_wall_reduction_vs6144=1-candidate/slower,
        incremental_quality_5120_vs6144=qa,frozen_gpu_quality=gpu,completion_scope=COMPLETION_SCOPE,
        limits=["Small non-exclusive-device pilot; no statistical/thermal qualification.",
               "One fixed FFN graph per process, not changing shares through hot recompilation.",
               "W8 is approximate; pixel similarity does not establish semantic quality.",
               "API/CB spans are not hardware kernel time; overlapping spans must not be summed."])
    args.output.mkdir(parents=True)
    sheet=Image.new("RGB",(1024,3*544),"white");draw=ImageDraw.Draw(sheet)
    for i in range(3):
        for j,(channels,value) in enumerate(((6144,b),(5120,a))):
            draw.text((j*512+8,i*544+8),f"FFN{channels} + QKV512/{args.ffn_activation}, seed {(73,74,73)[i]}, request {i}",fill="black")
            sheet.paste(Image.fromarray(value["images"][i][:,:,:3]),(j*512,i*544+32))
    sheet.save(args.output/"comparison.png")
    write_json(args.output/"audit.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
