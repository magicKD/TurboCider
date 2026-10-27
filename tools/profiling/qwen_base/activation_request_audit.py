#!/usr/bin/env python3
"""CPU matched complete-request activation audit against the current joint route."""
import argparse
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image,ImageDraw

from audit import normalized,pixels,quality
import collect
import joint_audit
from collect import sha256_file,write_json
from projection_request_audit import frozen_gpu_report_contract
from quality_audit import matched_request


def contract(value,activation):
    s=value["summary"]
    if (activation not in collect.FFN_ACTIVATIONS or s.get("ffn_activation")!=activation or
            s.get("ffn_projection")!="descriptor" or s.get("fixed_program") is not True or s.get("private_qkv") is not True or
            s.get("channels")!=6144 or s.get("resident_prefix") is not False or s.get("case") not in ("generate","edit") or
            s.get("mode") not in ("stock","timeline") or s.get("runtime_prefix_snapshot") is not (s["case"]=="edit") or
            s.get("environment")!=collect.environment("runtime",s["mode"],6144,s["case"]=="edit",False,True,True,"descriptor",activation)):
        raise ValueError("explicit fixed descriptor6144/QKV512 activation and matched edit-prefix recipe required")


def matched(a,b,activation):
    if activation not in collect.FFN_ACTIVATIONS[1:]:raise ValueError("explicit non-default activation required")
    contract(a,"exp");contract(b,activation)
    x,y=a["summary"],b["summary"]
    if x["case"]!=y["case"] or x["mode"]!="stock" or y["mode"]!="stock":raise ValueError("matched stock case required")
    left,right=x["environment"],y["environment"]
    if set(left)!=set(right) or {k for k in left if left[k]!=right[k]}!={"TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION"}:
        raise ValueError("only captured activation may differ")
    if any(x["assets"][k]!=y["assets"][k] for k in ("library","manifest","reference")):raise ValueError("matched native/shape/reference assets differ")
    for i in range(3):
        if normalized(a["root"]/f"trial/request-{i}.json")!=normalized(b["root"]/f"trial/request-{i}.json"):
            raise ValueError("activation changed the actual prompt/seed/geometry/execution request")


def repeated(stock,repeat,activation,mode):
    contract(stock,activation);contract(repeat,activation)
    a,b=stock["summary"],repeat["summary"]
    if (a["mode"]!="stock" or b["mode"]!=mode or a["case"]!=b["case"] or
            any(a["assets"][k]!=b["assets"][k] for k in ("library","manifest","reference"))):raise ValueError("same recipe/assets/case replay required")
    for i in range(3):
        if (normalized(stock["root"]/f"trial/request-{i}.json")!=normalized(repeat["root"]/f"trial/request-{i}.json") or
                not np.array_equal(stock["images"][i],repeat["images"][i])):raise ValueError("same recipe did not restore complete request/PNG")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("baseline","candidate","library","native-source-archive","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--activation",choices=collect.FFN_ACTIVATIONS[1:],required=True)
    parser.add_argument("--case",choices=("generate","edit"),required=True)
    parser.add_argument("--timeline",type=Path);parser.add_argument("--reverse-baseline",type=Path)
    parser.add_argument("--historical-exp",type=Path,help="old same-arithmetic request/PNG control only, never a timing denominator")
    parser.add_argument("--gpu-reference",type=Path,help="frozen matched GPU trial for quality only")
    parser.add_argument("--tool-source-archive",type=Path,action="append",default=[])
    args=parser.parse_args()
    if args.output.exists():parser.error("fresh complete-request CPU audit directory required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    joint_audit.archive_contract(args.native_source_archive,manifest);historical={}
    for path in args.tool_source_archive:
        with tarfile.open(path,"r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():historical.setdefault(member.name,set()).add(archive.extractfile(member).read())
    def load(root,activation,mode="stock"):
        return joint_audit.window(root,args.library,manifest,True,mode,args.case,args.case=="edit",historical,
            fixed_program=True,ffn_activation=activation)
    a,b=load(args.baseline,"exp"),load(args.candidate,args.activation);matched(a,b,args.activation)
    receipts=[a["receipt"],b["receipt"]]
    for root,stock,activation,mode in ((args.timeline,b,args.activation,"timeline"),(args.reverse_baseline,a,"exp","stock")):
        if root is not None:
            value=load(root,activation,mode);repeated(stock,value,activation,mode);receipts.append(value["receipt"])
    historical_exp=None
    if args.historical_exp:
        report=json.loads((args.historical_exp/"report.json").read_text())
        if report.get("status")!="complete" or len(report.get("runs",[]))!=3:raise ValueError("complete frozen original activation control required")
        collect.validate([r["metrics"] for r in report["runs"]],"runtime","stock",6144,args.case=="edit",False,True,True)
        for i in range(3):
            if (normalized(args.historical_exp/f"request-{i}.json")!=normalized(a["root"]/f"trial/request-{i}.json") or
                    not np.array_equal(pixels(args.historical_exp/f"{i}.png"),a["images"][i])):raise ValueError("new default changed frozen request/PNG")
        historical_exp=dict(rgba_exact=True,report_sha256=sha256_file(args.historical_exp/"report.json"),scope="PNG-only compatibility, not old timing")
    frozen=None;frozen_receipt=None
    if args.gpu_reference:
        report=json.loads((args.gpu_reference/"report.json").read_text());frozen_gpu_report_contract(report)
        collect.validate([r["metrics"] for r in report["runs"]],"gpu","stock")
        for i in range(3):matched_request(normalized(args.gpu_reference/f"request-{i}.json"),normalized(a["root"]/f"trial/request-{i}.json"))
        frozen={name:[quality(pixels(args.gpu_reference/f"{i}.png"),value["images"][i]) for i in range(3)] for name,value in (("exp",a),(args.activation,b))}
        frozen_receipt=dict(report_sha256=sha256_file(args.gpu_reference/"report.json"),png_sha256=[sha256_file(args.gpu_reference/f"{i}.png") for i in range(3)])
    result=dict(schema="tc-activation-complete-request-audit-v1",status="complete",qualification_passed=False,
        case=args.case,activation=args.activation,runtime_build_id=manifest["runtime_build_id"],library_sha256=sha256_file(args.library),
        native_source_archive_sha256=sha256_file(args.native_source_archive),script_sha256=sha256_file(Path(__file__)),cohorts=receipts,
        stock_hot_wall_reduction=1-b["receipt"]["hot_median_seconds"]/a["receipt"]["hot_median_seconds"],
        incremental_quality=[quality(a["images"][i],b["images"][i]) for i in range(3)],frozen_gpu_quality=frozen,
        frozen_gpu_receipt=frozen_receipt,historical_exp=historical_exp,
        limits=["Small nonexclusive-device pilot, not thermal/statistical or semantic image qualification.",
                "Approximate W8/A8/FP16 arithmetic; operator-only diagnostics are not complete model accuracy or speed.",
                "Timeline is diagnostic, not stock performance. Client selectors do not prove physical INT8 or device overlap."])
    args.output.mkdir(parents=True);sheet=Image.new("RGB",(1024,3*544),"white");draw=ImageDraw.Draw(sheet)
    for i in range(3):
        for j,(name,value) in enumerate((("exp",a),(args.activation,b))):
            draw.text((j*512+8,i*544+8),f"FFN {name} + QKV512, seed {(73,74,73)[i]}, request{i}",fill="black")
            sheet.paste(Image.fromarray(value["images"][i][:,:,:3]),(j*512,i*544+32))
    sheet.save(args.output/"comparison.png");write_json(args.output/"audit.json",result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
