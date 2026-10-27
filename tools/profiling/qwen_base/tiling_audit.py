#!/usr/bin/env python3
"""CPU-only fixed MIL K-split comparison, not physical kernel tiling proof."""
import argparse
import json
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image, ImageDraw

from audit import normalized, pixels, quality
import joint_audit
from collect import sha256_file, write_json


def matched(a,b,requested_tile):
    left,right=a["summary"],b["summary"]
    if (left["case"]!=right["case"] or left["channels"]!=6144 or right["channels"]!=6144 or
            left.get("private_qkv") is not True or right.get("private_qkv") is not True or
            left["runtime_prefix_snapshot"] is not right["runtime_prefix_snapshot"] or left["environment"]!=right["environment"]):
        raise ValueError("fixed6144 joint environment/phase must remain identical")
    if any(left["assets"][k]!=right["assets"][k] for k in ("library","reference")):
        raise ValueError("native/reference assets differ")
    manifests=[]
    for value in (a,b):
        request=normalized(value["root"]/"trial/request-0.json")
        path=Path(request["ane_manifest"])
        if sha256_file(path)!=value["summary"]["assets"]["manifest"]:
            raise ValueError("actual source descriptor changed")
        manifests.append(json.loads(path.read_text()))
    x,y=manifests
    if (x.get("tile_k")!=4096 or y.get("tile_k")!=requested_tile or requested_tile not in (512,1024) or
            {k:v for k,v in x.items() if k!="tile_k"}!={k:v for k,v in y.items() if k!="tile_k"}):
        raise ValueError("only descriptor tile_k may differ from the4096/default source")
    for i in range(3):
        x=normalized(a["root"]/f"trial/request-{i}.json");y=normalized(b["root"]/f"trial/request-{i}.json")
        x.pop("ane_manifest");y.pop("ane_manifest")
        if x!=y:raise ValueError("K-split changed actual prompt/seed/geometry")
    return manifests


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("baseline","candidate","library","native-source-archive","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--case",choices=("generate","edit"),required=True)
    parser.add_argument("--tile",type=int,choices=(512,1024),required=True)
    parser.add_argument("--timeline",type=Path)
    parser.add_argument("--reverse-baseline",type=Path)
    parser.add_argument("--gpu-reference",type=Path)
    parser.add_argument("--tool-source-archive",type=Path,action="append",default=[])
    args=parser.parse_args()
    if args.output.exists():parser.error("fresh audit directory required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    joint_audit.archive_contract(args.native_source_archive,manifest)
    historical={}
    for path in args.tool_source_archive:
        with tarfile.open(path,"r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():historical.setdefault(member.name,set()).add(archive.extractfile(member).read())
    def load(root,mode="stock"):
        return joint_audit.window(root,args.library,manifest,True,mode,args.case,args.case=="edit",historical)
    a,b=load(args.baseline),load(args.candidate)
    descriptors=matched(a,b,args.tile)
    cohorts=[a["receipt"],b["receipt"]]
    for root,base,mode in ((args.timeline,b,"timeline"),(args.reverse_baseline,a,"stock")):
        if root is None:continue
        value=load(root,mode)
        if any(not np.array_equal(x,y) for x,y in zip(value["images"],base["images"])):
            raise ValueError("timeline/reverse changed same-recipe PNGs")
        cohorts.append(value["receipt"])
    absolute=None
    if args.gpu_reference:absolute=[quality(pixels(args.gpu_reference/f"{i}.png"),b["images"][i]) for i in range(3)]
    result=dict(schema="tc-qwen-joint-fixed-k-split-audit-v1",status="complete",qualification_passed=False,
        case=args.case,library_sha256=sha256_file(args.library),script_sha256=sha256_file(Path(__file__)),
        native_source_archive_sha256=sha256_file(args.native_source_archive),runtime_build_id=manifest["runtime_build_id"],
        descriptors=descriptors,mil_k_splits_from_pinned_source=[2048,args.tile],cohorts=cohorts,
        stock_hot_wall_reduction=1-b["receipt"]["hot_median_seconds"]/a["receipt"]["hot_median_seconds"],
        incremental_quality=[quality(a["images"][i],b["images"][i]) for i in range(3)],frozen_gpu_quality=absolute,
        limits=["MIL projection splitting is not a measurement of physical ANE kernel tiling or INT8 arithmetic.",
                "One fixed program per process; no hot tile/share switching. Approximate reduction order can change images.",
                "Small non-exclusive-device pilot; no statistical, thermal or semantic quality qualification."])
    args.output.mkdir(parents=True)
    sheet=Image.new("RGB",(1024,3*544),"white");draw=ImageDraw.Draw(sheet)
    for i in range(3):
        for j,(tile,value) in enumerate(((2048,a),(args.tile,b))):
            draw.text((j*512+8,i*544+8),f"MIL Ksplit{tile}, joint6144+512, request{i}",fill="black")
            sheet.paste(Image.fromarray(value["images"][i][:,:,:3]),(j*512,i*544+32))
    sheet.save(args.output/"comparison.png")
    write_json(args.output/"audit.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
