#!/usr/bin/env python3
"""CPU-only full-array recheck of bounded QKV numerical counterfactuals."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import tarfile

import numpy as np

from flux9_bf16_framework_screen import write_json
from qwen_qkv_lora_block_audit import difference,same_quality
import qwen_qkv_numeric_screen as screen
from runtime_ane_common import sha256_file

ROOT=Path(__file__).resolve().parents[2]


def arrays(path,expected):
    with path.open("rb") as stream:
        length=struct.unpack("<Q",stream.read(8))[0]
        if not 2<=length<=1<<20:raise ValueError("bounded stage header required")
        header=json.loads(stream.read(length));header.pop("__metadata__",None)
    if set(header)!=set(expected):raise ValueError("all exact stage consumers required")
    result={};regions=[]
    for name,value in header.items():
        shape=expected[name];a,b=value["data_offsets"];count=int(np.prod(shape))
        if value["dtype"]!="BF16" or value["shape"]!=shape or type(a) is not int or type(b) is not int or a<0 or b-a!=count*2:
            raise ValueError("complete original BF16 stage ABI required")
        regions.append((a,b));result[name]=np.memmap(path,dtype="<u2",mode="r",offset=8+length+a,shape=tuple(shape))
    regions.sort()
    if regions[0][0]!=0 or any(a[1]!=b[0] for a,b in zip(regions,regions[1:])) or 8+length+regions[-1][1]!=path.stat().st_size:
        raise ValueError("overlapping/gapped/truncated stage payload")
    return result


def geometry(rope):
    shape=[1,32,1024,128];expected={"normalized_input":[1,1024,4096]}
    expected.update({f"reference-{stage}-{i}":shape for stage in range(4) for i in range(3)})
    for recipe in screen.RECIPES:
        expected.update({f"{recipe}-{stage}-{i}":shape for stage in range(4) for i in range(3)})
        expected.update({f"{recipe}-matched-{side}-{i}":shape for side in ("norm","rope") for i in range(3)})
        if rope:expected.update({f"{recipe}-no-contract-{i}":shape for i in range(3)})
    return expected


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--library",type=Path,required=True)
    p.add_argument("--native-source-archive",type=Path,required=True);p.add_argument("--cohort",type=Path,action="append",required=True)
    p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error("fresh audit output required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text());sources=manifest["inputs"]["sources"]
    current_matches=True
    with tarfile.open(args.native_source_archive,"r:gz") as archive:
        members={item.name:item for item in archive.getmembers()}
        if set(members)!={item["path"] for item in sources}:raise ValueError("complete linked native source archive required")
        for item in sources:
            stream=archive.extractfile(members[item["path"]])
            if stream is None or hashlib.sha256(stream.read()).hexdigest()!=item["sha256"]:raise ValueError("native source archive differs from linked manifest")
            current_matches &= sha256_file(ROOT/item["path"])==item["sha256"]
    observations=[];layer_sources=None;inputs={}
    for root in args.cohort:
        s=json.loads((root/"summary.json").read_text());n=json.loads((root/"trial/native.json").read_text());snapshot=json.loads((root/"source.json").read_text())
        if s["status"]!="complete" or s["asset_sha256"]["library"]!=sha256_file(args.library) or n["runtime_build_id"]!=manifest["runtime_build_id"]:
            raise ValueError("complete same-library numeric window required")
        for name,digest in s["source_sha256"].items():
            if hashlib.sha256(snapshot[name].encode()).hexdigest()!=digest:raise ValueError("bound stage-probe source changed")
        actual=screen.analyze(n)
        if actual!=s["analysis"]:raise ValueError("counterfactual summary or full-block denominator differs")
        rope=n.get("numeric_recipe")=="rope-contract-v2";values=arrays(root/"trial/stage-outputs.safetensors",geometry(rope));raw=[]
        for row in n["stage_quality"]:
            label,stage,i=row["recipe"],row["stage"],row["projection"]
            if label.endswith("-matched-norm") or label.endswith("-matched-rope"):
                suffix="-matched-norm" if stage==2 else "-matched-rope"
                ref=f"{label}-{i}";candidate=f"{label.removesuffix(suffix)}-{stage}-{i}"
            elif label.endswith("-no-contract"):ref=f"reference-3-{i}";candidate=f"{label}-{i}"
            else:ref=f"reference-{stage}-{i}";candidate=f"{label}-{stage}-{i}"
            q=difference(values[ref],values[candidate]);bit_changed=q["changed"]
            # The native report compares numeric F32 values. Record bitwise
            # differences separately instead of silently canonicalizing -0.
            if bit_changed:
                a=(values[ref].astype(np.uint32)<<16).view(np.float32);b=(values[candidate].astype(np.uint32)<<16).view(np.float32)
                q["changed"]=int(np.count_nonzero(a!=b))
            same_quality(q,row["quality"]);raw.append(dict(recipe=label,stage=stage,projection=i,quality=q,bit_changed=bit_changed))
        names=screen.ROPE_NAMES if rope else screen.NAMES
        blocks=arrays(root/"trial/block-outputs.safetensors",{name:[1,1024,4096] for name in ("reference",)+names});block_rows=[]
        for name in names:
            q=difference(blocks["reference"],blocks[name]);same_quality(q,[row for row in n["samples"] if row["arm"]==name][-1]["quality"])
            if name in screen.NAMES[:2] and q["changed"]!=0:raise ValueError("raw original GPU control is not bitwise exact")
            block_rows.append(dict(arm=name,quality=q))
        input_hash=hashlib.sha256(values["normalized_input"].tobytes()).hexdigest()
        if s["case"] in inputs and inputs[s["case"]]!=input_hash:raise ValueError("same-task normalized actual input differs")
        inputs[s["case"]]=input_hash
        if layer_sources is None:layer_sources=n["actual_layer_source_sha256"]
        elif layer_sources!=n["actual_layer_source_sha256"]:raise ValueError("layer0 original base/factor sources changed")
        memory=s["memory"]
        for key in ("evidence","report"):
            if sha256_file(root/memory[key])!=memory[key+"_sha256"]:raise ValueError("memory evidence changed")
        if not memory["verified"]["complete"] or memory["verified"]["command_exit_code"]!=0:raise ValueError("incomplete process-tree window")
        observations.append(dict(cohort=root.name,case=s["case"],analysis=actual,raw_stage_quality=raw,raw_block_quality=block_rows,
            normalized_input_sha256=input_hash,memory=memory["verified"],load_error=s.get("load_error"),summary_sha256=sha256_file(root/"summary.json")))
    result=dict(schema="tc-qwen-qkv-numeric-audit-v1",status="complete",qualification_passed=False,
        library_sha256=sha256_file(args.library),runtime_build_id=manifest["runtime_build_id"],native_sources_verified=len(sources),
        native_archive_sha256=sha256_file(args.native_source_archive),current_native_matches_archive=bool(current_matches),
        actual_layer_source_hash_count=len(layer_sources or {}),observations=observations,script_sha256=sha256_file(Path(__file__)),
        scope="CPU/full-array source-and-counterfactual recheck; local layer0 evidence, no unique whole-image cause/performance qualification")
    write_json(args.output,result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
