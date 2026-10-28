#!/usr/bin/env python3
"""Independent full-array/payload/selector audit of real FFN projection blocks."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import tarfile

import numpy as np

import ffn_projection_collect as screen
import joint_audit
from collect import ROOT,model_snapshot,sha256_file,write_json
from qkv_audit import arrays,quality
from vpipe_audit import trace


def factorial(records):
    """First four stock windows form the explicit 2x2, later rows are replays."""
    policies=("descriptor","up-single","down-single","single")
    if len(records)<4 or tuple(r["projection"] for r in records[:4])!=policies:
        raise ValueError("ordered descriptor/up-single/down-single/single factorial required")
    primary=dict(zip(policies,records[:4]));anchor=records[0]
    for row in records:
        if row["projection"] not in policies or row.get("activation","exp")!="exp" or row["case"]!=anchor["case"]:
            raise ValueError("same captured real source and explicit factorial policy required")
        if any(row["array_sha256"][key]!=anchor["array_sha256"][key] for key in
               ("ffn_input","block_reference","alternate_reference","other_weight_reference")):
            raise ValueError("factorial source inputs and full GPU controls changed")
    if any(row["mode"]!="stock" or row["analysis"]["measured_blocks"]!=anchor["analysis"]["measured_blocks"] for row in records[:4]):
        raise ValueError("equal-repeat stock factorial windows required")
    for row in records[4:]:
        if row["array_sha256"]!=primary[row["projection"]]["array_sha256"]:
            raise ValueError("same-policy repeat/timeline changed full arrays")
    medians={key:row["analysis"]["median_seconds"]["private-restaged"] for key,row in primary.items()}
    base,up,down,both=(medians[key] for key in policies)
    return dict(median_seconds=medians,up_only_delta_seconds=up-base,down_only_delta_seconds=down-base,
        both_delta_seconds=both-base,interaction_seconds=both-up-down+base,
        identical_source_and_gpu_controls=True,same_policy_replays_exact=True,
        replay_cohorts=[r["cohort"] for r in records[4:]],
        scope="observed complete-block differences across nonexclusive stock windows, not ANE operator hardware attribution or full-image speed")


def same_capture(records):
    if not records:raise ValueError("actual complete block capture records required")
    anchor=records[0];seen={}
    for row in records:
        if row["case"]!=anchor["case"] or any(row["array_sha256"][key]!=anchor["array_sha256"][key] for key in
                ("ffn_input","block_reference","alternate_reference","other_weight_reference")):
            raise ValueError("matched activation source inputs and ordinary full GPU controls required")
        key=(row["projection"],row.get("activation","exp"))
        if key in seen and row["array_sha256"]!=seen[key]["array_sha256"]:
            raise ValueError("same activation/projection recipe did not restore all complete arrays")
        seen.setdefault(key,row)
    return dict(identical_source_and_gpu_controls=True,same_recipe_replays_exact=True,
        recipes=[dict(projection=key[0],activation=key[1],first_cohort=row["cohort"]) for key,row in seen.items()])


def original_hashes(model):
    path=model/"diffusion_models/qwen_image_2.1_bf16.safetensors"
    with path.open("rb") as stream:
        size=struct.unpack("<Q",stream.read(8))[0]
        if not 2<=size<=16<<20:raise ValueError("bounded original checkpoint header required")
        header=json.loads(stream.read(size));result={}
        for layer in (0,1):
            for leaf,shape in (("gate_up",[24576,4096]),("out",[4096,12288])):
                name=f"transformer_blocks.{layer}.img_mlp.{leaf}";info=header[name+".weight"]
                if info["dtype"]!="BF16" or info["shape"]!=shape:raise ValueError("original dense BF16 FFN payload required")
                begin,end=info["data_offsets"]
                if end-begin!=math.prod(shape)*2:raise ValueError("complete FFN tensor extent required")
                stream.seek(8+size+begin);remaining=end-begin;digest=hashlib.sha256()
                while remaining:
                    block=stream.read(min(4<<20,remaining))
                    if not block:raise ValueError("truncated original FFN tensor")
                    digest.update(block);remaining-=len(block)
                result[name]=digest.hexdigest()
    return result


def window(root,library,probe,manifest,identity,payloads,historical):
    summary=json.loads((root/"summary.json").read_text());report=json.loads((root/"trial/report.json").read_text())
    if (summary["status"]!="complete" or summary["assets"]["native_library"]!=sha256_file(library) or
            summary["assets"]["library"]!=sha256_file(probe) or report["loaded_runtime_id"]!=manifest["runtime_build_id"] or
            summary["model_snapshot"]!=identity or report["source_arrays_sha256"]!=payloads):
        raise ValueError("bound native/probe/model and actual original FFN bytes required")
    source=json.loads((root/"source.json").read_text())
    if set(source)!=set(summary["source_sha256"]):raise ValueError("complete FFN block tool snapshot required")
    for name,digest in summary["source_sha256"].items():
        raw=source[name].encode()
        if hashlib.sha256(raw).hexdigest()!=digest or not (raw==(ROOT/name).read_bytes() or raw in historical.get(name,set())):
            raise ValueError("actual FFN block source changed")
    if report["cpp_sha256"]!=summary["source_sha256"]["tools/profiling/qwen_base/ffn_projection_probe.mm"]:
        raise ValueError("actual compiled FFN probe differs")
    if screen.validate(report,summary["repeats"],summary["projection"],summary.get("activation","exp"),channels=summary.get("channels",6144))!=summary["analysis"]:
        raise ValueError("actual full-block timing/recipe analysis differs")
    if sha256_file(root/"trial/arrays.safetensors")!=summary["arrays_sha256"]:raise ValueError("full BF16 arrays changed")
    values=arrays(root/"trial/arrays.safetensors")
    expected={"ffn_input","block_reference","block_candidate","alternate_reference","alternate_candidate", "other_weight_reference","other_weight_candidate", "last_gpu_original","last_gpu_staged","last_private"}
    if set(values)!=expected or any(v.shape!=(1,1024,4096) or not np.isfinite(v).all() for v in values.values()):
        raise ValueError("complete finite BF16 block arrays required")
    checks={}
    for field,a,b in (("complete_block_quality","block_reference","block_candidate"),("alternate_quality","alternate_reference","alternate_candidate"),
                      ("other_weight_quality","other_weight_reference","other_weight_candidate")):
        q=quality(values[a],values[b]);checks[field]=q
        for key in report[field]:
            if not math.isclose(q[key],report[field][key],rel_tol=1e-8,abs_tol=1e-10):raise ValueError("full-array independently recomputed quality differs")
    if not np.array_equal(values["last_gpu_original"],values["last_gpu_staged"]):raise ValueError("final staged GPU control is not exact")
    checks["last_private"]=quality(values["last_gpu_original"],values["last_private"])
    array_hashes={name:hashlib.sha256(value.astype("<f4",copy=False).tobytes(order="C")).hexdigest() for name,value in values.items()}
    for key in ("report","evidence"):
        if sha256_file(root/summary["memory"][key])!=summary["memory"][key+"_sha256"]:raise ValueError("owned memory evidence changed")
    if not summary["memory"]["verified"]["complete"] or summary["memory"]["verified"]["command_exit_code"]!=0:
        raise ValueError("complete successful owned model window required")
    observed=None;events=None
    if summary["mode"]=="timeline":
        raw=json.loads((root/"trial/observer.json").read_text())
        if raw["source_sha256"]!=summary["source_sha256"]["tools/profiling/qwen_base/observer.mm"]:raise ValueError("actual observer source differs")
        observed=screen.observer_contract(raw,summary["repeats"]+7)
        if observed!=summary["observer"]:raise ValueError("actual model/evaluation/hot compile analysis differs")
        events=trace(raw)
    return dict(cohort=root.name,projection=summary["projection"],activation=summary.get("activation","exp"),channels=summary.get("channels",6144),case=summary["case"],mode=summary["mode"],analysis=summary["analysis"],
        quality=checks,array_sha256=array_hashes,observer=observed,memory=summary["memory"]["verified"],load_error=summary.get("load_error"),
        source_weight_payloads_verified=True,summary_sha256=sha256_file(root/"summary.json")),events


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("native-library","probe-library","native-source-archive","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--cohort",type=Path,action="append",required=True)
    parser.add_argument("--tool-source-archive",type=Path,action="append",default=[])
    parser.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    parser.add_argument("--old-fingerprint",type=Path);parser.add_argument("--new-fingerprint",type=Path)
    parser.add_argument("--factorial",action="store_true",help="first four cohorts are ordered descriptor/up-single/down-single/single stock; extras must replay their arrays")
    parser.add_argument("--same-capture",action="store_true",help="all policy windows must share exact source/GPU controls; same-recipe complete arrays replay exactly")
    args=parser.parse_args()
    if args.output.exists():parser.error("fresh CPU audit directory required")
    manifest=json.loads((args.native_library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    joint_audit.archive_contract(args.native_source_archive,manifest)
    historical={}
    for path in args.tool_source_archive:
        with tarfile.open(path,"r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():historical.setdefault(member.name,set()).add(archive.extractfile(member).read())
    fingerprint=None
    if args.old_fingerprint or args.new_fingerprint:
        if not args.old_fingerprint or not args.new_fingerprint:parser.error("both emitter fingerprints required")
        old=json.loads(args.old_fingerprint.read_text());new=json.loads(args.new_fingerprint.read_text())
        if (old["cases"]!=new["cases"] or not old["cpu_only"] or not new["cpu_only"] or len(old["cases"])!=6 or
                new["runtime_build_id"]!=manifest["runtime_build_id"]):raise ValueError("default emitter changed original six cases")
        fingerprint=dict(unchanged=True,cases=6,old_runtime=old["runtime_build_id"],new_runtime=new["runtime_build_id"],
                         old_sha256=sha256_file(args.old_fingerprint),new_sha256=sha256_file(args.new_fingerprint))
    identity=model_snapshot(args.model);payloads=original_hashes(args.model);records=[];traces=[]
    for root in args.cohort:
        record,events=window(root,args.native_library,args.probe_library,manifest,identity,payloads,historical)
        records.append(record)
        if events:traces.append((root.name,events))
    args.output.mkdir(parents=True)
    for name,events in traces:write_json(args.output/(name+"-trace.json"),events)
    result=dict(schema="tc-ffn-projection-block-audit-v1",status="complete",qualification_passed=False,
        native_library_sha256=sha256_file(args.native_library),probe_library_sha256=sha256_file(args.probe_library),
        native_source_archive_sha256=sha256_file(args.native_source_archive),script_sha256=sha256_file(Path(__file__)),
        original_ffn_payload_sha256=payloads,default_fingerprint=fingerprint,cohorts=records,
        factorial=factorial(records) if args.factorial else None,
        same_capture=same_capture(records) if args.same_capture else None,
        scope="full-array original-payload block audit; not full-image throughput/semantic qualification, physical INT8 or hardware overlap")
    write_json(args.output/"audit.json",result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
