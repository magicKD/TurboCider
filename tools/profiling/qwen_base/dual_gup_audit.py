#!/usr/bin/env python3
"""Independent actual BF16 block/F32-partial and matched dual-GUP audit."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile

import numpy as np
import dual_gup_collect as screen
import ffn_projection_audit as block
import joint_audit
from collect import ROOT,model_snapshot,sha256_file,write_json
from qkv_audit import arrays,quality
from down_audit import read_components,original_down,sampled_dot
from vpipe_audit import trace


def hashes(values):return {n:hashlib.sha256(v.astype("<f4",copy=False).tobytes()).hexdigest() for n,v in values.items()}


def check_quality(reference,value,reported):
    q=quality(reference,value)
    for name,v in reported.items():
        if type(v) not in (int,float) or not math.isclose(q[name],v,rel_tol=1e-8,abs_tol=1e-10):raise ValueError("full-array quality differs from actual native report")
    return q


def matched(records):
    if not records:raise ValueError("complete matched records required")
    anchors={};replay={}
    for r in records:
        case=r["case"];a=anchors.setdefault(case,r)
        for name in ("ffn_input","block_reference","alternate_reference","other_weight_reference","private_original_a","private_original_b","private_original_w"):
            if r["block_sha256"][name]!=a["block_sha256"][name]:raise ValueError("actual source/ordinary GPU/original hybrid controls differ")
        if any(r["gpu_sha256"][n]!=a["gpu_sha256"][n] for n in ("input_a","input_b","hidden_original_a","hidden_original_b","partial_original_a","partial_original_b")):
            raise ValueError("actual original GPU complement source changed")
        key=(case,r["kind"],r["channels"])
        if key in replay and (r["block_sha256"]!=replay[key][0] or r["gpu_sha256"]!=replay[key][1]):raise ValueError("same dual recipe repeated/timeline complete arrays changed")
        replay[key]=(r["block_sha256"],r["gpu_sha256"])
    return dict(same_case_source_and_strong_controls_exact=True,same_recipe_replays_exact=True,
        comparisons=[dict(cohort=r["cohort"],case=r["case"],kind=r["kind"],mode=r["mode"],
            original_seconds=r["analysis"]["median_seconds"]["private-original"],fused_seconds=r["analysis"]["median_seconds"]["private-fused"],
            incremental_reduction=r["analysis"]["incremental_reduction"] if r["mode"]=="stock" else None,
            gpu_complement=r["analysis"]["gpu_complement_alone_medians"]) for r in records],
        scope="per-window matched complete hybrid blocks; never pool tasks or substitute ordinary GPU for incremental denominator")


def window(root,probe,native,manifest,identity,payloads,weights,historical):
    summary=json.loads((root/"summary.json").read_text());report=json.loads((root/"trial/report.json").read_text())
    if (summary["status"]!="complete" or summary["assets"]["native_library"]!=sha256_file(native) or summary["assets"]["library"]!=sha256_file(probe) or
        summary["model_snapshot"]!=identity or report["source_arrays_sha256"]!=payloads or report["loaded_runtime_id"]!=manifest["runtime_build_id"] or
        report.get("compiled_native_id")!=manifest["runtime_build_id"]):raise ValueError("actual bound helper/native/model/original FFN bytes required")
    source=json.loads((root/"source.json").read_text())
    if set(source)!=set(summary["source_sha256"]):raise ValueError("complete dual-GUP source snapshot required")
    for name,digest in summary["source_sha256"].items():
        raw=source[name].encode()
        if hashlib.sha256(raw).hexdigest()!=digest or not (raw==(ROOT/name).read_bytes() or raw in historical.get(name,set())):raise ValueError("bound current/archive dual source differs")
    if report["cpp_sha256"]!=summary["source_sha256"]["tools/profiling/qwen_base/ffn_projection_probe.mm"] or report["dual_gup"]["header_sha256"]!=summary["source_sha256"]["tools/profiling/qwen_base/dual_gup.hpp"]:
        raise ValueError("compiled helper/kernel header identity differs")
    c=summary["channels"];n=12288-c;repeats=summary["repeats"]
    analysis=screen.validate(report,repeats,c,summary["kind"]=="shared")
    if analysis!=summary["analysis"]:raise ValueError("independently recomputed paired complete timings differ")
    for name,digest in summary["arrays_sha256"].items():
        if sha256_file(root/"trial"/name)!=digest:raise ValueError("actual full block/component payload changed")
    if set(summary["arrays_sha256"])!={"arrays.safetensors","gpu-complement-arrays.safetensors"}:raise ValueError("both full block and typed GPU partial payloads required")
    values=arrays(root/"trial/arrays.safetensors")
    expected={"ffn_input","block_reference","block_candidate","alternate_reference","alternate_candidate","other_weight_reference","other_weight_candidate",
              "last_gpu_original","last_gpu_staged","last_private","private_original_a","private_original_b","private_original_w","last_private_original"}
    if set(values)!=expected or any(v.shape!=(1,1024,4096) or not np.isfinite(v).all() for v in values.values()):raise ValueError("fourteen complete finite original BF16 block arrays required")
    checks={}
    for name,left,right in (("complete_block_quality","block_reference","block_candidate"),("alternate_quality","alternate_reference","alternate_candidate"),
        ("other_weight_quality","other_weight_reference","other_weight_candidate"),("dual_incremental_quality_a","private_original_a","block_candidate"),
        ("dual_incremental_quality_b","private_original_b","alternate_candidate"),("dual_incremental_quality_w","private_original_w","other_weight_candidate")):
        checks[name]=check_quality(values[left],values[right],report[name])
    bh=hashes(values)
    increment_bits={name:bh[left]==bh[right] for name,left,right in (("a","private_original_a","block_candidate"),
        ("b","private_original_b","alternate_candidate"),("w","private_original_w","other_weight_candidate"),("last","last_private_original","last_private"))}
    for left,right in (("last_gpu_original","last_gpu_staged"),("last_gpu_original","alternate_reference"),("last_private","alternate_candidate"),("last_private_original","private_original_b")):
        if bh[left]!=bh[right]:raise ValueError("last actual input variant/private GPU replay differs")
    names={"input_a","input_b"}|{f"{part}_{kind}_{v}" for part in ("hidden","partial") for kind in ("original","fused") for v in ("a","b")}
    specs={name:("F32",(1,1024,4096)) if name.startswith("partial") else ("BF16",(1,1024,n if name.startswith("hidden") else 4096)) for name in names}
    gpu=read_components(root/"trial/gpu-complement-arrays.safetensors",specs)
    if any(not np.isfinite(v).all() for v in gpu.values()):raise ValueError("all complete BF16-hidden/F32-partial arrays must be finite")
    gh=hashes(gpu)
    if gh["input_a"]!=bh["ffn_input"]:raise ValueError("GPU complement was not fed actual source FFN input")
    coordinates=[(0,0),(1023,4095),(0,4095),(1023,0)]+[((i*137+11)%1024,(i*977+31)%4096) for i in range(28)]
    oracles={};gpu_checks={}
    for v in ("a","b"):
        gpu_checks[v]=check_quality(gpu["hidden_original_"+v],gpu["hidden_fused_"+v],report["dual_hidden_quality_"+v])
        if (gh["hidden_original_"+v]==gh["hidden_fused_"+v]) is not report["dual_hidden_bits_exact_"+v]:raise ValueError("actual full hidden bits gate differs")
        for kind in ("original","fused"):
            oracles[kind+"_"+v]=sampled_dot(weights,gpu["hidden_"+kind+"_"+v],gpu["partial_"+kind+"_"+v],coordinates)
        if gh["hidden_original_"+v]==gh["hidden_fused_"+v] and gh["partial_original_"+v]!=gh["partial_fused_"+v]:raise ValueError("unchanged hidden changed original F32 down arithmetic")
    for name in ("report","evidence"):
        if sha256_file(root/summary["memory"][name])!=summary["memory"][name+"_sha256"]:raise ValueError("owned memory evidence changed")
    if not summary["memory"]["verified"]["complete"] or summary["memory"]["verified"]["command_exit_code"]!=0:raise ValueError("complete successful owned model window required")
    observed=None;events=None
    if summary["mode"]=="timeline":
        raw=json.loads((root/"trial/observer.json").read_text())
        if raw["source_sha256"]!=summary["source_sha256"]["tools/profiling/qwen_base/observer.mm"]:raise ValueError("observer compiled source differs")
        observed=screen.observer_contract(raw,report,repeats)
        if observed!=summary["observer"]:raise ValueError("actual original/fused fixed-model selector contract differs")
        events=trace(raw)
    return dict(cohort=root.name,case=summary["case"],kind=summary["kind"],mode=summary["mode"],channels=c,repeats=repeats,analysis=analysis,
        block_quality=checks,incremental_bits_exact=increment_bits,gpu_quality=gpu_checks,block_sha256=bh,gpu_sha256=gh,sampled_f64_down_oracles=oracles,observer=observed,
        memory=summary["memory"]["verified"],load_error=summary.get("load_error"),summary_sha256=sha256_file(root/"summary.json")),events


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("native-library","native-source-archive","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--cohort",nargs=2,metavar=("ROOT","PROBE_LIBRARY"),action="append",required=True)
    p.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--tool-source-archive",type=Path,action="append",default=[]);args=p.parse_args()
    if args.output.exists():p.error("fresh independent full dual-GUP audit required")
    manifest=json.loads((args.native_library.parent/"runtime-build/runtime-build-manifest.json").read_text());joint_audit.archive_contract(args.native_source_archive,manifest)
    historical={}
    for path in args.tool_source_archive:
        with tarfile.open(path,"r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():historical.setdefault(member.name,set()).add(archive.extractfile(member).read())
    identity=model_snapshot(args.model);payloads=block.original_hashes(args.model);weights=original_down(args.model);records=[];traces=[]
    for root,probe in args.cohort:
        r,events=window(Path(root),Path(probe),args.native_library,manifest,identity,payloads,weights,historical);records.append(r)
        if events:traces.append((Path(root).name,events))
    result=dict(schema="tc-dual-gup-real-block-audit-v1",status="complete",qualification_passed=False,runtime_build_id=manifest["runtime_build_id"],
        native_library_sha256=sha256_file(args.native_library),native_source_archive_sha256=sha256_file(args.native_source_archive),script_sha256=sha256_file(Path(__file__)),
        cohorts=records,matched=matched(records),scope="original BF16 base40 first-decode real full-block paired controls; partial capture, not full40 image qualification")
    args.output.mkdir(parents=True)
    for name,events in traces:write_json(args.output/(name+"-trace.json"),events)
    write_json(args.output/"audit.json",result);print(json.dumps(result["matched"],indent=2))


if __name__=="__main__":main()
