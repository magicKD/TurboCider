#!/usr/bin/env python3
"""CPU-only original-payload/full-array/counter audit of FFN handoff replays."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile

import numpy as np

import ffn_projection_audit as arrays_audit
import ffn_projection_collect as block
import handoff_collect as screen
import joint_audit
from collect import ROOT,model_snapshot,sha256_file,write_json
from qkv_audit import arrays,quality


def matched(records):
    if not records:raise ValueError("actual complete handoff records required")
    anchor=records[0];seen={}
    for row in records:
        if any(row["array_sha256"][key]!=anchor["array_sha256"][key] for key in
                ("ffn_input","block_reference","alternate_reference","other_weight_reference")):
            raise ValueError("same real input and ordinary full GPU controls required")
        key=(row["channels"],row["gpu_complement"],row.get("down_policy","none"),row.get("gate_up","bf16"))
        if key in seen and row["array_sha256"]!=seen[key]["array_sha256"]:
            raise ValueError("same recipe/version/timeline did not restore all ten arrays")
        seen.setdefault(key,row)
    return dict(identical_sources_and_gpu_controls=True,same_recipe_complete_arrays_exact=True,
        recipes=[dict(channels=k[0],gpu_complement=k[1],down_policy=k[2],gate_up=k[3]) for k in seen])


def window(root,native,probe_dirs,manifest,identity,payloads,historical):
    summary=json.loads((root/"summary.json").read_text());report=json.loads((root/"trial/report.json").read_text())
    channels=summary["channels"];strategy=summary.get("gpu_complement","separate")
    down_policy=summary.get("down_policy","none");gate_up=summary.get("gate_up","bf16")
    if (summary["status"]!="complete" or summary["schema"]!="tc-ffn-handoff-block-screen-v1" or
            summary["assets"]["native_library"]!=sha256_file(native) or
            summary["runtime_build_id"]!=manifest["runtime_build_id"] or report["loaded_runtime_id"]!=manifest["runtime_build_id"] or
            summary["model_snapshot"]!=identity or report["source_arrays_sha256"]!=payloads):
        raise ValueError("bound complete native/model and original FFN payloads required")
    suffix="packed_" if strategy=="packed" else ""
    probe_name=(f"libqwen_ffn_down_{'fp16_' if gate_up=='fp16' else ''}{down_policy}_c{channels}.dylib" if down_policy!="none" else
                f"libqwen_ffn_handoff_{suffix}c{channels}.dylib")
    probes=[p/probe_name for p in probe_dirs]
    if not any(p.is_file() and sha256_file(p)==summary["assets"]["library"] for p in probes):
        raise ValueError("actual fixed-share probe binary unavailable or changed")
    if not any((p/"libqwen_handoff_observer.dylib").is_file() and sha256_file(p/"libqwen_handoff_observer.dylib")==summary["assets"]["observer"] for p in probe_dirs):
        raise ValueError("actual observer binary unavailable or changed")
    source=json.loads((root/"source.json").read_text())
    if set(source)!=set(summary["source_sha256"]):raise ValueError("complete bound tool source snapshot required")
    for name,digest in summary["source_sha256"].items():
        raw=source[name].encode()
        if hashlib.sha256(raw).hexdigest()!=digest or not (raw==(ROOT/name).read_bytes() or raw in historical.get(name,set())):
            raise ValueError("bound current/archive tool source differs")
    if report["cpp_sha256"]!=summary["source_sha256"]["tools/profiling/qwen_base/ffn_projection_probe.mm"]:
        raise ValueError("actual compiled FFN source differs")
    legacy="gpu_complement" not in summary
    if legacy:
        # V1 predates both the packing strategy and delayed-event control.
        # Its immutable archived probe has neither packing fields nor macro.
        # Do not fabricate missing setup cost or give it the new self-test.
        if (strategy!="separate" or "gpu_complement_strategy" in report or
                "TC_FFN_HANDOFF_PACKED_GPU" in source["tools/profiling/qwen_base/ffn_projection_probe.mm"]):
            raise ValueError("explicit archived v1 no-packing probe required")
        derived=block.validate(report,summary["repeats"],"descriptor","sigmoid-complement",channels=channels)
        rows=report["gpu_complement_alone_samples"]
        if (report.get("handoff_screen") is not True or report.get("ane_channels")!=channels or
                report.get("gpu_channels")!=12288-channels or len(rows)!=summary["repeats"]+2 or
                [r["rep"] for r in rows]!=list(range(-2,summary["repeats"]))):
            raise ValueError("actual archived fixed-share/complement samples required")
        for row in rows:
            if (row["warmup"] is not (row["rep"]<0) or row["input_variant"] is not bool(row["rep"]&1) or
                    type(row["seconds"]) not in (int,float) or not math.isfinite(row["seconds"]) or row["seconds"]<=0):
                raise ValueError("invalid archived complement timing")
        import statistics
        derived["gpu_complement_alone_median_seconds"]=statistics.median(r["seconds"] for r in rows if not r["warmup"])
    else:derived=screen.validate(report,summary["repeats"],channels,strategy,down_policy=down_policy,gate_up=gate_up)
    if derived!=summary["analysis"]:raise ValueError("independent actual block timing analysis differs")
    if sha256_file(root/"trial/arrays.safetensors")!=summary["arrays_sha256"]:raise ValueError("complete BF16 arrays changed")
    values=arrays(root/"trial/arrays.safetensors")
    expected={"ffn_input","block_reference","block_candidate","alternate_reference","alternate_candidate",
              "other_weight_reference","other_weight_candidate","last_gpu_original","last_gpu_staged","last_private"}
    if set(values)!=expected or any(v.shape!=(1,1024,4096) or not np.isfinite(v).all() for v in values.values()):
        raise ValueError("all ten complete finite real BF16 arrays required")
    checks={}
    for field,a,b in (("complete_block_quality","block_reference","block_candidate"),("alternate_quality","alternate_reference","alternate_candidate"),
                      ("other_weight_quality","other_weight_reference","other_weight_candidate")):
        q=quality(values[a],values[b]);checks[field]=q
        if any(not math.isclose(q[key],value,rel_tol=1e-8,abs_tol=1e-10) for key,value in report[field].items()):
            raise ValueError("independent full-array numerical comparison differs")
    if not np.array_equal(values["last_gpu_original"],values["last_gpu_staged"]):raise ValueError("staged GPU control is not exact")
    checks["last_private"]=quality(values["last_gpu_original"],values["last_private"])
    hashes={k:hashlib.sha256(v.astype("<f4",copy=False).tobytes(order="C")).hexdigest() for k,v in values.items()}
    for key in ("report","evidence"):
        if sha256_file(root/summary["memory"][key])!=summary["memory"][key+"_sha256"]:raise ValueError("original memory evidence changed")
    if not summary["memory"]["verified"]["complete"] or summary["memory"]["verified"]["command_exit_code"]!=0:
        raise ValueError("complete successful model process required")
    if "load" in summary and sha256_file(root/summary["load"]["evidence"])!=summary["load"]["evidence_sha256"]:
        raise ValueError("original load observation changed")
    observed=handoff=None
    if summary["mode"]=="timeline":
        raw=json.loads((root/"trial/observer.json").read_text())
        if (raw["source_sha256"]!=summary["source_sha256"]["tools/profiling/qwen_base/observer.mm"] or
                raw["handoff"]["source_sha256"]!=summary["source_sha256"]["tools/profiling/qwen_base/handoff_observer.mm"]):
            raise ValueError("actual base/encoder observer source differs")
        observed=block.observer_contract(raw,summary["repeats"]+7)
        handoff=screen.handoff_analysis(raw,report,summary["repeats"],delayed_self_test=not legacy)
        comparable=dict(handoff)
        if legacy:comparable.pop("delayed_event_self_test")
        if observed!=summary["observer"] or comparable!=summary["handoff"]:
            raise ValueError("actual selector/encoder timestamp analysis differs")
    return dict(cohort=root.name,channels=channels,gpu_complement=strategy,down_policy=down_policy,gate_up=gate_up,legacy_no_delayed_control=legacy,
        mode=summary["mode"],analysis=derived,quality=checks,array_sha256=hashes,observer=observed,handoff=handoff,
        gpu_packed_weight_bytes=report.get("gpu_packed_weight_bytes"),gpu_packed_setup_seconds=report.get("gpu_packed_setup_seconds"),
        source_weight_payloads_verified=True,memory=summary["memory"]["verified"],load_error=summary.get("load_error"),
        summary_sha256=sha256_file(root/"summary.json"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("native-library","native-source-archive","output"):p.add_argument("--"+name,type=Path,required=True)
    for name in ("probe-dir","cohort"):p.add_argument("--"+name,type=Path,action="append",required=True)
    p.add_argument("--tool-source-archive",type=Path,action="append",default=[])
    p.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1");args=p.parse_args()
    if args.output.exists():p.error("fresh audit directory required")
    manifest=json.loads((args.native_library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    joint_audit.archive_contract(args.native_source_archive,manifest)
    historical={}
    for path in args.tool_source_archive:
        with tarfile.open(path,"r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():historical.setdefault(member.name,set()).add(archive.extractfile(member).read())
    identity=model_snapshot(args.model);payloads=arrays_audit.original_hashes(args.model)
    records=[window(root,args.native_library,args.probe_dir,manifest,identity,payloads,historical) for root in args.cohort]
    controls=matched(records)
    args.output.mkdir(parents=True)
    result=dict(schema="tc-qwen-ffn-handoff-audit-v1",status="complete",qualification_passed=False,cohorts=records,matched=controls,
        native_library_sha256=sha256_file(args.native_library),native_source_archive_sha256=sha256_file(args.native_source_archive),
        original_ffn_payload_sha256=payloads,script_sha256=sha256_file(Path(__file__)),
        tool_source_archive_sha256={str(p):sha256_file(p) for p in args.tool_source_archive},
        scope="Real first-decode block replays and encoder timestamps; no full forty-step image speed, physical ANE INT8/overlap, shader busy-time, or general image semantics qualification.")
    write_json(args.output/"audit.json",result)
    print(json.dumps(dict(status=result["status"],cohorts=len(records),matched=controls),indent=2))


if __name__=="__main__":main()
