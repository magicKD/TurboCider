#!/usr/bin/env python3
"""Same-boundary GPU/QKV/static-Private/composition whole-block screen."""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

from flux9_bf16_framework_screen import write_json
from qwen_student_prefix_screen import adapter_pin
from runtime_ane_common import benchmark_environment,sha256_file
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ROOT=Path(__file__).resolve().parents[2]
NAMES=("gpu-original","gpu-staged","gpu-qkv","private","private-qkv")
FENCE_NAMES=NAMES+("private-fenced","private-qkv-fenced")
EVENT_NAMES=NAMES+("private-event","private-qkv-event")
ORDER_NAMES=EVENT_NAMES+("private-gpu-first","private-qkv-gpu-first")
PACK_NAMES=EVENT_NAMES+("private-packed","private-qkv-packed")


def recipe(native):
    version=native.get("schedule_recipe")
    if version is None:return NAMES,15,39,"private-qkv"
    if version=="packed-suffix-v4":
        early=native.get("ready_before_private_completion")
        setup=native.get("packed_suffix_setup_seconds")
        if (native.get("input_ready_fences")!=45 or native.get("event_value_fences")!=45 or
                native.get("gpu_first_submissions")!=0 or native.get("packed_suffix_calls")!=45 or native.get("packed_suffix_bytes")!=167772160 or
                native.get("packed_suffix_control_exact") is not True or native.get("packed_suffix_fresh_input_restoration") is not True or
                native.get("preflight_ready_wait")!="event-value-and-packed-suffix" or native.get("measured_samples_per_arm")!=18 or
                type(setup) not in (int,float) or not math.isfinite(setup) or setup<=0 or type(early) is not int or not 1<=early<=45):
            raise ValueError("actual byte-verified packed base, exact GPU controls and full ownership preflight required")
        return PACK_NAMES,18,132,"private-qkv-packed"
    if version=="submission-order-v3":
        early=native.get("ready_before_private_completion")
        if (native.get("input_ready_fences")!=45 or native.get("event_value_fences")!=45 or
                native.get("gpu_first_submissions")!=45 or native.get("gpu_first_fresh_input_restoration") is not True or
                native.get("preflight_ready_wait")!="event-value-and-gpu-first" or native.get("measured_samples_per_arm")!=18 or
                type(early) is not int or not 1<=early<=45):
            raise ValueError("actual GPU-first/event/control recipe and both ownership preflights required")
        return ORDER_NAMES,18,132,"private-qkv-gpu-first"
    if version=="producer-event-v2":
        early=native.get("ready_before_private_completion")
        if (native.get("input_ready_fences")!=37 or native.get("event_value_fences")!=37 or
                native.get("preflight_ready_wait")!="event-value" or native.get("measured_samples_per_arm")!=14 or
                type(early) is not int or not 1<=early<=37):
            raise ValueError("actual direct event-value fences including A/B/A preflight required")
        return EVENT_NAMES,14,71,"private-qkv-event"
    if version!="producer-fence-v1" or native.get("input_ready_fences")!=34 or native.get("measured_samples_per_arm")!=14:
        raise ValueError("actual bounded producer-fence recipe required")
    return FENCE_NAMES,14,71,"private-qkv-fenced"


def quality(value,limit,exact=False):
    if (any(type(value[k]) not in (int,float) or not math.isfinite(value[k]) for k in ("relative_l2","cosine","max_abs")) or
            not 0<=value["relative_l2"]<=limit or not .999<=value["cosine"]<=1.000001 or value["max_abs"]<0 or
            type(value["elements"]) is not int or value["elements"]!=1024*4096 or
            type(value["changed"]) is not int or not 0<=value["changed"]<=value["elements"] or (exact and value["changed"]!=0)):
        raise ValueError("complete consumer numerical gate differs")


def analyze(native):
    names,count,calls,primary=recipe(native)
    order_recipe=native.get("schedule_recipe")=="submission-order-v3"
    packed_recipe=native.get("schedule_recipe")=="packed-suffix-v4"
    preflight=6 if order_recipe or packed_recipe else 3
    if (native.get("status")!="measured" or native.get("runtime_lora_applied_terms")!=227 or native.get("image_export") is not False or
            native.get("borrowed_values_unchanged") is not True or native.get("fresh_gpu_arguments_verified") is not True or
            native.get("private_fresh_input_restoration") is not True or native.get("preflight_private_calls")!=preflight or
            native.get("successful_private_calls")!=calls or native.get("static_prefix_channels")!=2048 or native.get("shared_ranks") is not False or
            native.get("completed_full_dit_steps")!=1 or native.get("stopped_step")!=1 or native.get("stopped_layer")!=0 or
            native.get("qkv_packed_bytes")!=100663296):
        raise ValueError("complete bounded ordinary-block composition required")
    rows=native["samples"]
    if len(rows)!=(count+3)*len(names):raise ValueError("all first/warm/measured arms required")
    for iteration in range(-3,count):
        group=[row for row in rows if row["iteration"]==iteration]
        if [row["arm"] for row in group]!=[names[(visit+iteration+len(names))%len(names)] for visit in range(len(names))]:raise ValueError("balanced arm order differs")
        for row in group:
            wall=row["seconds"];name=row["arm"]
            if type(wall) not in (int,float) or not math.isfinite(wall) or wall<=0:raise ValueError("finite complete block wall required")
            quality(row["quality"],.001,name in NAMES[:2])
            if name=="gpu-staged" or name.startswith("private"):
                for key in ("matched_input_ffn_quality","original_chain_ffn_quality"):quality(row[key],.03,name=="gpu-staged")
            elif row["matched_input_ffn_quality"] is not None or row["original_chain_ffn_quality"] is not None:
                raise ValueError("a fused block cannot claim separately observed FFN output")
            if native.get("schedule_recipe"):
                phases=row["private_phases"]
                if name.startswith("private"):
                    fenced=name.endswith("fenced") or name.endswith("event")
                    if phases["producer_fence"] is not fenced:raise ValueError("actual producer fence differs")
                    for key in ("corrections_ready_seconds","submission_seconds","input_ready_wait_seconds","exposed_wait_seconds","post_join_seconds"):
                        if type(phases[key]) not in (int,float) or not math.isfinite(phases[key]) or phases[key]<0:
                            raise ValueError("finite actual per-call phase span required")
                    if (phases["input_ready_wait_seconds"]>0) is not fenced:raise ValueError("actual ready-event wait differs")
                    if native.get("schedule_recipe") in ("producer-event-v2","submission-order-v3","packed-suffix-v4"):
                        if phases["ready_wait_kind"]!=("event-value" if fenced else "none") or\
                                (type(phases["private_complete_at_ready"]) is not bool if fenced else phases["private_complete_at_ready"] is not None):
                            raise ValueError("actual event-only readiness observation required")
                    if order_recipe or packed_recipe:
                        if phases["submission_order"]!=("gpu-first" if name.endswith("gpu-first") else "transfer-first"):
                            raise ValueError("actual complement/transfer order differs")
                        for key in ("gpu_submit_seconds","transfer_submit_seconds"):
                            if type(phases[key]) not in (int,float) or not math.isfinite(phases[key]) or phases[key]<=0:
                                raise ValueError("actual GPU and transfer submission spans required")
                    if packed_recipe and phases["gpu_suffix_recipe"]!=("packed-base" if name.endswith("packed") else "original"):
                        raise ValueError("actual packed GPU complement differs")
                elif phases is not None:raise ValueError("GPU-only arm claimed Private phase work")
    values={name:[row["seconds"] for row in rows if row["arm"]==name and row["iteration"]>=0] for name in names}
    medians={name:statistics.median(rows) for name,rows in values.items()}
    reductions={name:100*(1-value/medians["gpu-original"]) for name,value in medians.items()}
    result=dict(whole_block_seconds=values,medians=medians,reduction_vs_original_gpu_percent=reductions,
        qkv_incremental_reduction_vs_private_percent=100*(1-medians["private-qkv"]/medians["private"]),
        primary_recipe=primary,primary_local_gate_passed=reductions[primary]>=3.,
        scope="complete actual layer0 block only; ordinary compiled GPU denominator; no full-request/image-quality or physical overlap claim")
    if native.get("schedule_recipe"):
        phase_medians={}
        for name in names:
            if not name.startswith("private"):continue
            selected=[row for row in rows if row["arm"]==name and row["iteration"]>=0]
            phase_medians[name]={key:statistics.median(row["private_phases"][key] for row in selected) for key in
                ("corrections_ready_seconds","submission_seconds","input_ready_wait_seconds","exposed_wait_seconds","post_join_seconds")}
            if any(type(row["producer_seconds"]) not in (int,float) or not math.isfinite(row["producer_seconds"]) or row["producer_seconds"]<=0 for row in selected):
                raise ValueError("actual complete producer span required")
            phase_medians[name]["producer_seconds"]=statistics.median(row["producer_seconds"] for row in selected)
        fence_names=FENCE_NAMES[-2:] if native.get("schedule_recipe")=="producer-fence-v1" else EVENT_NAMES[-2:]
        result.update(private_phase_medians=phase_medians,
            fence_incremental_reduction_vs_combo_percent=100*(1-medians[fence_names[-1]]/medians["private-qkv"]),
            qkv_incremental_reduction_with_fence_percent=100*(1-medians[fence_names[-1]]/medians[fence_names[-2]]),
            phase_scope="host spans; submission includes input-ready wait; exposed wait overlaps GPU; never sum as physical work")
        if native.get("schedule_recipe") in ("producer-event-v2","submission-order-v3","packed-suffix-v4"):
            early=sum(row["private_phases"]["private_complete_at_ready"] is False for row in rows if row["arm"] in EVENT_NAMES[-2:])
            if not early<=native["ready_before_private_completion"]<=early+3:
                raise ValueError("early-ready observations disagree with actual preflight/measured calls")
            result["event_ready_before_private_completion_warm_measured"]=early
        if order_recipe or packed_recipe:
            for name in names:
                if not name.startswith("private"):continue
                selected=[row for row in rows if row["arm"]==name and row["iteration"]>=0]
                phase_medians[name].update({key:statistics.median(row["private_phases"][key] for row in selected) for key in
                    ("gpu_submit_seconds","transfer_submit_seconds")})
            if order_recipe:
                result.update(gpu_first_incremental_reduction_vs_combo_percent=100*(1-medians[names[-1]]/medians["private-qkv"]),
                    gpu_first_incremental_reduction_vs_event_combo_percent=100*(1-medians[names[-1]]/medians["private-qkv-event"]),
                    qkv_incremental_reduction_with_gpu_first_percent=100*(1-medians[names[-1]]/medians[names[-2]]))
            else:
                result.update(packed_incremental_reduction_vs_combo_percent=100*(1-medians[names[-1]]/medians["private-qkv"]),
                    packed_incremental_reduction_vs_event_combo_percent=100*(1-medians[names[-1]]/medians["private-qkv-event"]),
                    qkv_incremental_reduction_with_packed_suffix_percent=100*(1-medians[names[-1]]/medians[names[-2]]))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary",type=Path,required=True);p.add_argument("--library",type=Path,required=True)
    p.add_argument("--model",type=Path,default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--lora",type=Path,default=ROOT/"models/Viggle-Qwen-Image-2.1-viggle-turbo/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors")
    p.add_argument("--reference",type=Path);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    for key in ("binary","library","model","lora","reference"):
        if getattr(args,key) is not None:setattr(args,key,getattr(args,key).resolve(strict=True))
    pin=adapter_pin(args.lora)
    if pin["rank"]!="r128" or sha256_file(args.lora)!=pin["sha256"]:p.error("original pinned r128 required")
    manifest_path=args.library.parent/"runtime-build/runtime-build-manifest.json";manifest=json.loads(manifest_path.read_text())
    for item in manifest["inputs"]["sources"]:
        if sha256_file(ROOT/item["path"])!=item["sha256"]:p.error("current native headers must match the linked library")
    paths=[Path(__file__),ROOT/"tools/native/qwen21_qkv_private_combo_probe.mm",ROOT/"tools/native/qwen21_qkv_lora_prepare.hpp",
           ROOT/"tools/native/qwen21_static_lora_prefix.hpp",ROOT/"tools/native/qwen21_static_lora_mil.hpp",
           ROOT/"tools/native/build_qwen21_qkv_private_combo_probe.sh"]
    hashes={str(path.relative_to(ROOT)):sha256_file(path) for path in paths}
    d=json.loads(subprocess.check_output([str(args.binary),"--describe"],text=True,timeout=30))
    if (d["arms"]!=list(PACK_NAMES) or d["primary_recipe"]!="private-qkv-packed" or d["schedule_recipe"]!="packed-suffix-v4" or
            d.get("ready_wait_kind")!="mixed" or d.get("preflight_ready_wait")!="event-value-and-packed-suffix" or d["static_prefix_channels"]!=2048 or d["shared_ranks"] is not False or
            d["measured_samples_per_arm"]!=18 or d["warmups_per_arm"]!=3 or d["loaded_runtime_id"]!=manifest["runtime_build_id"] or
            any(d[key]!=hashes["tools/native/"+name] for key,name in (("cpp_sha256","qwen21_qkv_private_combo_probe.mm"),
                ("qkv_sha256","qwen21_qkv_lora_prepare.hpp"),("prefix_sha256","qwen21_static_lora_prefix.hpp"),("mil_sha256","qwen21_static_lora_mil.hpp")))):
        p.error("rebuild a current bound composition probe in a fresh output")
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    summary=dict(schema="tc-qwen-qkv-private-combo-screen-v1",status="running",qualification_passed=False,source_sha256=hashes,
        binary_sha256=sha256_file(args.binary),library_sha256=sha256_file(args.library),runtime_build_id=manifest["runtime_build_id"],
        manifest_sha256=sha256_file(manifest_path),adapter_sha256=pin["sha256"],probe_descriptor=d,case="edit" if args.reference else "generate",
        gates=dict(primary_recipe="private-qkv-packed",minimum_complete_block_reduction_percent=3.,maximum_block_relative_l2=.001,
                   maximum_ffn_relative_l2=.03,minimum_cosine=.999,strong_gpu_control_exact=True),
        scope="one actual r128/joint step1/layer0 composition; full consumers and actual Private jobs; not full image/physical overlap")
    if args.reference:summary["reference_sha256"]=sha256_file(args.reference)
    write_json(args.output/"source.json",{str(path.relative_to(ROOT)):path.read_text() for path in paths});write_json(args.output/"summary.json",summary)
    observer=LoadObservation(args.output/"load.jsonl")
    command=[str(args.binary),str(args.model),str(args.lora),str(args.output/"trial"),str(args.reference) if args.reference else "-"]
    try:
        with (args.output/"stdout.log").open("x") as stdout,(args.output/"stderr.log").open("x") as stderr:
            summary["memory"]=run_sampled(command,repo=ROOT,output=args.output,stem="trial",env=benchmark_environment(),
                stdout=stdout,stderr=stderr,timeout=1200,interval_ms=100,max_gap_ms=500,observer=observer)
        native=json.loads((args.output/"trial/native.json").read_text())
        if native["loaded_runtime_id"]!=manifest["runtime_build_id"]:raise ValueError("actual loaded runtime differs")
        summary["analysis"]=analyze(native);summary["status"]="complete"
    except Exception as error:
        summary.update(status="failed",error=str(error));path=args.output/"trial/native.json"
        if path.is_file():summary["native_error"]=json.loads(path.read_text()).get("error")
    try:summary["load"]=observer.verify()
    except ValueError as error:summary["load_error"]=str(error)
    if any(sha256_file(path)!=hashes[str(path.relative_to(ROOT))] for path in paths) or sha256_file(args.binary)!=summary["binary_sha256"] or\
            sha256_file(args.library)!=summary["library_sha256"]:summary.update(status="failed",error="source/artifact changed during measurement")
    write_json(args.output/"summary.json",summary);print(json.dumps(summary,indent=2))
    if summary["status"]!="complete":raise SystemExit(1)


if __name__=="__main__":main()
