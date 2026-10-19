#!/usr/bin/env python3
"""CPU-only source/exit audit of rejected activation screens, never a speed sample."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import joint_audit
from collect import ROOT, FFN_ACTIVATIONS, model_snapshot, sha256_file, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("native-library","probe-library","native-source-archive","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--failed",type=Path,action="append",required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error("fresh failure-audit directory required")
    manifest=json.loads((args.native_library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    joint_audit.archive_contract(args.native_source_archive,manifest)
    args.output.mkdir(parents=True);records=[]
    for root in args.failed:
        summary=json.loads((root/"summary.json").read_text());report=json.loads((root/"trial/report.json").read_text())
        if (summary["status"]!="failed" or report["status"]!="failed" or summary.get("analysis") or
                summary["activation"] not in FFN_ACTIVATIONS[1:] or report.get("activation")!=summary["activation"] or
                summary["projection"]!="descriptor" or report["projection"]!="descriptor" or
                report["loaded_runtime_id"]!=manifest["runtime_build_id"] or
                summary["assets"]["native_library"]!=sha256_file(args.native_library) or
                summary["assets"]["library"]!=sha256_file(args.probe_library) or
                summary["model_snapshot"]!=model_snapshot(ROOT/"models/Comfy-Org-Qwen-Image-2.1") or
                report.get("complete_request") is not False or report.get("image_export") is not False or
                report.get("completed_full_dit_steps")!=1 or report.get("samples") or
                (root/"trial/arrays.safetensors").exists() or (root/"trial/not-exported.png").exists() or
                "numerical/weight-switch self-test failed" not in report.get("error","")):
            raise ValueError("actual rejected activation self-test, no complete candidate performance/image required")
        actual=report["actual_source_request"]
        if (actual["model"],actual["operation"],actual["width"],actual["height"],actual["steps"],actual["seed"],
                actual["audio"],actual["lora_count"],actual["reference_count"])!=("qwen-image-2.1","image.generate",512,512,40,73,False,0,0):
            raise ValueError("actual original-base capture request differs")
        source=json.loads((root/"source.json").read_text())
        if set(source)!=set(summary["source_sha256"]):raise ValueError("complete rejected-screen source snapshot required")
        for name,digest in summary["source_sha256"].items():
            raw=source[name].encode()
            if hashlib.sha256(raw).hexdigest()!=digest or raw!=(ROOT/name).read_bytes():
                raise ValueError("rejected screen source changed")
        if report["cpp_sha256"]!=summary["source_sha256"]["tools/profiling/qwen_base/ffn_projection_probe.mm"]:
            raise ValueError("actual failed probe source differs")
        derived=args.output/(root.name+"-memory.json")
        verified=subprocess.run([sys.executable,"-B",str(ROOT/"tools/native/verify_process_tree_samples.py"),
            str(root/"trial-memory.jsonl"),"--output",str(derived)],capture_output=True,text=True,check=False)
        previous=json.loads((root/"trial-memory-report.json").read_text());memory=json.loads(derived.read_text())
        if (verified.returncode!=3 or memory!=previous or memory["complete"] is not False or
                memory["reasons"]!=["command_failed"] or memory["command_exit_code"]!=1):
            raise ValueError("expected owned terminal nonzero exit, without unrelated sampling defects")
        records.append(dict(cohort=root.name,activation=summary["activation"],actual_probe_error=report["error"],
            successful_candidate_samples=0,completed_source_steps=1,owned_nonzero_exit_verified=True,
            sampling_gap_within_limit=True,memory_qualification_passed=False,memory=memory,
            source_sha256=summary["source_sha256"],summary_sha256=sha256_file(root/"summary.json"),
            report_sha256=sha256_file(root/"trial/report.json"),memory_evidence_sha256=sha256_file(root/"trial-memory.jsonl")))
    result=dict(schema="tc-activation-rejection-audit-v1",status="complete",qualification_passed=False,
        runtime_build_id=manifest["runtime_build_id"],native_library_sha256=sha256_file(args.native_library),
        probe_library_sha256=sha256_file(args.probe_library),native_source_archive_sha256=sha256_file(args.native_source_archive),
        script_sha256=sha256_file(Path(__file__)),rejected=records,
        scope="expected failed self-test/exit evidence only; no activation accuracy arrays, successful runtime speed or complete image qualification")
    write_json(args.output/"audit.json",result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
