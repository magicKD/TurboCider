#!/usr/bin/env python3
"""Bind the safety-only revision to current sources and prior measured data.

New captures prove parity, not new performance. Historical source verification
uses its actual snapshot for changed files, never relabels the old library.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from flux9_bf16_framework_screen import write_json
from qwen_student_sampler_raw import tensor
from qwen_tokenizer_reuse_audit import memory,pixels,snapshots
from qwen_tokenizer_reuse_raw import analyze
from runtime_ane_common import sha256_file

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--raw-six",type=Path,required=True);p.add_argument("--raw-five",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--prior-library",type=Path,default=ROOT/"build/qwen512-tokenizer-reuse-20261018/libturbocider.dylib")
    p.add_argument("--prior-full",type=Path,default=ROOT/"outputs/qwen512-tokenizer-reuse-full-20261018")
    p.add_argument("--prior-audit",type=Path,default=ROOT/"outputs/qwen512-tokenizer-reuse-audit-20261018.json")
    args=p.parse_args()
    for name in ("library","raw_six","raw_five","prior_library","prior_full","prior_audit"):
        setattr(args,name,getattr(args,name).resolve(strict=True))
    if args.output.exists():p.error("fresh safety audit output required")
    manifest=json.loads((args.library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    for item in manifest["inputs"]["sources"]:
        if sha256_file(ROOT/item["path"])!=item["sha256"]:raise ValueError("current source differs from current build")
    previous=json.loads(args.prior_audit.read_text())
    old_manifest=json.loads((args.prior_library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    if previous["status"]!="complete" or previous["library_sha256"]!=sha256_file(args.prior_library) or\
            previous["manifest_sha256"]!=sha256_file(args.prior_library.parent/"runtime-build/runtime-build-manifest.json") or\
            previous["full_summary_sha256"]!=sha256_file(args.prior_full/"summary.json"):
        raise ValueError("historical measured audit identity changed")
    old_sources=json.loads((args.prior_full/"source.json").read_text());historical_files=[]
    for item in old_manifest["inputs"]["sources"]:
        path,digest=item["path"],item["sha256"]
        if sha256_file(ROOT/path)==digest:continue
        if path not in old_sources or hashlib.sha256(old_sources[path].encode()).hexdigest()!=digest:
            raise ValueError("missing exact historical source snapshot: "+path)
        historical_files.append(path)
    if set(historical_files)!={"native/models/qwen21/pipeline.cpp","native/models/qwen21/encoder_prefill.hpp"}:
        raise ValueError("safety revision contains another native change")
    for trial in previous["observations"]:
        folder=args.prior_full/trial["name"]
        if sha256_file(folder/"report.json")!=trial["report_sha256"]:raise ValueError("historical report changed")
        for i in range(8):
            if sha256_file(folder/f"{i}.png")!=trial["png_sha256"][i] or\
                    sha256_file(folder/f"request-{i}.json")!=trial["request_sha256"][i]:
                raise ValueError("historical media/request changed")
    rows=[]
    for steps,folder,old in ((6,args.raw_six,ROOT/"outputs/qwen512-tokenizer-reuse-raw-six-20261018"),
                             (5,args.raw_five,ROOT/"outputs/qwen512-tokenizer-reuse-raw-five-20261018")):
        summary=json.loads((folder/"summary.json").read_text());old_summary=json.loads((old/"summary.json").read_text())
        if summary["status"]!="complete" or summary["steps"]!=steps or summary["performance_denominator"] is not False or\
                summary["asset_sha256"]["library"]!=sha256_file(args.library):
            raise ValueError("complete current-library parity-only capture required")
        snapshots(folder,summary);actual=analyze(folder/"trial",steps)
        if actual!=summary["analysis"]:raise ValueError("current raw audit differs")
        expected=previous["raw_summary_sha256"] if steps==6 else previous["raw_five"]["summary_sha256"]
        if sha256_file(old/"summary.json")!=expected or old_summary["asset_sha256"]["library"]!=previous["library_sha256"]:
            raise ValueError("historical raw capture changed")
        arrays=0
        for index in range(4):
            for name in ("initial","text","latents","pixels")+(("reference_0",) if index>=2 else ()):
                a=tensor(folder/"trial"/f"dump-{index}"/f"qwen21_{name}.safetensors")
                b=tensor(old/"trial"/f"dump-{index}"/f"qwen21_{name}.safetensors")
                if a[0]!=b[0] or not np.array_equal(a[1],b[1]):raise ValueError("safety revision changed actual raw output")
                arrays+=1
            if not np.array_equal(pixels(folder/"trial"/f"{index}.png"),pixels(old/"trial"/f"{index}.png")):
                raise ValueError("safety revision changed RGBA")
        rows.append(dict(steps=steps,complete_requests=4,raw_arrays_restored=arrays,all_rgba_and_raw_equal=True,
                         memory=memory(folder,summary["memory"]),load_error=summary.get("load_error"),
                         summary_sha256=sha256_file(folder/"summary.json")))
    result=dict(schema="tc-qwen-tokenizer-reuse-guard-audit-v1",status="complete",qualification_passed=False,
        current_library_sha256=sha256_file(args.library),current_runtime_build_id=manifest["runtime_build_id"],
        current_native_source_hashes_verified=len(manifest["inputs"]["sources"]),
        prior_measured_library_sha256=previous["library_sha256"],prior_audit_sha256=sha256_file(args.prior_audit),
        prior_native_sources_verified=len(old_manifest["inputs"]["sources"]),exact_archived_sources_used=historical_files,
        parity_only_captures=rows,new_performance_claim=False,script_sha256=sha256_file(Path(__file__)),
        scope="current safety revision raw/media parity and exact source binding; prior timing remains historical")
    write_json(args.output,result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
