#!/usr/bin/env python3
"""Independent full-FP16-file/F64 oracle audit of raw activation operators."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import activation_op_collect as screen
import joint_audit
from collect import ROOT,sha256_file,write_json
from vpipe_audit import trace


def expected_input(rows,channels,pattern):
    if pattern=="wide":value=((np.arange(rows*channels,dtype=np.int64)%4096)-2048).reshape(channels,rows)/128.
    elif pattern in screen.PATTERNS:
        hidden=128 if channels==128 else 4096
        value=((np.arange(rows)[None,:]*3+(np.arange(channels)[:,None]%hidden)*7)%17-8)/8.*(-.25 if pattern=="negative025" else .125)
    else:raise ValueError("known synthetic input pattern required")
    value[value==0]=0
    return value.astype("<f2")


def quality(inputs,outputs,mode):
    x=inputs.astype(np.float64);y=outputs.astype(np.float64);e=np.exp(-np.abs(x))
    sigmoid=np.where(x<0,e/(1+e),1/(1+e))
    reference=x if mode=="identity" else sigmoid if mode=="sigmoid" else x*sigmoid
    diff=y-reference
    return dict(relative_l2=float(np.sqrt(np.sum(diff*diff)/max(float(np.sum(reference*reference)),1e-300))),
        max_abs=float(np.max(np.abs(diff))),
        changed_vs_fp16_oracle=int(np.count_nonzero(reference.astype("<f2").view("<u2")!=outputs.view("<u2"))),elements=int(x.size))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("root","native-library","probe-library","native-source-archive","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error("fresh CPU operator audit output required")
    manifest=json.loads((args.native_library.parent/"runtime-build/runtime-build-manifest.json").read_text())
    joint_audit.archive_contract(args.native_source_archive,manifest)
    summary=json.loads((args.root/"summary.json").read_text());report=json.loads((args.root/"trial/report.json").read_text())
    if (summary["status"]!="complete" or summary["assets"]["native_library"]!=sha256_file(args.native_library) or
            summary["assets"]["library"]!=sha256_file(args.probe_library) or report["loaded_runtime_id"]!=manifest["runtime_build_id"] or
            sha256_file(args.root/"trial/report.json")!=summary["report_sha256"]):raise ValueError("complete bound actual raw operator/native report required")
    source=json.loads((args.root/"source.json").read_text())
    if set(source)!=set(summary["source_sha256"]):raise ValueError("complete operator tool snapshot required")
    for name,digest in summary["source_sha256"].items():
        raw=source[name].encode()
        if hashlib.sha256(raw).hexdigest()!=digest or raw!=(ROOT/name).read_bytes():raise ValueError("actual diagnostic source changed")
    if report["cpp_sha256"]!=summary["source_sha256"]["tools/profiling/qwen_base/activation_op_probe.mm"]:
        raise ValueError("actual compiled operator source differs")
    observed=json.loads((args.root/"trial/observer.json").read_text())
    if observed["source_sha256"]!=summary["source_sha256"]["tools/profiling/qwen_base/observer.mm"] or screen.validate(report,observed)!=summary["analysis"]:
        raise ValueError("actual selector/shape/control evidence differs")
    results=[];input_hashes={};restoration={}
    for row in report["samples"]:
        rows,channels=row["rows"],row["channels"];arrays=[]
        for leaf in ("input","output"):
            name=row[leaf+"_file"]
            if Path(name).name!=name or not name.endswith(".f16"):raise ValueError("bounded local FP16 array basename required")
            path=args.root/"trial"/name
            if path.stat().st_size!=rows*channels*2 or sha256_file(path)!=row[leaf+"_sha256"]:
                raise ValueError("complete unchanged FP16 array required")
            value=np.fromfile(path,dtype="<f2").reshape(channels,rows)
            if not np.isfinite(value).all():raise ValueError("finite full FP16 arrays required")
            arrays.append(value)
        x,y=arrays
        if not np.array_equal(x.view("<u2"),expected_input(rows,channels,row["pattern"]).view("<u2")):
            raise ValueError("actual synthetic input differs from independent pattern")
        key=(rows,channels,row["pattern"])
        if key in input_hashes and input_hashes[key]!=row["input_sha256"]:raise ValueError("operators received different matched inputs")
        input_hashes[key]=row["input_sha256"]
        actual=quality(x,y,row["operator"])
        if row["operator"]=="identity" and not np.array_equal(x.view("<u2"),y.view("<u2")):
            raise ValueError("raw identity bits/layout failed independent check")
        for name,value in actual.items():
            if (isinstance(value,int) and value!=row["quality"][name]) or not math.isclose(value,row["quality"][name],rel_tol=1e-8,abs_tol=1e-10):
                raise ValueError("full-file independently recomputed operator oracle error differs")
        identity=(rows,channels,row["operator"])
        if row["pattern"]=="small0125":restoration[identity]=(row["input_sha256"],row["output_sha256"])
        if row["pattern"]=="small0125-restored" and restoration[identity]!=(row["input_sha256"],row["output_sha256"]):
            raise ValueError("changed-input replay did not restore all FP16 bits")
        results.append({k:row[k] for k in ("program","scope","operator","pattern","rows","channels","input_sha256","output_sha256")}|dict(quality=actual))
    for leaf in ("report","evidence"):
        if sha256_file(args.root/summary["memory"][leaf])!=summary["memory"][leaf+"_sha256"]:raise ValueError("owned memory evidence changed")
    if not summary["memory"]["verified"]["complete"] or summary["memory"]["verified"]["command_exit_code"]!=0:
        raise ValueError("complete successful owned diagnostic window required")
    args.output.mkdir(parents=True)
    result=dict(schema="tc-activation-operator-full-array-audit-v1",status="complete",qualification_passed=False,
        native_library_sha256=sha256_file(args.native_library),probe_library_sha256=sha256_file(args.probe_library),
        native_source_archive_sha256=sha256_file(args.native_source_archive),script_sha256=sha256_file(Path(__file__)),
        summary_sha256=sha256_file(args.root/"summary.json"),all_source_patterns_verified=True,identity_bits_exact=True,
        same_program_restoration_exact=True,observer=summary["analysis"],memory=summary["memory"]["verified"],samples=results,
        scope="raw FP16 operator arrays vs independent F64 oracle, not W8 FFN error, model performance, semantic image quality or hardware placement")
    write_json(args.output/"trace.json",trace(observed));write_json(args.output/"audit.json",result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
