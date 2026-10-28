#!/usr/bin/env python3
"""CPU-only complete synthetic projection oracle and typed GPU-control audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sysconfig

import numpy as np
import dual_gup_kernel_collect as screen
import joint_audit
from collect import ROOT,sha256_file,write_json
from qkv_audit import arrays,quality


def bf16(x):
    v=np.asarray(x,dtype="<f4").view("<u4")
    bits=((v+np.uint32(0x7fff)+((v>>16)&1))>>16).astype("<u2")
    return (bits.astype("<u4")<<16).view("<f4")


def raw(value):return (value.astype("<f4",copy=False).view("<u4")>>16).astype("<u2").tobytes()


def expected_input(m,k):
    i=(np.arange(m,dtype=np.int64)[:,None]+1)*k+np.arange(k)[None,:]
    return bf16((((i%127)*17+7)%29-14)/32.)[None,:,:]


def expected_weight_hash(f,k):
    digest=hashlib.sha256()
    for first in range(0,2*f,256):
        i=(np.arange(first,min(2*f,first+256),dtype=np.int64)[:,None]+2)*k+np.arange(k)[None,:]
        value=bf16((((i%257)*37+13)%31-15)/128.)
        digest.update(raw(value))
    return digest.hexdigest()


def projections(m,k,f,first,count,changed_input=False,changed_weight=False):
    # Source values are periodic in input scalar index (127) and weight scalar
    # index (257). Compute ALL unique full-K integer dots, then expand EVERY
    # output. This is a full oracle for the fixed fixture, not sampled points.
    index=np.arange(k,dtype=np.int64)
    x=(((np.arange(127)[:,None]+index)%127)*17+7)%29-14
    w=(((np.arange(257)[:,None]+index)%257)*37+13)%31-15
    table=x@w.T
    if changed_input:
        if changed_weight:raise ValueError("one independent source change at a time")
        table=3*table+2*w.sum(axis=1)[None,:];denominator=16384.
    elif changed_weight:table=7*table+x.sum(axis=1)[:,None];denominator=32768.
    else:denominator=4096.
    rp=((np.arange(m)+1)*k)%127
    result=[]
    for offset in (first,f+first):
        wp=((np.arange(count)+offset+2)*k)%257
        result.append(bf16(table[np.ix_(rp,wp)]/denominator)[None,:,:])
    return result


def math_nonlinear(g,u):
    x=g.astype(np.float64);y=u.astype(np.float64);e=np.exp(-np.abs(x))
    s=np.where(x<0,e/(1+e),1/(1+e));return [s,x*s,x*s*y]


def audit(root,probe,native,manifest):
    summary=json.loads((root/"summary.json").read_text());report=json.loads((root/"trial/report.json").read_text())
    if (summary["status"]!="complete" or summary["assets"]["library"]!=sha256_file(probe) or summary["assets"]["native_library"]!=sha256_file(native) or
        summary["runtime_build_id"]!=manifest["runtime_build_id"] or report["loaded_runtime_id"]!=manifest["runtime_build_id"] or
        report["compiled_native_id"]!=manifest["runtime_build_id"] or sha256_file(root/"trial/report.json")!=summary["report_sha256"]):raise ValueError("complete actual bound GPU kernel/native report required")
    source=json.loads((root/"source.json").read_text())
    if set(source)!=set(summary["source_sha256"]):raise ValueError("complete source snapshot required")
    for name,digest in summary["source_sha256"].items():
        if hashlib.sha256(source[name].encode()).hexdigest()!=digest or (ROOT/name).read_text()!=source[name]:raise ValueError("bound kernel source differs")
    for field,name in (("cpp_sha256","dual_gup_kernel_probe.mm"),("header_sha256","dual_gup.hpp")):
        if report[field]!=summary["source_sha256"]["tools/profiling/qwen_base/"+name]:raise ValueError("compiled kernel/header identity differs")
    mlx_root=Path(sysconfig.get_path("purelib"))/"mlx"
    if (summary["assets"]["mlx_library"]!=sha256_file(mlx_root/"lib/libmlx.dylib") or
        summary["assets"]["mlx_unary_header"]!=sha256_file(mlx_root/"include/mlx/backend/metal/kernels/unary_ops.h")):raise ValueError("installed MLX reference identity changed")
    if screen.validate(report)!=summary["analysis"]:raise ValueError("actual complete typed kernel stage controls differ")
    if set(summary["arrays_sha256"])!={"fixture-0.safetensors","fixture-1.safetensors"}:raise ValueError("both complete fixture payloads required")
    results=[]
    for fixture,(m,k,f,first,count) in enumerate(((33,128,320,17,257),(1024,4096,12288,0,6144))):
        p=root/f"trial/fixture-{fixture}.safetensors"
        if sha256_file(p)!=summary["arrays_sha256"][p.name]:raise ValueError("kernel arrays changed")
        a=arrays(p)
        expected={"input","changed-input","weight-changed-reference"}|{f"{name}-{s}" for name in ("reference","mpp","register","shared") for s in range(5)}|{
            name+suffix for name in ("register","shared") for suffix in ("-changed","-restored","-weight-changed")}
        if set(a)!=expected or a["input"].shape!=(1,m,k) or any(not np.isfinite(v).all() for v in a.values()):raise ValueError("complete finite source/stages/state payloads required")
        x=expected_input(m,k)
        if raw(a["input"])!=raw(x) or raw(a["changed-input"])!=raw(bf16(x*.75+.015625)):raise ValueError("independent full input grid/offset differs")
        relevant=[r for r in report["cases"] if r["fixture"]==fixture];whash=expected_weight_hash(f,k)
        if any(r["weight_sha256"]!=whash or r["input_sha256"]!=hashlib.sha256(raw(x)).hexdigest() for r in relevant):raise ValueError("full original physical weight/input recipe hash differs")
        g,u=projections(m,k,f,first,count)
        for name in ("reference","mpp","register","shared"):
            if raw(a[name+"-0"])!=raw(g) or raw(a[name+"-1"])!=raw(u):raise ValueError("full integer dense-dot oracle/physical gate-up mapping differs")
        for r in relevant:
            candidate=a[("shared" if r["shared"] else "register")+"-"+str(r["stage"])]
            for label,reference in (("quality_vs_mlx",a["reference-"+str(r["stage"])]),("quality_vs_mpp",a["mpp-"+str(r["stage"])])):
                q=quality(reference,candidate)
                if q["changed"] or raw(reference)!=raw(candidate) or r[label]["bits_exact"] is not True:raise ValueError("full typed GPU controls differ")
        for name in ("register","shared"):
            if raw(a[name+"-restored"])!=raw(a[name+"-4"]) or raw(a[name+"-weight-changed"])!=raw(a["weight-changed-reference"]):raise ValueError("full A/B/A or changed-weight GPU control differs")
        checks=[]
        for prefix,gg,uu in (("reference",g,u),("changed",*projections(m,k,f,first,count,changed_input=True)),
                             ("weight-changed",*projections(m,k,f,first,count,changed_weight=True))):
            refs=math_nonlinear(gg,uu)
            values=[a["reference-"+str(s)] for s in (2,3,4)] if prefix=="reference" else [a["register-"+prefix],a["shared-"+prefix]]
            for i,value in enumerate(values):
                target=refs[i] if prefix=="reference" else refs[2];q=quality(target,value)
                if q["relative_l2"]>=.02 or not np.all(np.abs(value-target)<=.005+.02*np.abs(target)):raise ValueError("independent full nonlinear mathematical screening bound differs")
                checks.append(dict(case=prefix,quality=q))
        results.append(dict(fixture=fixture,m=m,k=k,f=f,first=first,count=count,full_projection_oracle_exact=True,full_weight_sha256=whash,
                            full_typed_gpu_controls_exact=True,independent_nonlinear=checks))
    for name in ("report","evidence"):
        if sha256_file(root/summary["memory"][name])!=summary["memory"][name+"_sha256"]:raise ValueError("owned memory evidence changed")
    if not summary["memory"]["verified"]["complete"] or summary["memory"]["verified"]["command_exit_code"]!=0:raise ValueError("complete successful owned kernel window required")
    return dict(status="complete",qualification_passed=False,fixtures=results,source_pairs=40,illegal_rejections=20,
                memory=summary["memory"]["verified"],summary_sha256=sha256_file(root/"summary.json"),
                scope="complete periodic dyadic GPU fixture oracle and typed GPU controls; no checkpoint/full FFN/image or physical instruction qualification")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","probe-library","native-library","native-source-archive","output"):p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error("fresh CPU audit directory required")
    manifest=json.loads((args.native_library.parent/"runtime-build/runtime-build-manifest.json").read_text());joint_audit.archive_contract(args.native_source_archive,manifest)
    result=audit(args.root,args.probe_library,args.native_library,manifest);result["script_sha256"]=sha256_file(Path(__file__))
    args.output.mkdir(parents=True);write_json(args.output/"audit.json",result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
