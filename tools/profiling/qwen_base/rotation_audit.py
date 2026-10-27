#!/usr/bin/env python3
"""Independent full-array/F64 Sylvester oracle and rotation timing audit."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tarfile

import numpy as np
import rotation_collect as screen
import joint_audit
from collect import ROOT,sha256_file,write_json
from vpipe_audit import trace


def signs():
    result=[];mask=(1<<64)-1
    for i in range(512):
        x=(20260930+i*0x9e3779b97f4a7c15)&mask
        x=((x^(x>>30))*0xbf58476d1ce4e5b9)&mask;x=((x^(x>>27))*0x94d049bb133111eb)&mask
        result.append(-1 if (x^(x>>31))&1 else 1)
    return np.array(result,dtype=np.float64)


def expected_input(channels,rows,pattern):
    if channels%512 or rows not in (32,1024) or pattern not in screen.PATTERNS:raise ValueError("known complete rotation input required")
    if pattern=="signed-basis":
        c=np.arange(channels)[:,None];r=np.arange(rows)[None,:]
        values=np.where(c%512==(r*13)%512,signs()[(c%512)]*.5,0.)
    else:
        index=np.arange(channels*rows,dtype=np.uint64)
        v=(index*747796405+2891336453)&np.uint64(0xffffffff)
        v=(((v>>((v>>28)+4))^v)*277803737)&np.uint64(0xffffffff);v=(v>>22)^v
        values=((v&65535).astype(np.int64)-32768).reshape(channels,rows)/32768.*(16. if pattern=="wide" else .125)
    values[values==0]=0
    return values.astype("<f2")


def oracle(x):
    """All elements; scalar CPU F64 butterfly, not generated factor matrices."""
    channels,rows=x.shape
    if channels%512:raise ValueError("whole H512 groups required")
    value=x.astype(np.float64).reshape(channels//512,512,rows)*signs()[None,:,None]
    for half in (1,2,4,8,16,32,64,128,256):
        view=value.reshape(channels//512,-1,2*half,rows)
        a=view[:,:,:half,:].copy();b=view[:,:,half:,:].copy()
        view[:,:,:half,:]=a+b;view[:,:,half:,:]=a-b
    # Exactly the existing serialized FP16 H512 coefficient. No new ideal-
    # normalization target that would hide factor intermediate rounding.
    return value.reshape(channels,rows)*(181./4096.)


def quality(reference,output):
    if reference.shape!=output.shape or not np.isfinite(reference).all() or not np.isfinite(output).all():
        raise ValueError("same-shape complete finite oracle/output required")
    x=reference.astype(np.float64);y=output.astype(np.float64);d=y-x
    error=float(np.sum(d*d));energy=float(np.sum(x*x))
    l2=math.sqrt(error/max(energy,1e-300))
    violations=int(np.count_nonzero(np.abs(d)>.002+.005*np.abs(x)))
    return dict(relative_l2=l2,max_abs=float(np.max(np.abs(d))),elements=x.size,pointwise_violations=violations,
                changed_vs_fp16_oracle=int(np.count_nonzero(x.astype("<f2").view("<u2")!=output.view("<u2"))),
                screening_passed=l2<.002 and violations==0)


def operator_trace(raw,report,repeats):
    screen.observer_contract(raw,report,repeats)
    good={**raw,"events":[r for r in raw["events"] if r["ok"]]}
    good["counts"]=dict(Counter(r["name"] for r in good["events"]))
    value=trace(good);epoch=min(r["begin_ns"] for r in good["events"])
    for r in raw["events"]:
        if not r["ok"]:value["traceEvents"].append(dict(name="rejected setup compile",cat="capability-setup-rejection",ph="X",pid=1,
            tid=r["thread"],ts=(r["begin_ns"]-epoch)/1000,dur=(r["end_ns"]-r["begin_ns"])/1000,
            args=dict(phase_index=r["request"],model=r["model"],ok=False,scope="declared compiler rejection; not timed inference")))
    return value


def read_array(root,name,digest,channels,rows):
    if Path(name).name!=name or not name.endswith(".f16"):raise ValueError("bounded local FP16 array basename required")
    p=root/"trial"/name
    if p.is_symlink() or p.stat().st_size!=channels*rows*2 or sha256_file(p)!=digest:raise ValueError("complete unchanged FP16 payload required")
    value=np.fromfile(p,dtype="<f2").reshape(channels,rows)
    if not np.isfinite(value).all():raise ValueError("finite full operator data required")
    return value


def window(root,probe,native,manifest,historical):
    summary=json.loads((root/"summary.json").read_text());report=json.loads((root/"trial/report.json").read_text())
    if (summary["status"]!="complete" or summary["assets"]["native_library"]!=sha256_file(native) or
        summary["assets"]["library"]!=sha256_file(probe) or summary["runtime_build_id"]!=manifest["runtime_build_id"] or
        report["loaded_runtime_id"]!=manifest["runtime_build_id"] or report["compiled_native_id"]!=manifest["runtime_build_id"] or
        sha256_file(root/"trial/report.json")!=summary["report_sha256"]):raise ValueError("complete actual bound helper/native report required")
    snapshot=json.loads((root/"source.json").read_text())
    if set(snapshot)!=set(summary["source_sha256"]):raise ValueError("complete source snapshot required")
    for name,digest in summary["source_sha256"].items():
        raw=snapshot[name].encode()
        if hashlib.sha256(raw).hexdigest()!=digest or not (raw==(ROOT/name).read_bytes() or raw in historical.get(name,set())):
            raise ValueError("actual current/archive source differs")
    for field,file in (("cpp_sha256","rotation_probe.mm"),("header_sha256","rotation_mil.hpp")):
        if report[field]!=summary["source_sha256"]["tools/profiling/qwen_base/"+file]:raise ValueError("compiled helper/header identity differs")
    legacy=len(report["programs"])==16
    actual=screen.validate(report,summary["repeats"],legacy=legacy)
    if actual!=summary["analysis"]:raise ValueError("independent complete timing/shape/count recheck differs")
    observed=None
    if summary["mode"]=="timeline":
        raw=json.loads((root/"trial/observer.json").read_text())
        if raw["source_sha256"]!=summary["source_sha256"]["tools/profiling/qwen_base/observer.mm"]:raise ValueError("observer compiled source differs")
        observed=screen.observer_contract(raw,report,summary["repeats"])
        if observed!=summary["observer"]:raise ValueError("actual fixed operator selector contract differs")
    for name in ("report","evidence"):
        if sha256_file(root/summary["memory"][name])!=summary["memory"][name+"_sha256"]:raise ValueError("owned memory evidence changed")
    if not summary["memory"]["verified"]["complete"] or summary["memory"]["verified"]["command_exit_code"]!=0:raise ValueError("complete successful owned window required")
    cache={};references={};restoration={};results=[];eligible={};hashes={}
    for row in report["arrays"]:
        channels,rows=row["channels"],row["rows"];pattern=row["pattern"]
        if row["input_file"] not in cache:
            x=read_array(root,row["input_file"],row["input_sha256"],channels,rows)
            if not np.array_equal(x.view("<u2"),expected_input(channels,rows,pattern).view("<u2")):raise ValueError("actual full source differs from independent pattern")
            cache[row["input_file"]]=x
        else:
            if sha256_file(root/"trial"/row["input_file"])!=row["input_sha256"]:raise ValueError("shared source SHA differs")
            x=cache[row["input_file"]]
        y=read_array(root,row["output_file"],row["output_sha256"],channels,rows)
        key=(channels,rows,"random-a" if pattern=="random-a-restored" else pattern)
        if row["arm"]=="identity":
            if not np.array_equal(x.view("<u2"),y.view("<u2")):raise ValueError("independent full identity bits/layout failed")
            q=quality(x.astype(np.float64),y)
        else:
            if key not in references:references[key]=oracle(x)
            q=quality(references[key],y)
            if pattern=="signed-basis" and q["changed_vs_fp16_oracle"]!=0:raise ValueError("signed impulse exposed wrong factor/axes/signs")
        if row["arm"]=="conv512" and not q["screening_passed"]:raise ValueError("original rotation failed full independent oracle bound")
        eligible[row["ordinal"]]=eligible.get(row["ordinal"],True) and q["screening_passed"]
        identity=(channels,rows,row["arm"])
        if pattern=="random-a":restoration[identity]=(row["input_sha256"],row["output_sha256"])
        if pattern=="random-a-restored" and restoration[identity]!=(row["input_sha256"],row["output_sha256"]):raise ValueError("full A/B/wide/A bits not restored")
        hashes[(rows,channels,row["arm"],pattern)]=row["output_sha256"]
        results.append({k:row[k] for k in ("ordinal","arm","pattern","rows","channels","input_sha256","output_sha256")}|dict(quality=q))
    comparisons=[]
    for p in report["programs"]:
        if p["status"]!="ready" or p["arm"]=="identity":continue
        medians=actual["median_seconds"][f"m{p['rows']}-c{p['channels']}"]
        comparisons.append(dict(rows=p["rows"],channels=p["channels"],arm=p["arm"],numeric_eligible=eligible[p["ordinal"]],
            seconds=medians[p["arm"]],reduction_vs_conv512=1-medians[p["arm"]]/medians["conv512"] if eligible[p["ordinal"]] else None))
    return dict(summary=summary,hashes=hashes,receipt=dict(cohort=root.name,mode=summary["mode"],repeats=summary["repeats"],
        comparisons=comparisons,samples=results,timing=actual,observer=observed,memory=summary["memory"]["verified"],
        load_error=summary.get("load_error"),summary_sha256=sha256_file(root/"summary.json")))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("native-library","native-source-archive","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--cohort",nargs=2,metavar=("ROOT","PROBE_LIBRARY"),action="append",required=True)
    p.add_argument("--tool-source-archive",type=Path,action="append",default=[]);args=p.parse_args()
    if args.output.exists():p.error("fresh audit output required")
    manifest=json.loads((args.native_library.parent/"runtime-build/runtime-build-manifest.json").read_text());joint_audit.archive_contract(args.native_source_archive,manifest)
    historical={}
    for path in args.tool_source_archive:
        with tarfile.open(path,"r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():historical.setdefault(member.name,set()).add(archive.extractfile(member).read())
    values=[window(Path(root),Path(probe),args.native_library,manifest,historical) for root,probe in args.cohort]
    seen={};replay_count=0
    for value in values:
        for key,digest in value["hashes"].items():
            recipe=(value["summary"]["source_sha256"]["tools/profiling/qwen_base/rotation_mil.hpp"],value["summary"]["source_sha256"]["tools/profiling/qwen_base/rotation_probe.mm"],key)
            if recipe in seen:
                if seen[recipe]!=digest:raise ValueError("same-recipe repeated/timeline full operator output differs")
                replay_count+=1
            seen[recipe]=digest
    args.output.mkdir(parents=True)
    result=dict(schema="tc-h512-rotation-audit-v1",status="complete",qualification_passed=False,runtime_build_id=manifest["runtime_build_id"],
        native_library_sha256=sha256_file(args.native_library),native_source_archive_sha256=sha256_file(args.native_source_archive),
        tool_archive_sha256={str(x):sha256_file(x) for x in args.tool_source_archive},script_sha256=sha256_file(Path(__file__)),
        same_recipe_exact_replays=replay_count,cohorts=[v["receipt"] for v in values],
        scope="Full synthetic FP16 arrays/F64 Sylvester oracle and actual operator boundaries; no complete FFN/image, checkpoint quality, physical INT8/ANE duration or exclusive-device qualification")
    for root,_ in args.cohort:
        if (Path(root)/"trial/observer.json").exists():
            report=json.loads((Path(root)/"trial/report.json").read_text())
            write_json(args.output/(Path(root).name+"-trace.json"),operator_trace(json.loads((Path(root)/"trial/observer.json").read_text()),report,report["repeats"]))
    # The complete receipt is published only after all requested exports pass.
    write_json(args.output/"audit.json",result)
    print(json.dumps(dict(status="complete",cohorts=len(values),same_recipe_exact_replays=replay_count,
        comparisons=[dict(cohort=v["receipt"]["cohort"],values=v["receipt"]["comparisons"]) for v in values]),indent=2))


if __name__=="__main__":main()
