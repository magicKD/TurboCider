#!/usr/bin/env python3
"""Independent full packed-code/scale, overflow and hidden quant oracle audit."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import tarfile

import numpy as np
import hidden_quant_collect as screen
import joint_audit
from collect import ROOT,sha256_file,write_json
from rotation_audit import signs,oracle as rotation_oracle
from vpipe_audit import trace


def expected_input(channels,rows,pattern):
    if channels not in (512,6144) or rows not in (32,1024) or pattern not in screen.PATTERNS:raise ValueError("known bounded source required")
    if pattern=="zero":return np.zeros((channels,rows),dtype="<f2")
    if pattern=="tiny":return (((np.arange(channels)[:,None]*7+np.arange(rows)[None,:]*3)%17-8)*2.**-24).astype("<f2")
    if pattern=="overflow":return np.broadcast_to(signs()[np.arange(channels)%512,None]*4096,(channels,rows)).astype("<f2")
    i=np.arange(channels*rows,dtype=np.uint64);v=(i*747796405+2891336453)&np.uint64(0xffffffff)
    v=(((v>>((v>>28)+4))^v)*277803737)&np.uint64(0xffffffff);v=(v>>22)^v
    x=((v&65535).astype(np.int64)-32768).reshape(channels,rows)/32768.*(16. if pattern=="wide" else .125)
    x[x==0]=0
    return x.astype("<f2")


def read(root,name,digest,channels,rows,allow_nonfinite=False):
    if Path(name).name!=name or not name.endswith(".f16"):raise ValueError("bounded FP16 basename required")
    p=root/"trial"/name
    if p.is_symlink() or p.stat().st_size!=channels*rows*2 or sha256_file(p)!=digest:raise ValueError("complete unchanged packed/input payload required")
    value=np.fromfile(p,dtype="<f2").reshape(channels,rows)
    if not allow_nonfinite and not np.isfinite(value).all():raise ValueError("finite complete input/accepted output required")
    return value


def reference(x,rotate):
    hidden=rotation_oracle(x) if rotate else x.astype(np.float64)
    floor=np.maximum(np.max(np.abs(hidden),axis=0),2.**-12)
    codes=np.clip(np.rint(127.*hidden/floor),-127,127).astype(np.int16)
    # Exact authored FP16 128/127 scalar, and nominal packed FP16 scale edge.
    scale=(floor*float(np.float16(128./127.))).astype("<f2")
    return hidden,floor,codes,scale


def quality(x,packed,rotate,pattern):
    channels,rows=x.shape
    if packed.shape!=(channels+1,rows):raise ValueError("all normalized codes plus ONE token-scale row required")
    if pattern=="overflow" and rotate:
        hidden=rotation_oracle(x)
        if not np.all(np.max(np.abs(hidden),axis=0)>65504) or not np.all(~np.isfinite(packed[-1])):
            raise ValueError("finite source must produce actual nonfinite packed scale and reject before publication")
        return dict(expected_rejection=True,nonfinite_elements=int(np.count_nonzero(~np.isfinite(packed))),elements=packed.size)
    if not np.isfinite(packed).all():raise ValueError("accepted quantized output is not finite")
    normalized=packed[:-1].astype(np.float64);scales=packed[-1].astype(np.float64);codes=normalized*128.
    if not np.all(codes==np.rint(codes)) or np.any(np.abs(codes)>127) or not np.all(scales>0):
        raise ValueError("exact normalized int8 codes in[-127,127] and positive finite scales required")
    hidden,floor,expected_codes,expected_scale=reference(x,rotate)
    if not np.isfinite(expected_scale).all():raise ValueError("finite oracle scale required outside explicit overflow control")
    reconstructed=normalized*scales
    target=expected_codes.astype(np.float64)/128.*expected_scale.astype(np.float64)
    d=reconstructed-target;energy=float(np.sum(target*target));l2=math.sqrt(float(np.sum(d*d))/max(energy,1e-300))
    quantum=floor/127.
    violations=int(np.count_nonzero(np.abs(d)>2*quantum[None,:]+.002*np.abs(target)+1e-6))
    scale_relative=float(np.max(np.abs(scales-expected_scale.astype(np.float64))/expected_scale.astype(np.float64)))
    # A tiny INPUT is not necessarily an all-zero quantizer target after
    # H512. Do not hide any whole-expression discrepancy behind that label.
    zero_domain=pattern=="zero" or (pattern=="tiny" and not rotate)
    if zero_domain and (np.count_nonzero(codes) or np.count_nonzero(expected_codes) or not np.array_equal(packed[-1].view("<u2"),expected_scale.view("<u2"))):
        raise ValueError("intentional floor-dominated zero/tiny domain must retain exact zero codes and original scale")
    if pattern=="tiny" and not np.array_equal(packed[-1].view("<u2"),expected_scale.view("<u2")):
        raise ValueError("tiny floor-dominated token scale changed")
    return dict(expected_rejection=False,elements=packed.size,requantized_relative_l2=l2,requantized_max_abs=float(np.max(np.abs(d))),
        pointwise_violations=violations,max_scale_relative=scale_relative,changed_codes_vs_oracle=int(np.count_nonzero(codes!=expected_codes)),
        max_code_error=float(np.max(np.abs(codes-expected_codes))),screening_passed=l2<.02 and violations==0 and scale_relative<.01,
        scope="independent mathematical H512/raw->RNE int8->nominal FP16 hscale reconstruction; not an internal tap or complete FFN/image quality")


def operator_trace(raw,report,repeats):
    screen.observer_contract(raw,report,repeats)
    good={**raw,"events":[e for e in raw["events"] if e["ok"]]};good["counts"]=dict(Counter(e["name"] for e in good["events"]))
    result=trace(good);epoch=min(e["begin_ns"] for e in good["events"])
    for e in raw["events"]:
        if not e["ok"]:result["traceEvents"].append(dict(name="rejected setup compile",cat="capability-setup-rejection",ph="X",pid=1,tid=e["thread"],
            ts=(e["begin_ns"]-epoch)/1000,dur=(e["end_ns"]-e["begin_ns"])/1000,args=dict(ordinal=e["request"]//1000,ok=False)))
    return result


def window(root,probe,native,manifest,historical):
    summary=json.loads((root/"summary.json").read_text());report=json.loads((root/"trial/report.json").read_text())
    if (summary["status"]!="complete" or summary["assets"]["native_library"]!=sha256_file(native) or summary["assets"]["library"]!=sha256_file(probe) or
        summary["runtime_build_id"]!=manifest["runtime_build_id"] or report["compiled_native_id"]!=manifest["runtime_build_id"] or
        report["loaded_runtime_id"]!=manifest["runtime_build_id"] or sha256_file(root/"trial/report.json")!=summary["report_sha256"]):
        raise ValueError("complete bound actual helper/native/source report required")
    snapshot=json.loads((root/"source.json").read_text())
    if set(snapshot)!=set(summary["source_sha256"]):raise ValueError("complete tool source snapshot required")
    for name,digest in summary["source_sha256"].items():
        raw=snapshot[name].encode()
        if hashlib.sha256(raw).hexdigest()!=digest or not (raw==(ROOT/name).read_bytes() or raw in historical.get(name,set())):raise ValueError("bound tool current/archive source differs")
    for field,file in (("cpp_sha256","hidden_quant_probe.mm"),("header_sha256","hidden_quant_mil.hpp"),("rotation_header_sha256","rotation_mil.hpp")):
        if report[field]!=summary["source_sha256"]["tools/profiling/qwen_base/"+file]:raise ValueError("compiled helper/header source differs")
    timings=screen.validate(report,summary["repeats"])
    if timings!=summary["analysis"]:raise ValueError("complete timing/count/negative recheck differs")
    observed=None
    if summary["mode"]=="timeline":
        raw=json.loads((root/"trial/observer.json").read_text())
        if raw["source_sha256"]!=summary["source_sha256"]["tools/profiling/qwen_base/observer.mm"]:raise ValueError("actual observer source differs")
        observed=screen.observer_contract(raw,report,summary["repeats"])
        if observed!=summary["observer"]:raise ValueError("actual selector evidence differs")
    for name in ("report","evidence"):
        if sha256_file(root/summary["memory"][name])!=summary["memory"][name+"_sha256"]:raise ValueError("owned memory evidence changed")
    if not summary["memory"]["verified"]["complete"] or summary["memory"]["verified"]["command_exit_code"]!=0:raise ValueError("complete successful owned window required")
    cache={};restored={};records=[];eligible={};hashes={};references={}
    for row in report["arrays"]:
        c,m=row["channels"],row["rows"]
        if row["input_file"] not in cache:
            x=read(root,row["input_file"],row["input_sha256"],c,m)
            if not np.array_equal(x.view("<u2"),expected_input(c,m,row["pattern"]).view("<u2")):raise ValueError("actual complete input differs from independent pattern")
            cache[row["input_file"]]=x
        else:
            x=cache[row["input_file"]]
            if sha256_file(root/"trial"/row["input_file"])!=row["input_sha256"]:raise ValueError("shared actual source differs")
        y=read(root,row["output_file"],row["output_sha256"],c+1,m,row["expected_rejection"])
        q=quality(x,y,row["rotate"],row["pattern"])
        if q["expected_rejection"]!=row["expected_rejection"]:raise ValueError("actual output did not reproduce finite-overflow negative")
        if not q["expected_rejection"]:eligible[row["ordinal"]]=eligible.get(row["ordinal"],True) and q["screening_passed"]
        if row["pattern"]=="random-a":restored[row["ordinal"]]=(row["input_sha256"],row["output_sha256"])
        if row["pattern"]=="random-a-restored" and restored[row["ordinal"]]!=(row["input_sha256"],row["output_sha256"]):raise ValueError("full post-overflow source/output replay differs")
        key=(c,m,row["rotate"],row["pattern"])
        if row["policy"]=="divide":references[key]=y.copy()
        elif key not in references:raise ValueError("original matched quantization control missing")
        baseline=references[key]
        exact=bool(np.array_equal(y.view("<u2"),baseline.view("<u2")))
        hashes[(c,m,row["rotate"],row["policy"],row["pattern"])]=row["output_sha256"]
        records.append({k:row[k] for k in ("ordinal","policy","rotate","pattern","rows","channels","validation_flags","input_sha256","output_sha256")}|dict(quality=q,packed_exact_vs_original=exact))
    comparisons=[]
    for p in report["programs"]:
        if p["status"]!="ready":continue
        medians=timings["median_seconds"][f"m{p['rows']}-c{p['channels']}-{'rotate' if p['rotate'] else 'raw'}"]
        comparisons.append(dict(rows=p["rows"],channels=p["channels"],rotate=p["rotate"],policy=p["policy"],numeric_eligible=eligible[p["ordinal"]],
            seconds=medians[p["policy"]],reduction_vs_divide=1-medians[p["policy"]]/medians["divide"] if eligible[p["ordinal"]] else None))
    return dict(root=root,summary=summary,report=report,hashes=hashes,receipt=dict(cohort=root.name,mode=summary["mode"],repeats=summary["repeats"],
        timings=timings,comparisons=comparisons,arrays=records,observer=observed,memory=summary["memory"]["verified"],load_error=summary.get("load_error"),
        summary_sha256=sha256_file(root/"summary.json")))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("native-library","native-source-archive","output"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--cohort",nargs=2,metavar=("ROOT","PROBE_LIBRARY"),action="append",required=True)
    p.add_argument("--tool-source-archive",type=Path,action="append",default=[]);args=p.parse_args()
    if args.output.exists():p.error("fresh independent audit output required")
    manifest=json.loads((args.native_library.parent/"runtime-build/runtime-build-manifest.json").read_text());joint_audit.archive_contract(args.native_source_archive,manifest)
    historical={}
    for path in args.tool_source_archive:
        with tarfile.open(path,"r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():historical.setdefault(member.name,set()).add(archive.extractfile(member).read())
    values=[window(Path(root),Path(probe),args.native_library,manifest,historical) for root,probe in args.cohort]
    seen={};replays=0
    for value in values:
        for key,digest in value["hashes"].items():
            if key in seen:
                if seen[key]!=digest:raise ValueError("same-program repeat/timeline complete output changed")
                replays+=1
            seen[key]=digest
    args.output.mkdir(parents=True)
    for value in values:
        if value["summary"]["mode"]=="timeline":write_json(args.output/(value["root"].name+"-trace.json"),operator_trace(json.loads((value["root"]/"trial/observer.json").read_text()),value["report"],value["summary"]["repeats"]))
    result=dict(schema="tc-hidden-quant-audit-v1",status="complete",qualification_passed=False,runtime_build_id=manifest["runtime_build_id"],
        native_library_sha256=sha256_file(args.native_library),native_source_archive_sha256=sha256_file(args.native_source_archive),
        audit_source_sha256={name:sha256_file(ROOT/"tools/profiling/qwen_base"/name) for name in
            ("hidden_quant_audit.py","hidden_quant_collect.py","rotation_audit.py","collect.py","joint_audit.py")},
        script_sha256=sha256_file(Path(__file__)),same_program_exact_replays=replays,cohorts=[v["receipt"] for v in values],
        scope="Full synthetic hidden quant codes/scales, independent H512/raw RNE reconstruction and expected overflow; not full FFN/checkpoint/image or physical device timing qualification")
    write_json(args.output/"audit.json",result)
    print(json.dumps(dict(status="complete",cohorts=len(values),same_program_exact_replays=replays,
        comparisons=[dict(cohort=v["receipt"]["cohort"],values=v["receipt"]["comparisons"]) for v in values]),indent=2))


if __name__=="__main__":main()
