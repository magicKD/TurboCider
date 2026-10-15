#!/usr/bin/env python3
"""Independent read-only attention receipt/statistics verifier; no framework imports."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics

GATES={"full_original_relative_l2_max":.01,"independent_row_relative_l2_max":.01,"empty_row_max_abs":1e-6}


def digest(path):
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):
            value.update(chunk)
    return value.hexdigest()


def local(root,name):
    relative=Path(name);path=(root/relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("selected evidence missing or escaping its root")
    return path


def read(path):
    if path.stat().st_size>32*1024*1024:raise ValueError("bounded JSON required")
    return json.loads(path.read_text(encoding="utf-8"))


def metric_pass(metric):
    if not isinstance(metric,dict):return False
    value=metric.get("relative_l2")
    return metric.get("finite") is True and type(value) in (int,float) and math.isfinite(value) and 0<=value<=.01


def verify_statistics(document):
    if document.get("schema")!="tc-attention-experiment-v1" or document.get("gates")!=GATES:
        raise ValueError("fixed attention schema and numerical gates required")
    if (type(document.get("synthetic")) is not bool or document.get("production_qualified") is not False
            or document.get("physical_device_timestamps") is not None
            or document.get("candidate") not in ("single-mask","contiguous-v","head-chunks","noop")):
        raise ValueError("explicit candidate and host-only qualification scope required")
    rows=document["samples"]
    if [row["trial"] for row in rows]!=list(range(-4,8)):
        raise ValueError("complete ordered warmups and eight measured trials required")
    passed=True;pairs={}
    for row in rows:
        trial=row["trial"];expected="candidate" if trial%4 in (1,2) else "baseline"
        if (row["warmup"] is not (trial<0) or row["variant"]!=expected or row["pair"]!=trial//2
                or row["point"]!=document["point"] or row.get("source_kind")!="measured"
                or type(row["wall_ns"]) is not int or row["wall_ns"]<=0):
            raise ValueError("attention ABBA order, point or measured boundary differs")
        valid=metric_pass(row["independent_rows"])
        original=row.get("vs_original_replay")
        if document["synthetic"] is False:
            valid=valid and metric_pass(original)
        elif original is not None:
            raise ValueError("synthetic case cannot inherit original model evidence")
        empty=row["empty_row_max_abs"]
        valid=valid and type(empty) in (int,float) and math.isfinite(empty) and 0<=empty<=1e-6
        if row.get("component_numeric_pass") is not valid:
            raise ValueError("numerical gate marker disagrees with independent fields")
        passed=passed and valid
        if trial>=0:
            pair=pairs.setdefault(row["pair"],{})
            if row["variant"] in pair:raise ValueError("duplicate attention paired arm")
            pair[row["variant"]]=row
    if document.get("gpu_inputs_unchanged") is not True or document.get("source_inputs_unchanged") is not True:
        raise ValueError("changed operands/source cannot receive a score")
    if not passed:
        if document.get("statistics") is not None or document.get("status")!="component-numeric-rejected":
            raise ValueError("failed attention contract cannot expose a speed ratio")
        return None
    if document.get("status")!="component-numeric-pass":raise ValueError("numeric state differs")
    ratios=[pair["baseline"]["wall_ns"]/pair["candidate"]["wall_ns"] for pair in pairs.values()]
    reported=document.get("statistics")
    if not reported or reported.get("paired_ratios")!=ratios or reported.get("median_paired_host_ratio")!=statistics.median(ratios):
        raise ValueError("reported ratios differ from independently paired observations")
    rng=random.Random(37)
    bootstrap=sorted(statistics.median(rng.choices(ratios,k=len(ratios))) for _ in range(5000))
    return {"source_kind":"derived","paired_ratios":ratios,"median_paired_host_ratio":statistics.median(ratios),
            "exploratory_bootstrap_95_percentile":[bootstrap[124],bootstrap[4874]],
            "uncertainty_method":"4 paired ratios, 5000 seeded resamples, seed37; exploratory, not guaranteed coverage",
            "variant_median_ms":{arm:statistics.median(row["wall_ns"] for row in rows if row["variant"]==arm and not row["warmup"])/1e6
                                 for arm in ("baseline","candidate")},"production_qualified":False,
            "identity_control":document["candidate"]=="noop"}


def verify_snapshots(folder,items):
    for item in items:
        if digest(local(folder,item["path"]))!=item["sha256"]:
            raise ValueError("source snapshot changed")


def verify_source_link(value,capture,group):
    if not capture or capture["seed"]!=value["source_identity"]["source_seed"]:
        raise ValueError("real run source is not a verified selected capture")
    if group=="confirmation" and capture["seed"]!=43:raise ValueError("confirmation source seed differs")
    if group in ("discovery","diagnostic") and capture["seed"]!=42:raise ValueError("discovery source seed differs")
    original=[item for item in capture["driver_report"]["records"] if item["phase"]==value["point"]]
    if len(original)!=1 or value["source_record"]!=original[0] or value["segments"]!=original[0]["segments"]:
        raise ValueError("candidate original phase/segments differ from selected capture")
    tensors={item["name"]:item for item in original[0]["tensors"]}
    if any(value["shape_"+name]!=tensors[name]["shape"] for name in ("q","k","v")):
        raise ValueError("real attention shape differs; no truncated-head/query surrogate")
    for layout in value["gpu_layouts"]:
        if layout["actual_element_strides"]!=tensors[layout["name"]]["replay_element_strides"]:
            raise ValueError("real replay physical layout differs")


def curate(root,captures,runs,diagnostics):
    result={"schema":"tc-qwen-attention-evidence-v1","date":"2026-10-16","curator_sha256":digest(Path(__file__)),
            "production_qualified":False,"captures":[],"runs":[],"diagnostics":[]}
    allowed_receipts={}
    for identity in captures:
        folder=local(root,identity+"/receipt.json").parent;path=folder/"receipt.json";value=read(path)
        if value.get("status")!="captured-replay-with-controls" or value.get("model_source_stats_unchanged") is not True or value.get("source_inputs_unchanged") is not True:
            raise ValueError("source-stable completed replay capture required")
        if value["build_inputs_verified"]!=value["build_inputs_verified_after"] or value["driver_report"]["loaded_runtime_id"]!=value["runtime_build_id"]:
            raise ValueError("capture native build identity differs")
        verify_snapshots(folder,value["source_snapshots"])
        if digest(folder/"capture-qwen-attention")!=value["binary_sha256"]:raise ValueError("capture binary changed")
        total=0
        for item in value["artifacts"]:
            asset=local(folder,item["path"])
            if digest(asset)!=item["sha256"] or asset.stat().st_size!=item["bytes"]:raise ValueError("captured tensor file changed")
            total+=item["bytes"]
        if total!=value["driver_report"]["raw_capture_bytes"] or total>128*1024*1024:raise ValueError("capture bytes differ")
        for record in value["driver_report"]["records"]:
            if record["full_block_relative_l2"]>.001 or record["actual_prefix_exact"] is not True or record["source_inputs_unchanged"] is not True:
                raise ValueError("original-input full-block/prefix gate failed")
        record={"id":identity,"raw_receipt_sha256":digest(path),"capture_kind":value["capture_kind"],
                **{key:value[key] for key in ("runtime_build_id","runtime_manifest_sha256","native_library_sha256","build_inputs_verified",
                    "source_snapshots","binary_sha256","seed","maximum_tensor_bytes","driver_report","artifacts","model_source_stats_unchanged","source_inputs_unchanged")}}
        result["captures"].append(record);allowed_receipts[record["raw_receipt_sha256"]]=record
    for selection in runs:
        group,identity=selection.split(":")
        if group not in ("control","discovery","confirmation","diagnostic"):
            raise ValueError("unknown campaign selection group")
        folder=local(root,identity+"/report.json").parent;path=folder/"report.json";value=read(path)
        verify_snapshots(folder,value["source_snapshots"])
        stats=verify_statistics(value)
        if (value["synthetic"] is True)!=(group=="control") or (value["candidate"]=="noop")!=(group=="diagnostic"):
            raise ValueError("synthetic and identity controls cannot inherit a real candidate group")
        if value["synthetic"] is False:
            capture=allowed_receipts.get(value["source_identity"]["receipt_sha256"])
            verify_source_link(value,capture,group)
        record={"id":identity,"group":group,"raw_report_sha256":digest(path),"independent_statistics":stats,
                "input_digest_encoding":"exact BF16 values represented in CPU FP32; distinct from captured BF16 byte hashes",
                **{key:item for key,item in value.items() if key not in ("errors","statistics")}}
        result["runs"].append(record)
    for identity in diagnostics:
        folder=local(root,identity+"/report.json").parent;path=folder/"report.json";value=read(path)
        if value.get("status")!="diagnostic-collected" or value.get("formal_performance_score") is not False:
            raise ValueError("diagnostic collection is not performance qualification")
        if digest(folder/"attention_empty_probe.py")!=value["runner_sha256"]:raise ValueError("diagnostic runner changed")
        result["diagnostics"].append({"id":identity,"raw_report_sha256":digest(path),"report":value})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--capture",action="append",default=[])
    parser.add_argument("--run",action="append",default=[],help="control|discovery|confirmation|diagnostic:run-id")
    parser.add_argument("--diagnostic",action="append",default=[])
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    try:
        names=[*args.capture,*args.diagnostic,*(name for selected in args.run for name in selected.split(":"))]
        if any(not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,127}",name) for name in names):raise ValueError("bounded local IDs required")
        if len(args.run)!=len(set(args.run)) or len(args.capture)!=len(set(args.capture)):raise ValueError("duplicate selection")
        value=curate(args.root.resolve(),args.capture,args.run,args.diagnostic)
        encoded=json.dumps(value,indent=2,allow_nan=False)+"\n"
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x",encoding="utf-8") as stream:stream.write(encoded)
    except (ValueError,KeyError,OSError) as error:parser.error(str(error))
    print("verified explicit attention sources, all paired samples, numerical rejections and diagnostic-only observations")
    return 0


if __name__=="__main__":raise SystemExit(main())
