#!/usr/bin/env python3
"""Read-only independent verifier/curator for explicit GPU-affine reports.

No benchmark module imports or device calls. Recompute data-level gate/ratio
consistency, retain all samples and verify source snapshots. This does not rerun
the original oracle, prove genuine vectors, or confer request qualification.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local(root, name):
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("unregistered or escaping report asset")
    return path


def statistics_for(report):
    if report["status"] != "same_recipe_numeric_pass":
        return None
    for sample in report["samples"]:
        metrics = [sample["vs_same_recipe_fp32"],sample["vs_independent_fp64"]]
        passed = all(value.get("finite") is True and type(value.get("relative_l2")) in (int,float)
                     and math.isfinite(value["relative_l2"]) and 0 <= value["relative_l2"] <= .02 for value in metrics)
        if not passed or sample.get("same_recipe_numerical_pass") is not True:
            raise ValueError("claimed numeric pass contradicts recorded metrics")
        original = sample["vs_original_bf16_replica"]
        quality = None if original is None else (original.get("finite") is True and original["relative_l2"] is not None
                   and math.isfinite(original["relative_l2"]) and original["relative_l2"] <= .03 and original["cosine"] >= .999)
        if sample["original_component_budget_pass"] is not quality:
            raise ValueError("original-source quality marker contradicts recorded error")
    expected = report["repeats"]
    quality = all(sample["original_component_budget_pass"] is True for sample in report["samples"]) if not report["synthetic"] else None
    if report.get("original_component_budget_pass") is not quality:
        raise ValueError("global original-source quality marker disagrees with its samples")
    warm = [sample for sample in report["samples"] if sample["warmup"] is False]
    if len(warm) != expected or {sample["trial"] for sample in warm} != set(range(expected)):
        raise ValueError("incomplete/duplicate measured trials")
    baseline = report["baseline"]
    pairs = {}
    for sample in warm:
        wall = sample["wall_seconds"]
        if type(wall) not in (int,float) or not math.isfinite(wall) or wall <= 0:
            raise ValueError("invalid measured wall")
        arms = pairs.setdefault(sample["pair"],{})
        if sample["variant"] in arms:
            raise ValueError("duplicate paired arm")
        arms[sample["variant"]] = sample
    ratios = []
    for arms in pairs.values():
        if set(arms) != {baseline,"candidate"} or arms[baseline]["point"] != arms["candidate"]["point"]:
            raise ValueError("missing paired arm or mismatched input")
        ratios.append(arms[baseline]["wall_seconds"]/arms["candidate"]["wall_seconds"])
    if baseline != "packed" and not all(sample["original_component_budget_pass"] is True for sample in report["samples"]):
        if "statistics" in report:
            raise ValueError("original-model ratio exposed despite failed source-quality budget")
        return None
    rng = random.Random(37)
    boot = sorted(statistics.median(rng.choices(ratios,k=len(ratios))) for _ in range(5000))
    result = {"source_kind":"derived","pairs":len(ratios),"paired_ratios":ratios,
              "median_paired_speedup":statistics.median(ratios),"exploratory_bootstrap_95_percentile":[boot[124],boot[4874]],
              "median_wall_seconds":{name:statistics.median(sample["wall_seconds"] for sample in warm if sample["variant"]==name) for name in (baseline,"candidate")},
              "scope":"same-encoded storage implementation only" if baseline=="packed" else "original-source component only, not full request"}
    if "statistics" not in report or result["paired_ratios"] != report["statistics"]["paired_ratios"]:
        raise ValueError("raw statistics disagree with independent paired recomputation")
    return result


def curate(root, runs, diagnostics, failed_launches=(), fixture=None):
    root = root.resolve()
    result = {"schema":"tc-gpu-affine-evidence-v1","date":"2026-10-16","production_qualified":False,
              "scope":"real full Qwen layer0 FFN plus separately identified small format/kernel diagnostics",
              "curator_sha256":sha(Path(__file__)),"runs":[],"diagnostics":[],"failed_setup_attempts":[],
              "metric_provenance":{"wall_seconds":{"source_kind":"measured","unit":"seconds","device_kernel_time":None},
                                   "paired_ratios":{"source_kind":"derived","scope":"within each declared pair"},
                                   "logical_memory_bytes":{"source_kind":"derived","scope":"logical target allocations, not physical traffic or hard memory bound"},
                                   "error_metrics":{"source_kind":"derived","scope":"actual outputs against recorded references"},
                                   "allocator_peak":{"source_kind":"api-reported","scope":"whole child including references, not per-arm peak"}},
              "limitations":["Layer0 and one prompt/seed trajectory only; no LoRA, request-image quality or other SKU qualification.",
                             "Same packed bytes do not imply bitwise identical dequantization or reduction.",
                             "Cache preparation and added residency are separate from warm latency.",
                             "No DRAM/bank-stall counters, instruction attribution or whole-GGML engine benchmark.",
                             "Initial original-BF16 discovery had a legacy FP16 boundary label, despite explicit original_source metadata and BF16 ABI assertions; later confirmation corrects the label.",
                             "Input/weights are immutable per child; cache is not a production cross-checkpoint/LoRA cache.",
                             "All variants, samples, warmups and programming-error attempts remain local; no raw model/media embedded."]}
    expected_gates={"full_relative_l2_max":.02,"independent_sample_relative_l2_max":.02,"original_bf16_relative_l2_max":.03,"original_bf16_cosine_min":.999}
    fixture_sha=None
    if fixture:
        fixture_path=local(root,fixture)
        fixture_data=json.loads(fixture_path.read_text())
        fixture_sha=sha(fixture_path)
        result["fixture_provenance"]={"fixture_sha256":fixture_sha,
            **{key:fixture_data[key] for key in ("shape_mhf","source_layer","source_encoding","original_gate_up_tensor_sha256",
                "original_down_tensor_sha256","capture_receipt_sha256","native_capture_runtime_build_id","points","files","original_reference_scope")}}
    for group,name in runs:
        path=local(root,name+"/report.json")
        value=json.loads(path.read_text())
        if value.get("schema")!="tc-quantized-gpu-experiment-v1" or value.get("phase")!="terminal" or value["gates"]!=expected_gates:
            raise ValueError("unsupported/nonterminal run or changed gates")
        if not value["synthetic"] and (fixture_sha is None or value["fixture_sha256"]!=fixture_sha):
            raise ValueError("real run requires explicit matching fixture provenance")
        for item in value["source_snapshots"]:
            if sha(local(path.parent,item["path"]))!=item["sha256"]:
                raise ValueError("source snapshot changed")
        run={key:val for key,val in value.items() if key not in ("errors","statistics")}
        run["errors"]=[{"phase":error["phase"],"type":error["type"],"classification":"see retained raw report; not dropped"} for error in value["errors"]]
        run.update(id=name,group=group,raw_report_sha256=sha(path),independent_statistics=statistics_for(value))
        result["runs"].append(run)
    for name in diagnostics:
        path=local(root,name+"/report.json")
        value=json.loads(path.read_text())
        for item in value.get("source_snapshots",[]):
            if sha(local(path.parent,item["path"]))!=item["sha256"]:
                raise ValueError("diagnostic source snapshot changed")
        result["diagnostics"].append({"id":name,"raw_report_sha256":sha(path),"data":value})
    for name in failed_launches:
        receipt=local(root,name+"/receipt.json")
        stderr=local(root,name+"/stderr.log")
        launch=json.loads(receipt.read_text())
        if launch["status"]!="failed":
            raise ValueError("failed-attempt receipt is not a failure")
        text=stderr.read_text()
        result["failed_setup_attempts"].append({"id":name,"receipt_sha256":sha(receipt),"stderr_sha256":sha(stderr),
                                                "returncode":launch["returncode"],"classification":"frontend-numpy-scalar-initialization-error" if "Invalid type float16 received" in text else "unclassified-see-raw",
                                                "hardware_unsupported_conclusion":False,"valid_dequant_or_matmul_result":False})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--fixture",help="explicit relative fixture.json for real runs")
    parser.add_argument("--run",action="append",required=True,help="group:relative-run-name")
    parser.add_argument("--diagnostic",action="append",default=[])
    parser.add_argument("--failed-launch",action="append",default=[])
    parser.add_argument("--compact",action="store_true")
    args=parser.parse_args()
    runs=[]
    for value in args.run:
        group,separator,name=value.partition(":")
        if not separator or group not in ("smoke","discovery","confirmation") or not re.fullmatch(r"[a-z0-9-]+",name):
            parser.error("expected explicit group:run-name")
        runs.append((group,name))
    if len({name for _,name in runs})!=len(runs):
        parser.error("duplicate run")
    print(json.dumps(curate(args.root,runs,args.diagnostic,args.failed_launch,args.fixture),indent=None if args.compact else 2,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
