#!/usr/bin/env python3
"""Publish the completed five-route Z cohort; no model loading or new experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from flux9_bf16_framework_screen import write_json
from flux9_framework_evidence import portable
from qwen21_mflux_framework_evidence import rgba,compare_rgba
from runtime_ane_common import sha256_file
from unsloth_image_reference_screen import placement,validate_generation
from z_unsloth_framework_screen import ROUTES,validate_cohort

ROOT = Path(__file__).resolve().parents[2]


def compact_trial(trial):
    result = dict(index=trial["index"],route=trial["route"],memory=trial["memory"],png_sha256=trial["png_sha256"])
    report = dict(trial["report"])
    metrics_keys = ("model","operation","width","height","steps","seed","text_tokens",
        "prompt_cache_hit","lora_strategy","encoder_execution","runtime_backend","runtime_precision","timings_seconds")
    hybrid_keys = ("ane_mlp_range","bucket","hidden","mlp_width","block_count","compute_units","weight_variant",
        "checkpoint_sha256_verified","runtime_calls_session_total","runtime_failures_session_total","runtime_failed","warmup_calls_session_total")
    rows = []
    for original in report["runs"]:
        row = dict(original)
        if "metrics" in row:
            metrics = row["metrics"]
            row["metrics"] = {key:metrics.get(key) for key in metrics_keys}
            row["metrics"]["hybrid"] = {key:metrics["hybrid"].get(key) for key in hybrid_keys} if metrics.get("hybrid") else None
        rows.append(row)
    report["runs"] = rows
    result["report"] = report
    return portable(result)


def audit(screen):
    data = validate_cohort(json.loads((screen / "summary.json").read_text()))
    routes = []
    checks = []
    for route in ROUTES:
        trials = [trial for trial in data["trials"] if trial["route"] == route]
        for trial in trials:
            output = screen / f"{trial['index']}-{route}"
            memory = trial["memory"]
            if (sha256_file(output / memory["evidence"]) != memory["evidence_sha256"] or
                    sha256_file(output / memory["report"]) != memory["report_sha256"]):
                raise ValueError("actual independent memory chain differs")
            if route.startswith("unsloth-"):
                mode = "gpu" if route == "unsloth-gpu-text" else "default"
                placement((output / "server.stdout.txt").read_text()+(output / "server.stderr.txt").read_text(),mode)
                for index,row in enumerate(trial["report"]["runs"]):
                    payload = json.loads((output / f"request-{index}.json").read_text())
                    actual = validate_generation(row["info"],payload,(output / f"request-{index}.log").read_text())
                    if any(row[key] != value for key,value in actual.items()):
                        raise ValueError("reported stage wall differs from original log")
            for index,digest in enumerate(trial["png_sha256"]):
                image = output / f"{index}.png"
                rgba(image)
                if sha256_file(image) != digest:
                    raise ValueError("actual full PNG differs")
        for index in range(4):
            first = screen / f"{trials[0]['index']}-{route}" / f"{index}.png"
            last = screen / f"{trials[1]['index']}-{route}" / f"{index}.png"
            checks.append(dict(route=route,index=index,rgba_equal=bool(np.array_equal(rgba(first),rgba(last)))))
        median = data["medians"][route]
        routes.append(dict(route=route,fresh_warm_seconds=median["samples"],fresh_warm_median=median["fresh_warm_seconds"],
            cold_seconds=median["cold_seconds"],repeat_seconds=median["repeat_seconds"],constructor_seconds=median["constructor_seconds"],
            physical_peaks=[trial["memory"]["verified"]["tree_peak_phys_footprint_bytes"] for trial in trials],
            compression_bytes=[trial["memory"]["verified"]["compression_bytes"] for trial in trials],
            swap_out_bytes=[trial["memory"]["verified"]["swap_out_bytes"] for trial in trials]))
    gpu = [trial for trial in data["trials"] if trial["route"] == "native-gpu"]
    mixed = [trial for trial in data["trials"] if trial["route"] == "native-frozen"]
    quality = [dict(prompt_index=index,direction=direction,**compare_rgba(
        rgba(screen / f"{gpu[direction]['index']}-native-gpu" / f"{index}.png"),
        rgba(screen / f"{mixed[direction]['index']}-native-frozen" / f"{index}.png")))
        for direction in range(2) for index in range(4)]
    medians = {row["route"]:row["fresh_warm_median"] for row in routes}
    best = min(("unsloth-gpu-text","unsloth-cpu-text"),key=medians.get)
    result = dict(schema="tc-z-unsloth-five-route-validation-v1",date="2026-10-12",status="complete_diagnostic",
        qualification_passed=False,goal_complete=False,reference_latency_ranking_valid=True,
        hardware="M4 Max64GiB/macOS26.6.2",steps=8,requests_per_process=4,order=data["order"],routes=routes,
        library_sha256=data["library_sha256"],native_inputs_matched=data["native_inputs_matched"],
        model_sha256={portable(key):value for key,value in data["model_sha256"].items()},
        source_snapshot_sha256=hashlib.sha256(json.dumps({portable(key):value for key,value in data["source_sha256"].items()},sort_keys=True).encode()).hexdigest(),
        unsloth_commit=data["unsloth_commit"],unsloth_backend_build=data["unsloth_backend_build"],mflux_commit=data["mflux_commit"],
        timing_scope=data["timing_scope"],representation_scope=data["representation_scope"],source_policy=data["source_policy"],
        evidence=str((screen / "summary.json").relative_to(ROOT)),evidence_sha256=sha256_file(screen / "summary.json"),
        output_checks=checks,native_mixed_quality=quality,
        comparisons=dict(best_measured_unsloth_route=best,unsloth_over_native_gpu=medians[best]/medians["native-gpu"],
            unsloth_over_native_frozen=medians[best]/medians["native-frozen"],
            mixed_reduction_vs_unsloth_percent=100*(1-medians["native-frozen"]/medians[best]),
            mflux_over_native_frozen=medians["mflux-bf16"]/medians["native-frozen"]),
        trials=[compact_trial(trial) for trial in data["trials"]])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error("fresh evidence output required")
    result=audit(args.screen.resolve(strict=True))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,result)
    print(json.dumps(dict(status=result["status"],comparisons=result["comparisons"],output_checks=len(result["output_checks"]))))


if __name__=="__main__":main()
