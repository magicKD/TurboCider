#!/usr/bin/env python3
"""Audit completed ComfyUI/native media, events and memory; no new inference."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from flux9_bf16_framework_screen import write_json
from flux9_framework_evidence import portable
from qwen21_mflux_framework_evidence import rgba, compare_rgba
from runtime_ane_common import sha256_file
from z_comfy_framework_screen import ORDER, ROUTES, validate_trial, workflow
from z_unsloth_framework_evidence import compact_trial

ROOT = Path(__file__).resolve().parents[2]


def event_spans(events):
    boundaries = [(e["data"]["node"], e["seconds"]) for e in events
                  if e["type"] == "executing" and e["data"].get("node")]
    end = next(e["seconds"] for e in events if e["type"] == "execution_success")
    return {node: (boundaries[i + 1][1] if i + 1 < len(boundaries) else end) - start
            for i, (node, start) in enumerate(boundaries)}


def validate_cohort(data):
    if data.get("status") != "complete_diagnostic" or data.get("count") != 4 or data.get("order") != list(ORDER):
        raise ValueError("complete four-route palindrome required")
    if len(data.get("trials", [])) != len(ORDER):
        raise ValueError("all eight independent processes required")
    for index, trial in enumerate(data["trials"]):
        if (trial["index"], trial["route"]) != (index, ORDER[index]):
            raise ValueError("actual trial order differs")
        validate_trial(trial)
    for route in ROUTES:
        trials = [t for t in data["trials"] if t["route"] == route]
        samples = [r["request_wall_including_export"] for t in trials for r in t["report"]["runs"][1:3]]
        median = data["medians"][route]
        if median["samples"] != samples or median["fresh_warm_seconds"] != statistics.median(samples):
            raise ValueError("median must use own actual four fresh samples")
    return data


def audit_comfy(output, report):
    log = (output / "server.stderr.txt").read_text() + (output / "server.stdout.txt").read_text()
    route = output.name.split("-", 1)[1]
    expected = "Using pytorch attention" if route == "comfy-sdpa-gpu" else "Using sub quadratic optimization for attention"
    device = "mps" if route == "comfy-sdpa-gpu" else "cpu"
    if expected not in log or f"CLIP/text encoder model load device: {device}" not in log or "Device: mps" not in log:
        raise ValueError("actual attention/encoder/MPS placement missing")
    if "Failed to initialize database" in log:
        raise ValueError("instance database must not conflict with user's server")
    stages = []
    for index, row in enumerate(report["runs"]):
        graph = json.loads((output / f"request-{index}.json").read_text())
        if graph != workflow(row["prompt"], index):
            raise ValueError("actual task workflow differs")
        history = json.loads((output / f"history-{index}.json").read_text())
        if history["prompt"][2] != graph or history["status"]["status_str"] != "success":
            raise ValueError("server's actual successful prompt differs")
        events = json.loads((output / f"events-{index}.json").read_text())
        if not events or any(e["data"].get("prompt_id") != row["prompt_id"] for e in events):
            raise ValueError("event chain must belong to this owned request")
        progress = [e["data"]["value"] for e in events if e["type"] == "progress" and e["data"].get("node") == "8"]
        if any(e["data"].get("max") != 8 for e in events if e["type"] == "progress" and e["data"].get("node") == "8"):
            raise ValueError("actual sampler step totals differ")
        executed = [e["data"]["node"] for e in events if e["type"] == "executing" and e["data"].get("node")]
        cached = [n for e in events if e["type"] == "execution_cached" for n in e["data"]["nodes"]]
        if (progress, executed, cached) != (row["actual_steps"], row["executed_nodes"], row["cached_nodes"]):
            raise ValueError("actual step/node/cache evidence differs")
        spans = event_spans(events)
        stages.append(dict(encoder_seconds=spans.get("4", 0), sampler_seconds=spans["8"],
                           vae_seconds=spans["9"], save_seconds=spans["10"]))
    return stages


def audit(screen):
    data = validate_cohort(json.loads((screen / "summary.json").read_text()))
    routes, checks, trials_compact = [], [], []
    for route in ROUTES:
        trials = [t for t in data["trials"] if t["route"] == route]
        all_stages = []
        for trial in trials:
            output = screen / f"{trial['index']}-{route}"
            memory = trial["memory"]
            if any(sha256_file(output / memory[key]) != memory[key + "_sha256"] for key in ("evidence", "report")):
                raise ValueError("independent memory chain differs")
            if route.startswith("comfy-"):
                stages = audit_comfy(output, trial["report"])
            else:
                stages = [r["metrics"]["timings_seconds"] for r in trial["report"]["runs"]]
            all_stages.extend(stages[1:3])
            for index, digest in enumerate(trial["png_sha256"]):
                path = output / f"{index}.png"
                if rgba(path).shape != (512, 512, 4) or sha256_file(path) != digest:
                    raise ValueError("complete original PNG evidence differs")
            compact = compact_trial(trial)
            compact["stage_seconds"] = stages
            if route.startswith("comfy-"):
                # argv is retained in raw history; distributed identity comes
                # from pinned commit, packages, launch hash and source digest.
                compact["report"]["system_stats"]["system"].pop("argv", None)
                compact["launch_sha256"] = sha256_file(output / "launch.json")
                compact["request_evidence_sha256"] = [{kind: sha256_file(output / f"{kind}-{i}.json")
                    for kind in ("events", "history", "request")} for i in range(4)]
                compact["server_log_sha256"] = {name: sha256_file(output / name) for name in ("server.stdout.txt", "server.stderr.txt")}
            trials_compact.append(compact)
        for index in range(4):
            paths = [screen / f"{t['index']}-{route}" / f"{index}.png" for t in trials]
            checks.append(dict(route=route, index=index, rgba_equal=bool(np.array_equal(rgba(paths[0]), rgba(paths[1])))))
        median = data["medians"][route]
        stages_median = {k: statistics.median(r[k] for r in all_stages) for k in all_stages[0]}
        routes.append(dict(route=route, fresh_warm_seconds=median["samples"], fresh_warm_median=median["fresh_warm_seconds"],
            cold_seconds=median["cold_seconds"], repeat_seconds=median["repeat_seconds"], constructor_seconds=median["constructor_seconds"],
            fresh_stage_medians=stages_median, physical_peaks=[t["memory"]["verified"]["tree_peak_phys_footprint_bytes"] for t in trials],
            compression_bytes=[t["memory"]["verified"]["compression_bytes"] for t in trials],
            swap_out_bytes=[t["memory"]["verified"]["swap_out_bytes"] for t in trials]))
    medians = {r["route"]: r["fresh_warm_median"] for r in routes}
    best = min(ROUTES[2:], key=medians.get)
    source_hashes = {portable(k) if k.startswith(str(ROOT)) else "comfy-source/" + k.split("/ComfyUI/ComfyUI/", 1)[-1]: v
                     for k, v in data["source_sha256"].items()}
    gpu = [t for t in data["trials"] if t["route"] == "native-gpu"]
    mixed = [t for t in data["trials"] if t["route"] == "native-frozen"]
    quality = [dict(direction=d, prompt_index=i, **compare_rgba(
        rgba(screen / f"{gpu[d]['index']}-native-gpu" / f"{i}.png"),
        rgba(screen / f"{mixed[d]['index']}-native-frozen" / f"{i}.png"))) for d in range(2) for i in range(4)]
    return dict(schema="tc-z-comfy-four-route-validation-v1", date=data["date"], status="complete_diagnostic", hardware="M4 Max64GiB/macOS26.6.2",
        qualification_passed=False, goal_complete=False, reference_latency_ranking_valid=True, steps=8, requests_per_process=4, order=list(ORDER),
        routes=routes, trials=sorted(trials_compact, key=lambda t: t["index"]), output_checks=checks, native_mixed_quality=quality,
        library_sha256=data["library_sha256"], native_inputs_matched=data["native_inputs_matched"], comfy_commit=data["comfy_commit"],
        comfy_local_changes=data["comfy_local_changes"], model_sha256={portable(k): v for k, v in data["model_sha256"].items()},
        source_snapshot_sha256=hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
        evidence=str((screen / "summary.json").relative_to(ROOT)), evidence_sha256=sha256_file(screen / "summary.json"),
        timing_scope=data["timing_scope"], representation_scope=data["representation_scope"], source_policy=data["source_policy"],
        stage_scope="Comfy host WebSocket node-boundary envelopes, not GPU-exclusive time or extra fences; native original stage spans",
        comparisons=dict(best_measured_comfy_route=best, comfy_over_native_gpu=medians[best] / medians["native-gpu"],
            comfy_over_native_frozen=medians[best] / medians["native-frozen"],
            mixed_reduction_vs_comfy_percent=100 * (1 - medians["native-frozen"] / medians[best])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh evidence output required")
    result = audit(args.screen.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(dict(status=result["status"], comparisons=result["comparisons"], output_checks=len(result["output_checks"]))))


if __name__ == "__main__":
    main()
