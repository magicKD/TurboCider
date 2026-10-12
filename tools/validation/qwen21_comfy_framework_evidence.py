#!/usr/bin/env python3
"""Audit complete Qwen2.1 ComfyUI/native images, schedule, events and memory."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from flux9_bf16_framework_screen import write_json
from flux9_framework_evidence import portable
from qwen21_comfy_framework_screen import ORDER, ROUTES, STEPS, sigmas, validate_trial, workflow
from qwen21_mflux_framework_evidence import rgba, compare_rgba
from runtime_ane_common import sha256_file
from z_comfy_framework_evidence import event_spans
from z_unsloth_framework_evidence import compact_trial

ROOT = Path(__file__).resolve().parents[2]


def validate_cohort(data):
    if (data.get("status") != "complete_diagnostic" or data.get("count") != 4 or data.get("order") != list(ORDER) or
            data.get("steps") != STEPS or len(data.get("trials", [])) != len(ORDER)):
        raise ValueError("complete Qwen five-route/ten-process40step palindrome required")
    if data["schedule"] != dict(sigmas=sigmas(), rule="40Euler,mu(.5-.9 at256-8192),terminal.02 then0", cfg=1, seed=29, repeat_seed=30):
        raise ValueError("actual declared base Euler/terminal schedule differs")
    for index, trial in enumerate(data["trials"]):
        if (trial["index"], trial["route"]) != (index, ORDER[index]):
            raise ValueError("actual full route order differs")
        validate_trial(trial)
    for route in ROUTES:
        trials = [t for t in data["trials"] if t["route"] == route]
        samples = [r["request_wall_including_export"] for t in trials for r in t["report"]["runs"][1:3]]
        m = data["medians"][route]
        if m["samples"] != samples or m["fresh_warm_seconds"] != statistics.median(samples):
            raise ValueError("own actual four fresh samples must determine the median")
    return data


def validate_reference_cohort(data):
    if (data.get("schema") != "tc-qwen21-comfy-reference-screen-v1" or data.get("reference_only") is not True or
            data.get("status") != "complete_diagnostic" or data.get("count") != 4 or data.get("steps") != STEPS or
            data.get("order") != list(ROUTES[2:]) or len(data.get("trials", [])) != 3):
        raise ValueError("complete three-preset reference-only Qwen cohort required")
    for index, trial in enumerate(data["trials"]):
        if (trial["index"], trial["route"]) != (index, ROUTES[2:][index]):
            raise ValueError("actual reference-only order differs")
        validate_trial(trial)
        samples = [r["request_wall_including_export"] for r in trial["report"]["runs"][1:3]]
        m = data["medians"][trial["route"]]
        if m["samples"] != samples or m["fresh_warm_seconds"] != statistics.median(samples):
            raise ValueError("own actual two fresh samples required")
    return data


def existing_native_best(model_hashes):
    configs_path = ROOT / "docs/benchmark/best-local-configs-20261011.json"
    configs = json.loads(configs_path.read_text())
    selected = [r for r in configs["configurations"] if r["model"] == "qwen-image-2.1"]
    if len(selected) != 1:
        raise ValueError("one preserved Qwen source-bound best configuration required")
    cfg = selected[0]
    evidence_path = ROOT / cfg["evidence"]
    evidence = json.loads(evidence_path.read_text())
    if evidence["status"] != "complete_diagnostic" or evidence["reference_latency_ranking_valid"] is not True:
        raise ValueError("existing baseline must have valid complete original evidence")
    for name, digest in model_hashes.items():
        if evidence["model_sha256"].get(portable(name)) != digest:
            raise ValueError("existing native best must use the same original BF16 components")
    rows = {r["route"]: r for r in evidence["routes"]}
    for route, field in (("native-frozen", "fresh_warm_median_seconds"), ("native-gpu", "gpu_control_seconds")):
        row = rows[route]
        if len(row["fresh_warm_seconds"]) != 4 or statistics.median(row["fresh_warm_seconds"]) != cfg[field]:
            raise ValueError("saved native best must derive from its own original four actual samples")
    return dict(evidence=cfg["evidence"], evidence_sha256=sha256_file(evidence_path),
        configuration_evidence="docs/benchmark/best-local-configs-20261011.json",
        source_library=cfg["library"], source_library_sha256=evidence["library_sha256"],
        frozen_manifest=cfg["compiled_manifest"], native_gpu_seconds=cfg["gpu_control_seconds"],
        native_frozen_seconds=cfg["fresh_warm_median_seconds"], native_samples={r: rows[r]["fresh_warm_seconds"] for r in ("native-gpu", "native-frozen")},
        scope="existing preserved native best, not retested or a same-window paired comparison; same original BF16 components/task; original build/cache/precision lifecycle retained")


def audit_comfy(output, report):
    route = output.name.split("-", 1)[1]
    log = (output / "server.stderr.txt").read_text() + (output / "server.stdout.txt").read_text()
    attention = "Using pytorch attention" if route == "comfy-sdpa-gpu" else "Using sub quadratic optimization for attention"
    device = "cpu" if route == "comfy-default" else "mps"
    if (attention not in log or f"CLIP/text encoder model load device: {device}" not in log or "Device: mps" not in log or
            "Requested to load QwenImage21" not in log or "model weight dtype torch.bfloat16" not in log):
        raise ValueError("actual Qwen MPS model/attention/text placement missing")
    if "Failed to initialize database" in log:
        raise ValueError("owned database must not conflict with user service")
    stages, step_envelopes = [], []
    for index, row in enumerate(report["runs"]):
        graph = json.loads((output / f"request-{index}.json").read_text())
        if graph != workflow(row["prompt"], index):
            raise ValueError("actual full Qwen workflow/task differs")
        history = json.loads((output / f"history-{index}.json").read_text())
        if history["prompt"][2] != graph or history["status"]["status_str"] != "success":
            raise ValueError("server's actual successful graph differs")
        events = json.loads((output / f"events-{index}.json").read_text())
        if not events or any(e["data"].get("prompt_id") != row["prompt_id"] for e in events):
            raise ValueError("event chain belongs to another request")
        steps = [e for e in events if e["type"] == "progress" and e["data"].get("node") == "8"]
        if any(e["data"].get("max") != STEPS for e in steps) or [e["data"]["value"] for e in steps] != row["actual_steps"]:
            raise ValueError("actual all40 sampler steps/total differ")
        executed = [e["data"]["node"] for e in events if e["type"] == "executing" and e["data"].get("node")]
        cached = [n for e in events if e["type"] == "execution_cached" for n in e["data"]["nodes"]]
        if (executed, cached) != (row["executed_nodes"], row["cached_nodes"]):
            raise ValueError("actual node/cache evidence differs")
        spans = event_spans(events)
        stages.append(dict(encoder_seconds=spans.get("4", 0), sampler_seconds=spans["8"],
            vae_seconds=spans["9"], save_seconds=spans["10"]))
        start = next(e["seconds"] for e in events if e["type"] == "executing" and e["data"].get("node") == "8")
        intervals = [steps[i]["seconds"] - steps[i - 1]["seconds"] for i in range(1, len(steps))]
        step_envelopes.append(dict(first_progress_after_sampler_start=steps[0]["seconds"] - start,
            subsequent_progress_intervals=intervals, subsequent_progress_interval_median=statistics.median(intervals)))
    return stages, step_envelopes


def audit(screen):
    raw = json.loads((screen / "summary.json").read_text())
    reference_only = raw.get("reference_only") is True
    data = validate_reference_cohort(raw) if reference_only else validate_cohort(raw)
    selected_routes = list(dict.fromkeys(data["order"]))
    routes, checks, compact = [], [], []
    for route in selected_routes:
        trials = [t for t in data["trials"] if t["route"] == route]
        stage_samples = []
        for trial in trials:
            output = screen / f"{trial['index']}-{route}"
            memory = trial["memory"]
            if any(sha256_file(output / memory[k]) != memory[k + "_sha256"] for k in ("evidence", "report")):
                raise ValueError("independent memory chain differs")
            if route.startswith("comfy-"):
                stages, steps = audit_comfy(output, trial["report"])
            else:
                stages, steps = [r["metrics"]["timings_seconds"] for r in trial["report"]["runs"]], []
            stage_samples.extend(stages[1:3])
            for index, digest in enumerate(trial["png_sha256"]):
                path = output / f"{index}.png"
                if rgba(path).shape != (512, 512, 4) or sha256_file(path) != digest:
                    raise ValueError("complete unresized original PNG differs")
            item = compact_trial(trial)
            item["stage_seconds"] = stages
            if route.startswith("comfy-"):
                item["report"]["system_stats"]["system"].pop("argv", None)
                item["step_progress_envelopes"] = steps
                item["launch_sha256"] = sha256_file(output / "launch.json")
                item["request_evidence_sha256"] = [{kind: sha256_file(output / f"{kind}-{i}.json")
                    for kind in ("request", "events", "history")} for i in range(4)]
                item["server_log_sha256"] = {n: sha256_file(output / n) for n in ("server.stdout.txt", "server.stderr.txt")}
            compact.append(item)
        if len(trials) == 2:
            for index in range(4):
                paths = [screen / f"{t['index']}-{route}" / f"{index}.png" for t in trials]
                checks.append(dict(route=route, index=index, rgba_equal=bool(np.array_equal(rgba(paths[0]), rgba(paths[1])))))
        m = data["medians"][route]
        routes.append(dict(route=route, fresh_warm_seconds=m["samples"], fresh_warm_median=m["fresh_warm_seconds"],
            cold_seconds=m["cold_seconds"], repeat_seconds=m["repeat_seconds"], constructor_seconds=m["constructor_seconds"],
            fresh_stage_medians={k: statistics.median(s[k] for s in stage_samples) for k in stage_samples[0]},
            physical_peaks=[t["memory"]["verified"]["tree_peak_phys_footprint_bytes"] for t in trials],
            compression_bytes=[t["memory"]["verified"]["compression_bytes"] for t in trials],
            swap_out_bytes=[t["memory"]["verified"]["swap_out_bytes"] for t in trials]))
    medians = {r["route"]: r["fresh_warm_median"] for r in routes}
    best = min(ROUTES[2:], key=medians.get)
    gpu = [t for t in data["trials"] if t["route"] == "native-gpu"]
    mixed = [t for t in data["trials"] if t["route"] == "native-frozen"]
    quality = [] if reference_only else [dict(direction=d, prompt_index=i, **compare_rgba(
        rgba(screen / f"{gpu[d]['index']}-native-gpu" / f"{i}.png"),
        rgba(screen / f"{mixed[d]['index']}-native-frozen" / f"{i}.png"))) for d in range(2) for i in range(4)]
    baseline = existing_native_best(data["model_sha256"]) if reference_only else {}
    gpu_seconds = baseline["native_gpu_seconds"] if reference_only else medians["native-gpu"]
    mixed_seconds = baseline["native_frozen_seconds"] if reference_only else medians["native-frozen"]
    return dict(schema="tc-qwen21-comfy-reference-validation-v1" if reference_only else "tc-qwen21-comfy-five-route-validation-v1",
        reference_only=reference_only, existing_native_baseline=baseline, same_window_native_pair=not reference_only,
        date=data["date"], status="complete_diagnostic",
        hardware="M4 Max64GiB/macOS26.6.2", qualification_passed=False, goal_complete=False, reference_latency_ranking_valid=True,
        steps=STEPS, requests_per_process=4, order=data["order"], schedule=data["schedule"], routes=routes,
        trials=sorted(compact, key=lambda t: t["index"]), output_checks=checks, native_mixed_quality=quality,
        library_sha256=data["library_sha256"], native_inputs_matched=data["native_inputs_matched"], comfy_commit=data["comfy_commit"],
        comfy_local_changes=data["comfy_local_changes"], model_sha256={portable(k): v for k, v in data["model_sha256"].items()},
        source_snapshot_sha256=hashlib.sha256(json.dumps({portable(k): v for k, v in data["source_sha256"].items()}, sort_keys=True).encode()).hexdigest(),
        evidence=str((screen / "summary.json").relative_to(ROOT)), evidence_sha256=sha256_file(screen / "summary.json"),
        timing_scope=data["timing_scope"], representation_scope=data["representation_scope"], source_policy=data["source_policy"],
        stage_scope="Comfy WebSocket host node/step envelopes include preparation/eviction/load, not GPU-exclusive traces; native original stage spans",
        comparisons=dict(best_measured_comfy_route=best, comfy_over_native_gpu=medians[best] / gpu_seconds,
            comfy_over_native_frozen=medians[best] / mixed_seconds,
            mixed_reduction_vs_comfy_percent=100 * (1 - mixed_seconds / medians[best])))


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
