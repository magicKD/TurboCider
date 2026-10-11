#!/usr/bin/env python3
"""Audit every declared Z partition candidate, preserving failure receipts."""
import argparse
import json
from pathlib import Path

from runtime_ane_common import sha256_file
from z_mflux_framework_evidence import audit
from z_private_partition_sweep import ORDER, candidates
from z_mflux_framework_screen import PRIVATE_SHARES

ROOT = Path(__file__).resolve().parents[2]


def validate_grid(data):
    expected = candidates(PRIVATE_SHARES, ["f32", "f16"])
    if (data.get("status") not in ("complete_diagnostic", "complete_grid_with_retained_failures") or
            data.get("candidates") != expected or len(data.get("trials", [])) != len(expected)):
        raise ValueError("terminal complete fourteen-candidate grid required")
    for index, (trial, candidate) in enumerate(zip(data["trials"], expected)):
        if trial["index"] != index or any(trial[key] != value for key, value in candidate.items()):
            raise ValueError("actual complete candidate order differs")
        if trial["accepted"] is not (trial["exit_code"] == 0):
            raise ValueError("failed candidate was relabelled successful")
    return expected


def collect(grid, layout, build_manifest):
    data = json.loads((grid / "summary.json").read_text())
    validate_grid(data)
    result = dict(schema="tc-z-private-partition-grid-validation-v1", status=data["status"], date="2026-10-11",
        qualification_passed=False, goal_complete=False, evidence=str((grid / "summary.json").relative_to(ROOT)),
        evidence_sha256=sha256_file(grid / "summary.json"), candidates=data["candidates"], trials=[], comparisons=[])
    for trial in data["trials"]:
        path = ROOT / trial["evidence"]
        if not path.is_file() and not trial["accepted"] and trial["evidence_sha256"] is None:
            logs = {}
            for kind in ("stdout", "stderr"):
                log = grid / f'candidate-{trial["index"]}-{kind}.txt'
                if not log.is_file():
                    raise ValueError("failed startup raw logs missing")
                logs[kind + "_sha256"] = sha256_file(log)
            result["trials"].append({**trial, "status": "startup_failed_without_summary", "raw_log_bindings": logs,
                "failure_scope": "no successful model/timing proof; retained raw startup failure, not latency denominator"})
            continue
        if not path.is_file() or sha256_file(path) != trial["evidence_sha256"]:
            raise ValueError("candidate summary binding missing/changed")
        raw = json.loads(path.read_text())
        if not trial["accepted"]:
            result["trials"].append({**trial, "status": raw["status"], "failure_scope": "retained original failed full-request candidate; never used as latency denominator"})
            continue
        if raw["order"] != ORDER.split(",") or raw["private_channels"] != trial["channels"] or raw["private_join"] != trial["join"]:
            raise ValueError("actual complete task/partition differs")
        checked = audit(path.parent, layout, build_manifest)
        result["trials"].append(dict(index=trial["index"], channels=trial["channels"], join=trial["join"], accepted=True,
            evidence=trial["evidence"], evidence_sha256=trial["evidence_sha256"], audit=checked))
        routes = {row["route"]: row for row in checked["routes"]}
        gpu, frozen, private = (routes[name]["fresh_warm_median"] for name in ("native-gpu", "native-frozen", "native-private"))
        result["comparisons"].append(dict(index=trial["index"], channels=trial["channels"], join=trial["join"],
            gpu_seconds=gpu, frozen_seconds=frozen, private_seconds=private,
            gpu_over_private=gpu/private, frozen_over_private=frozen/private,
            private_reduction_vs_gpu_percent=100*(1-private/gpu),
            scope="all medians from this candidate's actual six-process cohort; no cross-candidate minimum-sample pooling"))
    if result["comparisons"]:
        result["best_tested_private"] = min(result["comparisons"], key=lambda row: row["private_seconds"])
    result["selection_scope"] = "lowest complete private candidate median among this finite grid; compare its own GPU/frozen controls; retain all failures and quality/memory; not all sizes/devices/global optimum or automatic production routing"
    if "/Users/" in json.dumps(result):
        raise ValueError("portable grid evidence contains machine-specific paths")
    return result


def compact(full, full_path):
    """Keep per-request counters/pressure; bind full-size image comparisons."""
    result = {key: value for key, value in full.items() if key != "trials"}
    result.update(full_audit=str(full_path.relative_to(ROOT)), full_audit_sha256=sha256_file(full_path), trials=[])
    for trial in full["trials"]:
        if not trial["accepted"]:
            result["trials"].append(trial)
            continue
        audit = trial["audit"]
        if "shared_source" not in result:
            result["shared_source"] = {key: value for key, value in audit.items() if key not in
                ("trials", "routes", "quality", "output_checks", "private_channels", "private_join", "order", "evidence", "evidence_sha256")}
        retained = {key: audit[key] for key in ("order", "routes", "private_channels", "private_join", "evidence", "evidence_sha256")}
        retained["trials"] = []
        for request_trial in audit["trials"]:
            item = {key: value for key, value in request_trial.items() if key != "report"}
            item["report"] = {key: value for key, value in request_trial["report"].items() if key != "runs"}
            item["report"]["runs"] = []
            for row in request_trial["report"]["runs"]:
                item["report"]["runs"].append(dict(index=row["index"], seconds=row["seconds"], actual_steps=row["actual_steps"],
                    prompt_cache_hit=row["prompt_cache_hit"], timings_seconds=row["timings_seconds"],
                    hybrid={key: row["hybrid"][key] for key in ("checkpoint_sha256_verified", "runtime_calls_session_total",
                        "runtime_failed", "runtime_failures_session_total") if key in row["hybrid"]},
                    runtime_weight={key: row["runtime_weight"][key] for key in ("executor_backend", "data_path", "partition_axis",
                        "ane_channels", "gpu_channels", "fp32_channel_join_enabled", "io_path", "channel_blocks_session_total",
                        "fallback_blocks_session_total", "overflow_retries_session_total", "headroom_scale") if key in row["runtime_weight"]}))
            retained["trials"].append(item)
        retained["quality"] = [row for row in audit["quality"] if row["request_index"] == 2]
        retained["output_check_summary"] = dict(total=len(audit["output_checks"]),
            forward_reverse_rgba_exact=sum(row["forward_reverse_rgba_exact"] for row in audit["output_checks"]))
        result["trials"].append({**trial, "audit": retained})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("grid", "layout", "build-manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--compact-output", type=Path, help="optional concise publication bound to all complete original-size image comparisons")
    args = parser.parse_args()
    if args.output.exists() or args.compact_output and args.compact_output.exists():
        parser.error("fresh audited grid destination required")
    result = collect(args.grid.resolve(), args.layout.resolve(), args.build_manifest.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    if args.compact_output:
        publication = compact(result, args.output.resolve())
        args.compact_output.parent.mkdir(parents=True, exist_ok=True)
        with args.compact_output.open("x") as output:
            json.dump(publication, output, indent=2, allow_nan=False)
            output.write("\n")
    print(json.dumps(dict(status=result["status"], best_tested_private=result.get("best_tested_private"))))


if __name__ == "__main__":
    main()
