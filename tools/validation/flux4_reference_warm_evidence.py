#!/usr/bin/env python3
"""Audit reference-only FLUX4 PNGs; preserve collector-only failures verbatim."""
import argparse
import copy
import json
from pathlib import Path
import statistics

from flux4_reference_warm_screen import ROUTES, validate_trial
from flux9_bf16_framework_screen import write_json
from flux9_framework_evidence import portable
from runtime_ane_common import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def audit(screen):
    data = json.loads((screen / "summary.json").read_text())
    recovery = data["status"] == "failed" and data.get("error") == "actual four denoise completion events required"
    if not (data["status"] == "complete_reference_only" or recovery) or data["reference_only"] is not True or data["native_retested"] is not False:
        raise ValueError("complete references or known post-validation stage-name collector error required")
    trials, routes = [], []
    for index, route in enumerate(ROUTES):
        output = screen / f"{index}-{route}"
        payload_dir = output / "request-output" if route == "mflux-q4" else output
        report = json.loads((payload_dir / "report.json").read_text())
        if report["status"] != "complete_reference_only":
            raise ValueError("actual full requests must already have completed")
        report = copy.deepcopy(report)
        if route == "mlxserve-q4":
            for i, row in enumerate(report["runs"]):
                events = json.loads((output / f"events-{i}.json").read_text())
                row["denoise_progress"] = [[e["step"], e["total"]] for e in events if e.get("type") == "progress" and e.get("stage") == "Generating"]
                row["events_sha256"] = sha256_file(output / f"events-{i}.json")
        stored = next((t for t in data["trials"] if t["route"] == route), None)
        if stored:
            memory = stored["memory"]
        else:
            if not recovery or route != "mlxserve-q4":
                raise ValueError("missing independent reference memory")
            verified = json.loads((output / "run-memory-report.json").read_text())
            memory = dict(evidence="run-memory.jsonl", evidence_sha256=sha256_file(output / "run-memory.jsonl"),
                report="run-memory-report.json", report_sha256=sha256_file(output / "run-memory-report.json"), verified=verified,
                scope="post-validation recovery only; original sampler/verifier already complete and exit0; no weakened gap/role/data rule")
        if memory["verified"].get("command_exit_code") != 0 or memory["verified"].get("complete") is not True:
            raise ValueError("collector recovery cannot promote failed/inconclusive execution or memory")
        for k in ("evidence", "report"):
            if sha256_file(output / memory[k]) != memory[k + "_sha256"]:
                raise ValueError("independent memory chain differs")
        trial = dict(index=index, route=route, report=report, memory=memory)
        validate_trial(trial)
        from PIL import Image
        for i, row in enumerate(report["runs"]):
            path = payload_dir / f"{i}.png"
            with Image.open(path) as image:
                image.load()
                if image.size != (512,512):
                    raise ValueError("full-size original PNG required")
            if sha256_file(path) != row["png_sha256"]:
                raise ValueError("original PNG differs")
        trials.append(portable(trial))
        samples = [r["request_wall_including_export"] for r in report["runs"][1:3]]
        routes.append(dict(route=route, fresh_warm_seconds=samples, fresh_warm_median=statistics.median(samples),
            cold_seconds=[report["runs"][0]["request_wall_including_export"]], repeat_seconds=[report["runs"][3]["request_wall_including_export"]],
            constructor_seconds=report.get("constructor_seconds", report.get("ready_seconds")),
            physical_peak_bytes=memory["verified"]["tree_peak_phys_footprint_bytes"], swap_out_bytes=memory["verified"]["swap_out_bytes"]))
    native_path = ROOT / "docs/design/validation/flux4-native-saved-q4-precheck-20261012.json"
    native = json.loads(native_path.read_text())
    # Existing native precheck is intentionally not promoted to a paired run.
    return dict(schema="tc-flux4-warm-reference-validation-v1", status="complete_reference_only", date=data["date"],
        reference_only=True, native_retested=False, same_window_native_pair=False, qualification_passed=False, goal_complete=False,
        evidence=str((screen / "summary.json").relative_to(ROOT)), evidence_sha256=sha256_file(screen / "summary.json"),
        collector_record_status=data["status"], collector_stage_name_recovered=recovery,
        recovery_scope="only stage filter corrected from denoise to source-confirmed Generating; all original reports/PNG/events/memory/failed summary unchanged",
        model_sha256={portable(k):v for k,v in data["model_sha256"].items()}, routes=routes, trials=trials,
        existing_native_evidence=str(native_path.relative_to(ROOT)), existing_native_evidence_sha256=sha256_file(native_path),
        timing_scope=data["scope"], notes="reference twofresh each; mflux cache0/stock saved loader with full387/905/266 coverage; mlx-serve26.10.1 SSE fouractual Generating steps and completePNG; native previous Q4/g64 precheck only, no paired/global/ANE claim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh audit output required")
    result = audit(args.screen.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(dict(status=result["status"], recovered=result["collector_stage_name_recovered"],
        medians={r["route"]:r["fresh_warm_median"] for r in result["routes"]})))


if __name__ == "__main__":
    main()
