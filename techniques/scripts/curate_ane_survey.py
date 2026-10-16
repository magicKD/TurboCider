#!/usr/bin/env python3
"""Read existing survey receipts and write a portable, non-attesting capsule.

Does not execute devices, reacquire sources or change old experiment outcomes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs"))
from ane.survey_evidence import compression, training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="ignored survey experiment root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("capsule already exists")
    capsule = {"schema": "ane-survey-evidence-v1", "recorded_at": "2026-10-16",
        "scope": "synthetic complete-output component observations; not hardware or model-quality qualification",
        "physical_ane_placement": "unverified", "production_qualified": False,
        "verifier_sha256": hashlib.sha256((ROOT / "labs/ane/survey_evidence.py").read_bytes()).hexdigest(),
        "compression_runs": [], "training_runs": [], "nonqualifying_runs": []}
    for folder in sorted(args.root.iterdir()):
        report_path = folder / "report.json"
        if not report_path.is_file():
            continue
        raw = report_path.read_bytes()
        report = json.loads(raw)
        if report.get("schema") not in ("ane-compression-survey-v1", "ane-training-probe-v1"):
            continue
        source = folder / "source.py"
        if hashlib.sha256(source.read_bytes()).hexdigest() != report["source_sha256"]:
            raise ValueError("per-run source snapshot mismatch")
        binding = {"run_id": folder.name, "raw_report_sha256": hashlib.sha256(raw).hexdigest(),
                   "source_sha256": report["source_sha256"]}
        if report.get("status") not in ("finished", "component-pass") or any(
                a.get("status") == "setup-failed" for a in report.get("arms", {}).values()):
            capsule["nonqualifying_runs"].append({**binding, "original_status": report["status"],
                "reason": "historical client error / incomplete protocol, no Core ML comparison qualification",
                "arm_statuses": {k: v["status"] for k, v in report.get("arms", {}).items()}})
            continue
        if report["schema"] == "ane-compression-survey-v1":
            capsule["compression_runs"].append({**binding,
                **{k: report[k] for k in ("shape", "layers", "seed", "rounds", "units", "environment",
                                          "weight_sha256", "input_sha256", "base_export_seconds", "gates_relative_l2")},
                "arms": compression(report), "samples": report["samples"]})
        else:
            capsule["training_runs"].append({**binding, "summary": training(report),
                "calls": report["calls"], "steps": report["steps"]})
    if len(capsule["compression_runs"]) < 4 or len(capsule["training_runs"]) < 3:
        raise ValueError("required independent/control protocols missing")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(capsule, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"compression_runs": len(capsule["compression_runs"]),
                      "training_runs": len(capsule["training_runs"]),
                      "nonqualifying_runs": len(capsule["nonqualifying_runs"])}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
