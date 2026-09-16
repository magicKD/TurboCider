#!/usr/bin/env python3
"""Evaluate already-generated paired sparse videos without rerunning inference."""
import argparse
import json
from pathlib import Path
from video_quality_gate import compare, passes
from benchmark_ltx_resident import bf16_metrics


def tensor_metrics(reference, candidate, audio):
    names = ["stage1_video.bf16", "stage2_input_video.bf16", "stage2_video.bf16"]
    if audio:
        names += ["stage1_audio.bf16", "stage2_audio.bf16"]
    metrics = {name: bf16_metrics(reference / name, candidate / name) for name in names}
    unchanged = ["stage1_video.bf16", "stage2_input_video.bf16"]
    if audio:
        unchanged.append("stage1_audio.bf16")
    if not all(metrics[name]["byte_exact"] for name in unchanged):
        raise RuntimeError("Stage-2 isolation failed: Stage-1 or Stage-2 video input differs")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    source = json.loads(args.report.read_text())
    if not source.get("complete"):
        parser.error("source benchmark is incomplete; wait for paired measurements and isolation checks")
    baseline = next(r for r in source["runs"] if r["variant"] == "baseline")
    reference = Path(baseline["request"]["output"])
    output = args.report.with_name("decoded-quality.json")
    if output.exists():
        parser.error("decoded-quality.json already exists; preserve existing measurements")
    report = {"source_report": str(args.report.resolve()), "complete": False, "comparisons": []}
    for row in source["runs"]:
        if row["variant"] != "candidate":
            continue
        candidate = Path(row["request"]["output"])
        tensors = tensor_metrics(Path(baseline["request"]["dump_tensors"]),
                                 Path(row["request"]["dump_tensors"]),
                                 source["workload"].get("audio", False))
        metrics = compare(reference, candidate)
        report["comparisons"].append(dict(reference=str(reference), candidate=str(candidate),
                                          metrics=metrics, tensor_metrics=tensors,
                                          rgb_gate_passed=passes(metrics)))
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in metrics.items() if k != "frame_metrics"}), flush=True)
    report["complete"] = True
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
