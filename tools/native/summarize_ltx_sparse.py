#!/usr/bin/env python3
"""Summarize completed paired experiments, never treating missing quality as pass."""
import argparse
import json
import math
import statistics
from pathlib import Path


def summarize(report, decoded=None):
    if report.get("schema") != "ltx-sparse-stage2-paired-v1" or not report.get("complete"):
        return None
    groups = {name: [r for r in report["runs"] if r["variant"] == name]
              for name in ("baseline", "candidate")}
    if not all(groups.values()):
        raise ValueError("completed report lacks measured baseline/candidate")
    def durations(field, required):
        values = {name: [r["result"]["timings_seconds"].get(field) for r in rows]
                  for name, rows in groups.items()}
        if any(value is None for rows in values.values() for value in rows):
            if required:
                raise ValueError(f"completed report lacks {field} duration")
            return None
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
               not math.isfinite(value) or value <= 0
               for rows in values.values() for value in rows):
            raise ValueError(f"invalid {field} duration")
        return {name: statistics.median(rows) for name, rows in values.items()}
    times = durations("stage2", required=True)
    wall_times = durations("request_wall", required=False)
    quality = report.get("quality", [])
    isolated = len(quality) == len(groups["candidate"]) and all(
        q.get(name, {}).get("byte_exact") is True for q in quality
        for name in ("stage1_video.bf16", "stage2_input_video.bf16"))
    gates = [q.get("rgb_gate_passed") for q in quality]
    if decoded is not None and decoded.get("complete"):
        gates = [q.get("rgb_gate_passed") for q in decoded.get("comparisons", [])]
    status = "unverified"
    if any(g is False for g in gates):
        status = "failed"
    elif len(gates) == len(groups["candidate"]) and all(g is True for g in gates):
        status = "passed"
    return dict(workload=report["workload"], candidate=report["candidate"],
                method=report["method"], measured_order=[r["variant"] for r in report["runs"]],
                native_profile_enabled=report.get("native_profile_enabled"),
                stage2_seconds=times, stage2_speedup=times["baseline"] / times["candidate"],
                request_wall_seconds=wall_times,
                request_wall_speedup=(wall_times["baseline"] / wall_times["candidate"]
                                      if wall_times is not None else None),
                stage2_input_isolated=isolated, rgb_quality=status,
                library_sha256=report["library_sha256"], shader_sha256=report["shader_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, omitted = [], []
    for path in sorted(args.directory.rglob("report.json")):
        report = json.loads(path.read_text())
        decoded_path = path.with_name("decoded-quality.json")
        decoded = json.loads(decoded_path.read_text()) if decoded_path.exists() else None
        if decoded is not None and Path(decoded["source_report"]).resolve() != path.resolve():
            raise ValueError(f"quality report points to a different experiment: {decoded_path}")
        row = summarize(report, decoded)
        if row is None:
            omitted.append(str(path.resolve()))
        else:
            row["source_report"] = str(path.resolve())
            rows.append(row)
    with args.output.open("x") as output:
        json.dump(dict(schema="ltx-sparse-evidence-summary-v1", rows=rows,
                       omitted_incomplete_or_unsupported=omitted), output, indent=2)
        output.write("\n")
    print(f"Summarized {len(rows)} completed experiments; omitted {len(omitted)}")


if __name__ == "__main__":
    main()
