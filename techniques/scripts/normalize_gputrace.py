#!/usr/bin/env python3
"""Normalize an explicit gputrace timing JSON without launching a profiler.

This is an evidence adapter, not a timer. Unknown parser versions and clocks
remain unknown. Heuristic/synthetic values cannot become measured GPU time;
even profiler-reported durations do not establish physical timeline endpoints.
Only selected fields are exported; raw metadata, paths and addresses stay local.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

AUDITED_COMMIT = "3d7d32e66605ba994fb0765bf9f287682bd7ccc7"
HEURISTICS = {"synthetic", "extracted", "store0"}


def unsigned(value, label):
    if type(value) is not int or not 0 <= value < 2**64:
        raise ValueError(label + " must be a uint64, not a boolean or float")
    return value


def normalize(document, parser_commit):
    if not isinstance(document, dict) or not re.fullmatch(r"[0-9a-f]{40}", parser_commit):
        raise ValueError("timing object and explicit full parser commit required")
    source = document.get("timing_source", "unknown")
    approximate = document.get("timing_approximate")
    if not isinstance(source, str) or not source or (approximate is not None and type(approximate) is not bool):
        raise ValueError("invalid timing source/approximation metadata")
    rows = document.get("encoder_timings")
    if rows is None:
        rows = []
    if not isinstance(rows, list) or len(rows) > 100000:
        raise ValueError("bounded encoder timing list required")
    issues = []
    if parser_commit != AUDITED_COMMIT:
        issues.append("parser-commit-not-audited")
    if source in HEURISTICS and approximate is not True:
        issues.append("heuristic-source-without-true-approximation-marker")
    if source not in HEURISTICS | {"profiler"}:
        issues.append("missing-or-unknown-timing-source")
    elif source == "profiler":
        issues.append("profiler-duration-claim-not-independently-verified; endpoints-may-be-serialized")
    events = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("label"), str):
            raise ValueError("encoder row requires a string label")
        duration = unsigned(row.get("duration_ns"), "duration_ns")
        endpoints = {key: unsigned(row[key], key) if key in row else None
                     for key in ("start_timestamp", "end_timestamp")}
        if "duration_ms" in row:
            value = row["duration_ms"]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("duration_ms must be finite and nonnegative")
        # Keep shader symbols, but do not forward path-bearing/free-form metadata.
        label = row["label"]
        safe = bool(re.fullmatch(r"[A-Za-z0-9_ .,:<>()\[\]+*=-]{1,512}", label))
        if re.search(r"(?i)(?:[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}|sk-[a-z0-9_-]{12,}|gh[pousr]_[a-z0-9_]{20,})", label):
            safe = False
        source_kind = "predicted" if source in HEURISTICS and parser_commit == AUDITED_COMMIT else "unknown"
        events.append({
            "index": index, "label": label if safe else "<redacted-label>",
            "label_sha256": hashlib.sha256(label.encode("utf-8")).hexdigest(), "label_redacted": not safe,
            "reported_duration": {"value": duration if duration > 0 else None,
                                  "unit": "ns as declared by exporter; not clock validation",
                                  "source_kind": source_kind if duration > 0 else "unknown"},
            "reported_endpoints": endpoints,
            "endpoint_semantics": "unverified; never physical overlap evidence",
            "label_semantics": "parser timing label; not proof this kernel executed",
            "physical_gpu_duration_seconds": None,
        })
    count = document.get("total_encoders")
    if count is not None and unsigned(count, "total_encoders") != len(rows):
        issues.append("reported-encoder-count-does-not-match-rows")
    total = document.get("total_duration")
    if total is not None:
        total = unsigned(total, "total_duration")
    return {
        "schema": "tc-gputrace-timing-normalized-v1",
        "status": "observations-only" if any(row["reported_duration"]["value"] for row in events) else "no-usable-timing",
        "parser_commit": parser_commit, "parser_semantics_audited": parser_commit == AUDITED_COMMIT,
        "reported_source": source, "reported_approximate": approximate,
        "reported_total_duration": total, "reported_total_unit": "Go time.Duration ns; source may be heuristic",
        "physical_gpu_duration_seconds": None,
        "shape": None, "physical_dispatch_count": None,
        "row_count": {"value": len(events), "source_kind": "derived", "scope": "exporter rows, not dispatch count"},
        "events": events, "issues": issues,
        "production_qualified": False, "formal_performance_score": False,
        "raw_metadata_exported": False,
        "scope": "Selected timing fields only; no trace replay, viewer, source scanning, speedup or hardware qualification. Label filtering is not a general privacy audit.",
    }


def reject_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="explicit timing --json output, never a trace bundle")
    parser.add_argument("--parser-commit", required=True)
    parser.add_argument("--output", type=Path, help="new normalized JSON; otherwise stdout")
    args = parser.parse_args()
    try:
        if args.input.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("timing JSON exceeds 32 MiB adapter cap")
        raw = args.input.read_bytes()
        result = normalize(json.loads(raw, parse_constant=reject_constant), args.parser_commit)
        result["input_sha256"] = hashlib.sha256(raw).hexdigest()
        encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
        else:
            print(encoded, end="")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
