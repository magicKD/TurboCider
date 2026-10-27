"""Intersect synchronous Metal host API calls with observed callback windows.

Host API latency includes driver/cache work, not necessarily actual compilation.
This observer excludes asynchronous APIs, GPU execution and MLX graph tracing.
Nested/parallel calls use interval union, never an inflated sum of durations.
"""
import json
from pathlib import Path


def union_ns(intervals):
    total, previous_end = 0, None
    for start, end in sorted(intervals):
        if previous_end is None or start > previous_end:
            total += end - start
        else:
            total += max(0, end - previous_end)
        previous_end = max(previous_end or end, end)
    return total


def summarize_trace(path, start_ns, end_ns, events, expected_pid):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    setup = [row for row in rows if row["type"] == "setup"]
    if len(setup) != 1 or setup[0].get("schema_version") != 1:
        raise ValueError("Metal observer did not produce one supported setup record")
    setup = setup[0]
    if setup["pid"] != expected_pid or setup["clock"] != "mach_absolute_time_ns":
        raise ValueError("Metal trace belongs to a different process or clock")
    for key in ("library_source", "library_url", "library_data", "pipeline_function", "pipeline_options", "pipeline_descriptor"):
        if setup.get(key) is not True:
            raise ValueError("incomplete Metal observer setup: " + key)
    if any(row["type"] == "truncated" for row in rows):
        raise ValueError("Metal trace exceeded its bounded capacity")
    calls = [row for row in rows if row["type"] == "metal_api"]
    if not calls:
        raise ValueError("Metal observer captured no API calls; injection may have been lost across exec")
    if any(row["pid"] != expected_pid or row["end_ns"] < row["start_ns"] for row in calls):
        raise ValueError("invalid Metal API record")

    def window(start, end):
        relevant = [row for row in calls if row["start_ns"] < end and row["end_ns"] > start]
        def intervals(values):
            return [(max(row["start_ns"], start), min(row["end_ns"], end)) for row in values]
        build = [row for row in relevant if row["kind"] != "library_load"]
        return dict(wall_seconds=(end - start) / 1e9, call_count=len(relevant),
                    failed_calls=sum(not row["success"] for row in relevant),
                    all_api_host_union_seconds=union_ns(intervals(relevant)) / 1e9,
                    build_api_host_union_seconds=union_ns(intervals(build)) / 1e9,
                    by_kind={kind: dict(count=len(values := [r for r in relevant if r["kind"] == kind]),
                                       host_union_seconds=union_ns(intervals(values)) / 1e9)
                             for kind in sorted({row["kind"] for row in relevant})},
                    longest_calls=[dict(kind=row["kind"], name=row["name"],
                                        seconds=(min(row["end_ns"], end) - max(row["start_ns"], start)) / 1e9)
                                   for row in sorted(relevant, key=lambda r: r["end_ns"] - r["start_ns"], reverse=True)[:5]])

    boundaries = {}
    for event in events:
        if event["phase"] == "denoise":
            boundaries.setdefault(event["completed"], event["observed_monotonic_ns"])
    return dict(setup=setup, request=window(start_ns, end_ns),
                steps=[dict(step=step, **window(boundaries[step - 1], boundaries[step]))
                       for step in sorted(boundaries) if step > 0 and step - 1 in boundaries])
