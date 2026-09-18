#!/usr/bin/env python3
"""Independently verify process-tree sampler JSONL evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from process_tree_sampler import EXIT_INCONCLUSIVE, SCHEMA, canonical_json


class EvidenceError(ValueError):
    pass


def read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise EvidenceError(f"cannot read evidence {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceError(
                f"invalid JSONL at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise EvidenceError(f"record {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise EvidenceError("evidence is empty")
    return rows


def unsigned(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceError(f"{label} must be an unsigned integer")
    return value


def identity_key(value: Any, label: str) -> tuple[int, int, int]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} identity is missing")
    return (
        unsigned(value.get("pid"), f"{label}.pid"),
        unsigned(value.get("start_seconds"), f"{label}.start_seconds"),
        unsigned(value.get("start_microseconds"), f"{label}.start_microseconds"),
    )


def verify(path: Path) -> dict[str, Any]:
    rows = read_rows(path)
    previous = "0" * 64
    for index, row in enumerate(rows):
        if row.get("schema") != SCHEMA:
            raise EvidenceError(f"record {index} has the wrong schema")
        if row.get("sequence") != index:
            raise EvidenceError(f"record {index} sequence is not contiguous")
        if row.get("previous_digest") != previous:
            raise EvidenceError(f"record {index} hash-chain parent differs")
        digest = row.get("record_digest")
        if not isinstance(digest, str) or len(digest) != 64:
            raise EvidenceError(f"record {index} digest is invalid")
        payload = dict(row)
        payload.pop("record_digest")
        expected = hashlib.sha256(canonical_json(payload)).hexdigest()
        if digest != expected:
            raise EvidenceError(f"record {index} digest mismatch")
        previous = digest

    if rows[0].get("type") != "sampler_start":
        raise EvidenceError("first record is not sampler_start")
    start = rows[0]
    revision = start.get("revision")
    if not isinstance(revision, str) or not revision:
        raise EvidenceError("start sampler revision is invalid")
    if not isinstance(start.get("capabilities"), dict):
        raise EvidenceError("start capabilities are missing")
    root_identity = identity_key(start.get("root_identity"), "start.root")
    allowed_types = {
        "sampler_start", "process_start", "process_exit", "sample",
        "terminal",
    }
    for index, row in enumerate(rows):
        if row.get("type") not in allowed_types:
            raise EvidenceError(f"record {index} has an unsupported type")
        if index and row.get("type") == "sampler_start":
            raise EvidenceError("sampler_start appears more than once")
        if index != len(rows) - 1 and row.get("type") == "terminal":
            raise EvidenceError("terminal appears before the final record")
    terminal = rows[-1]
    if terminal.get("type") != "terminal" or terminal.get("terminal_sample") is not True:
        raise EvidenceError("last record is not a terminal sample")
    allowed_gap = unsigned(rows[0].get("max_gap_ns"), "start.max_gap_ns")
    correlation = rows[0].get("correlation_id")
    if not isinstance(correlation, str) or not correlation:
        raise EvidenceError("start correlation id is invalid")

    samples = [row for row in rows if row.get("type") == "sample"]
    terminal_sample_count = unsigned(
        terminal.get("sample_count"), "terminal.sample_count"
    )
    if len(samples) != terminal_sample_count:
        raise EvidenceError("terminal sample count differs")
    max_gap = 0
    peak_rss = 0
    peak_phys = 0
    peak_rss_mono = 0
    peak_phys_mono = 0
    previous_mono: int | None = None
    previous_system: dict[str, int] | None = None
    for sample_index, sample in enumerate(samples):
        if sample.get("sample_index") != sample_index:
            raise EvidenceError("sample indexes are not contiguous")
        if sample.get("correlation_id") != correlation:
            raise EvidenceError("sample correlation id differs")
        mono = unsigned(sample.get("mono_ns"), "sample.mono_ns")
        gap = unsigned(sample.get("gap_ns"), "sample.gap_ns")
        expected_gap = 0 if previous_mono is None else mono - previous_mono
        if expected_gap < 0 or gap != expected_gap:
            raise EvidenceError("sample gap does not match monotonic timestamps")
        if sample.get("gap_exceeded") != (gap > allowed_gap):
            raise EvidenceError("sample gap flag differs")
        max_gap = max(max_gap, gap)
        previous_mono = mono
        processes = sample.get("processes")
        if not isinstance(processes, list):
            raise EvidenceError("sample process list is missing")
        identities: set[tuple[int, int, int]] = set()
        rss = 0
        phys = 0
        for process_index, process in enumerate(processes):
            if not isinstance(process, dict):
                raise EvidenceError("sample process is not an object")
            identity = identity_key(
                process.get("identity"), f"sample.process[{process_index}]"
            )
            if identity in identities:
                raise EvidenceError("sample contains a duplicate process identity")
            identities.add(identity)
            rss += unsigned(process.get("resident_bytes"), "process.resident_bytes")
            phys += unsigned(
                process.get("phys_footprint_bytes"),
                "process.phys_footprint_bytes",
            )
        root_alive = sample.get("root_alive")
        if not isinstance(root_alive, bool):
            raise EvidenceError("sample root_alive is invalid")
        if root_alive != (root_identity in identities):
            raise EvidenceError("sample root_alive differs from process list")
        if sample.get("tree_rss_bytes") != rss:
            raise EvidenceError("tree RSS total differs")
        if sample.get("tree_phys_footprint_bytes") != phys:
            raise EvidenceError("tree physical-footprint total differs")
        if rss > peak_rss:
            peak_rss, peak_rss_mono = rss, mono
        if phys > peak_phys:
            peak_phys, peak_phys_mono = phys, mono
        system = sample.get("system")
        if not isinstance(system, dict):
            raise EvidenceError("sample system counters are missing")
        current_system = {
            key: unsigned(system.get(key), f"system.{key}")
            for key in (
                "pagein_bytes", "pageout_bytes", "compression_bytes",
                "decompression_bytes", "swap_in_bytes", "swap_out_bytes",
            )
        }
        if previous_system is not None and any(
            current_system[key] < previous_system[key]
            for key in current_system
        ):
            raise EvidenceError("system delta counters moved backwards")
        previous_system = current_system

    if terminal.get("correlation_id") != correlation:
        raise EvidenceError("terminal correlation id differs")
    if terminal.get("revision") != revision:
        raise EvidenceError("terminal sampler revision differs")
    if terminal.get("allowed_max_gap_ns") != allowed_gap:
        raise EvidenceError("terminal maximum-gap policy differs")
    expected = {
        "max_gap_ns": max_gap,
        "tree_peak_rss_bytes": peak_rss,
        "tree_peak_phys_footprint_bytes": peak_phys,
        "tree_peak_rss_mono_ns": peak_rss_mono,
        "tree_peak_phys_footprint_mono_ns": peak_phys_mono,
    }
    for key, value in expected.items():
        if terminal.get(key) != value:
            raise EvidenceError(f"terminal {key} differs")
    if samples:
        final_system = samples[-1]["system"]
        for key in (
            "swap_in_bytes", "swap_out_bytes", "compression_bytes",
            "decompression_bytes",
        ):
            if terminal.get(key) != final_system.get(key):
                raise EvidenceError(f"terminal {key} differs from last sample")

    reasons = terminal.get("reasons")
    if not isinstance(reasons, list) or any(
        not isinstance(reason, str) or not reason for reason in reasons
    ):
        raise EvidenceError("terminal reasons are invalid")
    expected_complete = not reasons
    if terminal.get("complete") is not expected_complete:
        raise EvidenceError("terminal complete flag differs")
    if terminal.get("status") != ("complete" if expected_complete else "inconclusive"):
        raise EvidenceError("terminal status differs")
    if max_gap > allowed_gap and "sample_gap_exceeded" not in reasons:
        raise EvidenceError("terminal omitted sample-gap failure")
    unknown = terminal.get("unknown_children")
    if not isinstance(unknown, list):
        raise EvidenceError("terminal unknown-child list is invalid")
    if unknown and "unknown_child" not in reasons:
        raise EvidenceError("terminal omitted unknown-child failure")
    for index, value in enumerate(unknown):
        identity_key(value, f"terminal.unknown_children[{index}]")
    expected_root_alive = samples[-1]["root_alive"] if samples else False
    if revision == "tc-process-tree-sampler-darwin-v1":
        stop_reason = "root_exit"
        root_alive_at_stop = expected_root_alive
    else:
        stop_reason = terminal.get("stop_reason")
        if stop_reason not in ("root_exit", "external_stop", "sampler_error"):
            raise EvidenceError("terminal stop reason is invalid")
        root_alive_at_stop = terminal.get("root_alive_at_stop")
    if root_alive_at_stop is not expected_root_alive:
        raise EvidenceError("terminal root-alive state differs")
    if stop_reason == "root_exit" and expected_root_alive:
        raise EvidenceError("root-exit terminal still reports a live root")
    if stop_reason == "sampler_error" and "sampler_error" not in reasons:
        raise EvidenceError("sampler-error terminal omitted sampler failure")
    command_exit = terminal.get("command_exit_code")
    if (
        command_exit is not None and
        (isinstance(command_exit, bool) or not isinstance(command_exit, int))
    ):
        raise EvidenceError("terminal command exit code is invalid")
    command_failed = command_exit not in (None, 0)
    if command_failed != ("command_failed" in reasons):
        raise EvidenceError("terminal command-failure reason differs")

    return {
        "schema": "turbocider-process-tree-verification-v1",
        "status": terminal["status"],
        "complete": terminal["complete"],
        "correlation_id": correlation,
        "sampler_revision": revision,
        "record_count": len(rows),
        "sample_count": len(samples),
        "final_evidence_digest": terminal["record_digest"],
        "max_gap_ns": max_gap,
        "allowed_max_gap_ns": allowed_gap,
        "tree_peak_rss_bytes": peak_rss,
        "tree_peak_phys_footprint_bytes": peak_phys,
        "swap_in_bytes": terminal["swap_in_bytes"],
        "swap_out_bytes": terminal["swap_out_bytes"],
        "compression_bytes": terminal["compression_bytes"],
        "decompression_bytes": terminal["decompression_bytes"],
        "reasons": reasons,
        "command_exit_code": command_exit,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify(args.evidence)
    except EvidenceError as exc:
        print(f"process-tree evidence verification failed: {exc}", file=sys.stderr)
        return 2
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")
    return 0 if result["complete"] else EXIT_INCONCLUSIVE


if __name__ == "__main__":
    raise SystemExit(main())
