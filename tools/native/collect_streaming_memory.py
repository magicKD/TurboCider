#!/usr/bin/env python3
"""Run a command under the process-tree sampler and emit a summary JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from process_tree_sampler import EXIT_INCONCLUSIVE, SamplerError, main as sampler_main
from verify_process_tree_samples import EvidenceError, verify


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", required=True, type=Path)
    value.add_argument("--summary", required=True, type=Path)
    value.add_argument("--correlation-id", required=True)
    value.add_argument("--interval-ms", type=int, default=20)
    value.add_argument("--max-gap-ms", type=int, default=100)
    value.add_argument("--root-role", default="turbocider-root")
    value.add_argument("--role", action="append", default=[], metavar="ROLE=GLOB")
    value.add_argument("command", nargs=argparse.REMAINDER)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise SystemExit("collect_streaming_memory requires a command after --")
    if args.output.exists():
        raise SystemExit(f"evidence output already exists: {args.output}")
    if args.summary.exists():
        raise SystemExit(f"summary output already exists: {args.summary}")
    sampler_args = [
        "--launch", "--output", str(args.output),
        "--correlation-id", args.correlation_id,
        "--interval-ms", str(args.interval_ms),
        "--max-gap-ms", str(args.max_gap_ms),
        "--root-role", args.root_role,
    ]
    for role in args.role:
        sampler_args.extend(["--role", role])
    sampler_args.extend(["--", *command])
    status = sampler_main(sampler_args)
    try:
        verified = verify(args.output)
    except EvidenceError as exc:
        summary = {
            "schema": "turbocider-streaming-memory-summary-v1",
            "correlation_id": args.correlation_id,
            "status": "invalid",
            "complete": False,
            "reasons": ["evidence_invalid"],
            "error": str(exc),
            "evidence_path": str(args.output),
        }
        with args.summary.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
            stream.write("\n")
        return 2
    if verified["correlation_id"] != args.correlation_id:
        raise SamplerError("verified evidence correlation id differs")
    summary = {
        "schema": "turbocider-streaming-memory-summary-v1",
        "sampler_revision": verified.get("sampler_revision"),
        "correlation_id": args.correlation_id,
        "status": verified["status"],
        "complete": verified["complete"],
        "reasons": verified["reasons"],
        "sample_count": verified["sample_count"],
        "max_gap_ns": verified["max_gap_ns"],
        "allowed_max_gap_ns": verified["allowed_max_gap_ns"],
        "tree_peak_rss_bytes": verified["tree_peak_rss_bytes"],
        "tree_peak_phys_footprint_bytes": verified[
            "tree_peak_phys_footprint_bytes"
        ],
        "swap_in_bytes": verified["swap_in_bytes"],
        "swap_out_bytes": verified["swap_out_bytes"],
        "compression_bytes": verified["compression_bytes"],
        "decompression_bytes": verified["decompression_bytes"],
        "command_exit_code": verified["command_exit_code"],
        "evidence_digest": verified["final_evidence_digest"],
        "evidence_path": str(args.output),
    }
    with args.summary.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    return status if status in (0, EXIT_INCONCLUSIVE) else status


if __name__ == "__main__":
    raise SystemExit(main())
