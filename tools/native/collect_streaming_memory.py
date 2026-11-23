#!/usr/bin/env python3
"""Run a command under the process-tree sampler and emit a summary JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from process_tree_sampler import SamplerError, main as sampler_main
from verify_process_tree_samples import (
    MEMORY_SUMMARY_SCHEMA,
    EvidenceError,
    memory_summary,
    verify,
)


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
            "schema": MEMORY_SUMMARY_SCHEMA,
            "correlation_id": args.correlation_id,
            "status": "invalid",
            "complete": False,
            "reasons": ["evidence_invalid"],
            "error": str(exc),
            "evidence_path": str(args.output),
        }
        status = 2
    else:
        if verified["correlation_id"] != args.correlation_id:
            raise SamplerError("verified evidence correlation id differs")
        summary = memory_summary(verified, args.output)
    with args.summary.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
