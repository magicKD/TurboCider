#!/usr/bin/env python3
"""Serial unchanged-source full encoder import with bounded 1/4/8 MiB buffers."""
import argparse
import json
import math
from pathlib import Path
import statistics

from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_sampled

ROOT = Path(__file__).resolve().parents[2]
ORDER = (1, 4, 8, 8, 4, 1)


def validate_record(record, buffer_mib, source_hash, read_ahead=False):
    if (record.get("schema") != "tc-qwen21-encoder-import-screen-v1" or
        record.get("source_sha256") != source_hash or record.get("embedding_mode") != "packed" or
        record.get("qualification_passed") is not False or record.get("verification_overlap") is not False or
        record.get("source_verified_before_publication") is not True):
        raise ValueError("actual registered packed import recipe differs")
    for field, value in (("workers", 8), ("verification_bytes", 0), ("registration_cache_hits", 1),
                         ("read_buffer_capacity_bytes", (buffer_mib << 20) * (2 if read_ahead else 1))):
        if type(record.get(field)) is not int or record[field] != value:
            raise ValueError("actual source/read buffer counter differs: " + field)
    for field in ("source_read_bytes", "packed_capacity_bytes", "managed_peak_bytes", "final_weight_logical_bytes",
                  "mlx_logical_peak_bytes"):
        if type(record.get(field)) is not int or record[field] <= 0:
            raise ValueError("missing actual bounded model import byte counter")
    if record["managed_peak_bytes"] > record["packed_capacity_bytes"] + (buffer_mib << 20) * (2 if read_ahead else 1):
        raise ValueError("read buffer or packed claims exceed bounded plan")
    for field in ("native_verification_seconds", "bank_load_seconds", "bank_read_seconds", "bank_decode_seconds",
                  "embedding_setup_seconds", "token_gather_seconds", "total_observed_seconds"):
        value = record.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid actual import/read/decode timing")
    if "read_ahead_enabled" in record or read_ahead:
        if (record.get("read_ahead_enabled") is not read_ahead or record.get("read_ahead_drained") is not read_ahead or
            type(record.get("read_buffer_slots")) is not int or record["read_buffer_slots"] != (2 if read_ahead else 1)):
            raise ValueError("actual read schedule/storage/drain changed")
        jobs, read_bytes, waited = record.get("read_ahead_jobs"), record.get("read_ahead_bytes"), record.get("read_wait_seconds")
        if (type(jobs) is not int or type(read_bytes) is not int or
            type(waited) not in (int, float) or not math.isfinite(waited) or waited < 0 or
            (jobs <= 0 or not 0 < read_bytes <= record["source_read_bytes"] if read_ahead else jobs != 0 or read_bytes != 0 or waited != 0)):
            raise ValueError("actual prefetched source/job/wait evidence missing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--registration-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--read-ahead-comparison", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("fresh evidence directory required")
    probe, library, source, cache = (p.resolve(strict=True) for p in
                                    (args.probe, args.library, args.source, args.registration_cache))
    hashes = {str(p): sha256_file(p) for p in (probe, library, source)}
    args.output.mkdir(parents=True)
    order = [(mib, ahead) for mib in (1, 4, 8) for ahead in (False, True)] if args.read_ahead_comparison else [(mib, False) for mib in ORDER]
    if args.read_ahead_comparison:
        order += list(reversed(order))
    summary = dict(schema="tc-qwen21-registered-buffer-window-v1", status="running", qualification_passed=False,
                   scope="full original encoder import, native registered proof and 13 gathered tokens; no full encoder forward timing",
                   source_identities=hashes, order=order, trials=[])
    destination = args.output / "summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    for index, (mib, ahead) in enumerate(order):
        stem = f"{index}-{mib}mib-{'ahead' if ahead else 'serial'}"
        print(json.dumps(dict(starting=stem)), flush=True)
        env = benchmark_environment(); env["TURBOCIDER_SOURCE_REGISTRATION_CACHE"] = str(cache)
        with (args.output / f"{stem}.stdout.json").open("x") as stdout, (args.output / f"{stem}.stderr.txt").open("x") as stderr:
            command = [str(probe), str(source), "8", "packed", "--registered", str(mib)] + (["--read-ahead"] if ahead else [])
            memory = run_sampled(command,
                repo=ROOT, output=args.output, stem=stem, env=env, stdout=stdout, stderr=stderr,
                timeout=180, interval_ms=100, max_gap_ms=500)
        record = json.loads((args.output / f"{stem}.stdout.json").read_text())
        validate_record(record, mib, hashes[str(source)], ahead)
        summary["trials"].append(dict(buffer_mib=mib, read_ahead=ahead, native=record, memory=memory))
        destination.write_text(json.dumps(summary, indent=2) + "\n")
    if any(sha256_file(p) != hashes[str(p)] for p in (probe, library, source)):
        raise ValueError("original source/probe/runtime changed during window")
    summary["medians"] = {f"{mib}-{'ahead' if ahead else 'serial'}": {field: statistics.median(t["native"][field] for t in summary["trials"]
                             if t["buffer_mib"] == mib and t["read_ahead"] == ahead) for field in
                             ("bank_load_seconds", "bank_read_seconds", "bank_decode_seconds", "total_observed_seconds")}
                          for mib, ahead in sorted(set(order))}
    summary["status"] = "complete_diagnostic"
    destination.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
