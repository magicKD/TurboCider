#!/usr/bin/env python3
"""Fresh-process full encoder import screens, with complete native SHA proof."""
import argparse
import json
import math
from pathlib import Path
import statistics

from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_sampled

ROOT = Path(__file__).resolve().parents[2]
ORDER = ((1, "packed"), (1, "dense"), (4, "packed"), (8, "packed"),
         (8, "packed"), (4, "packed"), (1, "dense"), (1, "packed"))


def validate_record(record, workers, mode, source_hash, source_bytes):
    if (record.get("schema") != "tc-qwen21-encoder-import-screen-v1" or
        record.get("qualification_passed") is not False or record.get("source_sha256") != source_hash or
        type(record.get("workers")) is not int or record["workers"] != workers or
        record.get("embedding_mode") != mode):
        raise ValueError("wrong actual source/import recipe")
    for field in ("verification_bytes", "source_read_bytes", "packed_capacity_bytes", "read_buffer_capacity_bytes",
                  "managed_peak_bytes", "final_weight_logical_bytes", "mlx_logical_peak_bytes"):
        if type(record.get(field)) is not int or record[field] <= 0:
            raise ValueError("missing actual positive byte counter: " + field)
    if (record["verification_bytes"] != source_bytes or record["read_buffer_capacity_bytes"] != 1 << 20 or
        record["managed_peak_bytes"] > record["packed_capacity_bytes"] + (1 << 20)):
        raise ValueError("full proof or bounded shared read-buffer contract differs")
    for field in ("native_verification_seconds", "bank_load_seconds", "bank_read_seconds", "bank_decode_seconds",
                  "embedding_setup_seconds", "token_gather_seconds", "total_observed_seconds"):
        if type(record.get(field)) not in (int, float) or not math.isfinite(record[field]) or record[field] < 0:
            raise ValueError("invalid observed stage timing: " + field)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("fresh evidence directory required")
    probe, library, source = (p.resolve(strict=True) for p in (args.probe, args.library, args.source))
    hashes = {str(p): sha256_file(p) for p in (probe, library, source)}
    args.output.mkdir(parents=True)
    summary = dict(schema="tc-qwen21-encoder-import-window-v1", status="running", qualification_passed=False,
                   scope="same binary forward/reverse fresh-process import; not whole encoder/model qualification",
                   source_identities=hashes, order=ORDER, trials=[])
    destination = args.output / "summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    for index, (workers, mode) in enumerate(ORDER):
        stem = f"{index}-w{workers}-{mode}"
        print(json.dumps({"starting": stem}), flush=True)
        with (args.output / f"{stem}.stdout.json").open("x") as stdout, (args.output / f"{stem}.stderr.txt").open("x") as stderr:
            memory = run_sampled([str(probe), str(source), str(workers), mode], repo=ROOT, output=args.output,
                                 stem=stem, env=benchmark_environment(), stdout=stdout, stderr=stderr,
                                 timeout=180, interval_ms=100, max_gap_ms=500)
        record = json.loads((args.output / f"{stem}.stdout.json").read_text())
        validate_record(record, workers, mode, hashes[str(source)], source.stat().st_size)
        summary["trials"].append(dict(stem=stem, native=record, memory=memory))
        destination.write_text(json.dumps(summary, indent=2) + "\n")
    if any(sha256_file(p) != hashes[str(p)] for p in (probe, library, source)):
        raise ValueError("actual source/runtime changed during window")
    summary["medians"] = {f"w{workers}-{mode}": {
        field: statistics.median(t["native"][field] for t in summary["trials"]
                                 if t["native"]["workers"] == workers and t["native"]["embedding_mode"] == mode)
        for field in ("native_verification_seconds", "bank_load_seconds", "bank_read_seconds", "bank_decode_seconds",
                      "embedding_setup_seconds", "token_gather_seconds", "total_observed_seconds")}
        for workers, mode in dict.fromkeys(ORDER)}
    summary["status"] = "complete_diagnostic"
    destination.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
