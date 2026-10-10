#!/usr/bin/env python3
"""Serial real layer0 FFN packed vs bounded dequantize/dense GPU screening."""
import argparse
import json
import math
from pathlib import Path
import statistics

from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_sampled

ROOT = Path(__file__).resolve().parents[2]
ROWS = (32, 38, 128, 512, 1024, 1048, 2112, 4096)


def validate_record(record, component, source_hash, direct=False):
    if (record.get("schema") != "tc-qwen21-real-ffn-dense-screen-v1" or
        record.get("qualification_passed") is not False or record.get("component") != component or
        record.get("source_sha256") != source_hash or record.get("fresh_packed_arguments_dynamic") is not True):
        raise ValueError("wrong original source or dequantize execution contract")
    if record.get("direct_u4_mpp") is not direct:
        raise ValueError("actual direct U4 schedule differs")
    expected = dict(verification_bytes=0, registration_cache_hits=1, selected_tensor_count=2 if component == "dit" else 3,
                    dense_blocks_retained=1, group_size=32, K=4096, warmups=3, hot_samples=9)
    for field, value in expected.items():
        if type(record.get(field)) is not int or record[field] != value:
            raise ValueError("invalid exact probe counter: " + field)
    if (type(record.get("intermediate")) is not int or record["intermediate"] <= 0 or
        any(type(record.get(name)) is not int or record[name] not in (4, 8) for name in ("gate_bits", "down_bits"))):
        raise ValueError("invalid actual affine FFN geometry")
    if direct and (record["gate_bits"] != 4 or record["down_bits"] != 4):
        raise ValueError("direct U4 cannot reinterpret Q8 source planes")
    dense_bytes = 2 * (2 * record["intermediate"] * 4096 + 4096 * record["intermediate"])
    if type(record.get("dense_block_bytes")) is not int or record["dense_block_bytes"] != dense_bytes:
        raise ValueError("dense block capacity differs from actual geometry")
    for field in ("source_read_bytes", "packed_capacity_bytes", "mlx_logical_peak_bytes"):
        if type(record.get(field)) is not int or record[field] <= 0:
            raise ValueError("missing actual source/MLX byte counter")
    timing = record.get("initial_dequantize_seconds")
    if type(timing) not in (int, float) or not math.isfinite(timing) or timing <= 0:
        raise ValueError("invalid initial real dequantize timing")
    cases = record.get("cases")
    if not isinstance(cases, list) or [r.get("M") for r in cases] != list(ROWS):
        raise ValueError("complete uncached-prefill row sweep missing")
    for case in cases:
        if (type(case["M"]) is not int or type(case.get("complete_output_elements")) is not int or
            case["complete_output_elements"] != case["M"] * 4096):
            raise ValueError("complete FFN outputs not compared")
        for field in ("fresh_relative_l2", "cached_relative_l2", "fresh_cached_relative_l2"):
            value = case.get(field)
            if (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= .05 or
                (field == "fresh_cached_relative_l2" and value != 0)):
                raise ValueError("real complete FFN numerical contract failed")
        for field in ("packed_seconds", "fresh_dequant_dense_seconds", "cached_dense_seconds"):
            values = case.get(field)
            if (not isinstance(values, list) or len(values) != 9 or
                any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values)):
                raise ValueError("missing actual evaluated GPU sample span")
        if direct:
            error, samples = case.get("direct_u4_relative_l2"), case.get("direct_u4_mpp_seconds")
            if (type(error) not in (int, float) or not math.isfinite(error) or not 0 <= error <= .05 or
                not isinstance(samples, list) or len(samples) != 9 or
                any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in samples)):
                raise ValueError("missing/invalid real direct U4 complete FFN error or spans")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--registration-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direct-u4", action="store_true", help="validation-only Splash-style group32 integer matrix/group sums")
    parser.add_argument("--components", nargs="+", choices=("dit", "encoder"), default=("dit", "encoder"))
    args = parser.parse_args()
    if len(args.components) != len(set(args.components)):
        parser.error("distinct actual components required")
    if args.output.exists() or args.output.is_symlink():
        parser.error("fresh evidence directory required")
    probe, library, model, cache = (p.resolve(strict=True) for p in
                                  (args.probe, args.library, args.model, args.registration_cache))
    sources = dict(dit=model / "diffusion_models/qwen-image-2.1-Q4_K_M.gguf",
                   encoder=model / "text_encoders/Qwen3-VL-8B-Instruct-Q4_K_M.gguf")
    sources = {name: sources[name] for name in args.components}
    hashes = {str(p): sha256_file(p) for p in (probe, library, *sources.values())}
    args.output.mkdir(parents=True)
    summary = dict(schema="tc-qwen21-real-ffn-dense-window-v1", status="running", qualification_passed=False,
                   scope="one actual layer0 FFN at a time with synthetic FP16 inputs; no full-model/all-layer/ahead decode qualification",
                   direct_u4=args.direct_u4, components=args.components, source_identities=hashes, trials=[])
    destination = args.output / "summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    for component, source in sources.items():
        print(json.dumps(dict(starting=component)), flush=True)
        env = benchmark_environment()
        env["TURBOCIDER_SOURCE_REGISTRATION_CACHE"] = str(cache)
        with (args.output / f"{component}.stdout.json").open("x") as stdout, (args.output / f"{component}.stderr.txt").open("x") as stderr:
            command = [str(probe), str(source), component] + (["--direct-u4"] if args.direct_u4 else [])
            memory = run_sampled(command, repo=ROOT, output=args.output, stem=component,
                                 env=env, stdout=stdout, stderr=stderr, timeout=300, interval_ms=100, max_gap_ms=500)
        record = json.loads((args.output / f"{component}.stdout.json").read_text())
        validate_record(record, component, hashes[str(source)], args.direct_u4)
        timing_fields = ("packed_seconds", "fresh_dequant_dense_seconds", "cached_dense_seconds") + (
            ("direct_u4_mpp_seconds",) if args.direct_u4 else ())
        medians = [dict(M=case["M"], **{field: statistics.median(case[field]) for field in timing_fields})
                   for case in record["cases"]]
        summary["trials"].append(dict(component=component, native=record, medians=medians, memory=memory))
        destination.write_text(json.dumps(summary, indent=2) + "\n")
    if any(sha256_file(p) != hashes[str(p)] for p in (probe, library, *sources.values())):
        raise ValueError("original source/probe/runtime changed during window")
    summary["status"] = "complete_diagnostic"
    destination.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
