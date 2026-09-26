#!/usr/bin/env python3
"""Test register-cached Q fragments without modifying the production shader."""
import argparse
import json
import statistics
import subprocess
from pathlib import Path

from benchmark_ltx_qkv_replay import digest, validate_head_metrics


def query_cache_shader(source):
    declaration = "    ltx_sol_mma_tile<float, 1, 1, Fragment> query_tile;"
    load = ("            query_tile.template load<bfloat, 1, 1, QUERY_LD, 1>(\n"
            "                &query_shared[query_offset + dimension_tile * 8]);")
    boundary = ("    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                "    for (uint summary_block_start = 0u;")
    for needle, count in ((declaration, 1), (load, 2), (boundary, 1)):
        if source.count(needle) != count:
            raise ValueError("shader layout changed; review the query-cache transform")
    source = source.replace(declaration, declaration +
        "\n    ltx_sol_mma_tile<float, 1, 1, Fragment> cached_query[16];")
    source = source.replace(load, "            query_tile = cached_query[dimension_tile];")
    source = source.replace(boundary,
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "    LTX_SOL_UNROLL\n"
        "    for (short dimension_tile = 0; dimension_tile < 16; ++dimension_tile) {\n"
        "        cached_query[dimension_tile].template load<bfloat, 1, 1, QUERY_LD, 1>(\n"
        "            &query_shared[query_offset + dimension_tile * 8]);\n"
        "    }\n"
        "    for (uint summary_block_start = 0u;")
    return source


def main(transform=query_cache_shader, variant="query-cache", configurations=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=9)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.runs <= 100:
        parser.error("use a new output directory and runs in 1...100")
    root = Path(__file__).resolve().parents[2]
    shader = root / "native/models/ltx_runtime/ltx_shaders.metal"
    probe = root / "build/native/ltx-attention-probe"
    source = shader.read_text()
    transformed = transform(source)
    fixtures = []
    for path in args.fixture:
        path = path.resolve()
        meta = json.loads((path / "metadata.json").read_text())
        if (meta["schema"] != "ltx-qkv-replay-v1" or meta["dim"] != 128 or
                meta["dtype"] != "bf16" or meta["qkv_layout"] != "row-head-dim" or
                meta["rope_layout"] != "head-row-halfdim"):
            parser.error("unsupported fixture layout")
        fixtures.append((path, meta))
    args.output.mkdir(parents=True)
    paths = {}
    for name, text in (("current", source), (variant, transformed)):
        directory = args.output.resolve() / name
        directory.mkdir()
        paths[name] = directory / "ltx_shaders.metal"
        with paths[name].open("x") as stream:
            stream.write(text)
    report = dict(schema=f"ltx-{variant}-experiment-v1", complete=False,
                  scope="attention core only; not Stage-2 speed or video quality",
                  probe_sha256=digest(probe), source_sha256=digest(shader),
                  runner_sha256=digest(Path(__file__)),
                  variants={name: digest(path) for name, path in paths.items()},
                  fixtures=[dict(path=str(path), metadata=meta,
                    sha256={name: digest(path / name) for name in (
                        "query.bf16", "key.bf16", "value.bf16", "cosine.bf16", "sine.bf16", "gate.bf16")})
                    for path, meta in fixtures], parity=[], runs=[], comparisons=[])
    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    def run(name, rows, heads, mode, radius, keep, iterations, fixture=None):
        command = [str(probe), str(paths[name]), str(rows), str(heads), str(iterations),
                   "0", str(mode), str(radius), "0", "0", str(keep)]
        if fixture is not None:
            command.append(str(fixture))
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            report["failure"] = dict(command=command, stderr=result.stderr, stdout=result.stdout)
            save()
            raise RuntimeError(result.stderr or result.stdout)
        row = json.loads(result.stdout)
        validate_head_metrics(row)
        return dict(variant=name, command=command, metrics=row)
    save()
    # Scalar oracle and materialized-route checks live in the native probe.
    for rows in (1, 65, 257):
        for mode in range(6):
            report["parity"].append(run(variant, rows, 2, mode, 0, 2, 2))
            save()
    fields = ("relative_l2", "cosine", "max_abs", "exact_block_fraction", "per_head")
    for fixture, meta in fixtures:
        for label, mode, radius, keep in configurations or (("all-exact", 1, 256, 0),
                                          ("pooled-topk32", 5, 0, 32),
                                          ("pooled-topk64", 5, 0, 64)):
            measured = []
            for name in ("current", variant, variant, "current"):
                row = run(name, meta["rows"], meta["heads"], mode, radius, keep, args.runs, fixture)
                row.update(fixture=str(fixture), label=label)
                measured.append(row)
                report["runs"].append(row)
                save()
            medians = {name: statistics.median(r["metrics"]["sol_seconds"] for r in measured
                                               if r["variant"] == name) for name in paths}
            equal = all(all(row["metrics"][field] == measured[0]["metrics"][field] for field in fields)
                        for row in measured)
            comparison = dict(fixture=str(fixture), label=label, seconds=medians,
                              **{("current_over_cache" if variant == "query-cache" else
                                  "current_over_variant"): medians["current"] / medians[variant]},
                              equal_reported_numerical_metrics=equal,
                              method="ABBA processes; each probe alternates dense/sparse with two warmup pairs")
            report["comparisons"].append(comparison)
            save()
            print(json.dumps(comparison), flush=True)
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
