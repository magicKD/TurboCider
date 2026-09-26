#!/usr/bin/env python3
"""Compare matched float-route and packed-route probe/shader pairs."""
import argparse
import json
import shutil
import statistics
import subprocess
from pathlib import Path

from benchmark_ltx_qkv_replay import digest, validate_head_metrics


def equal_metrics(left, right):
    return all(left[key] == right[key] for key in
               ("relative_l2", "cosine", "max_abs", "exact_block_fraction", "per_head"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-directory", type=Path, required=True)
    parser.add_argument("--fixture", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.runs <= 100:
        parser.error("use fresh output and runs in 1...100")
    root = Path(__file__).resolve().parents[2]
    fixture_metadata = [(path.resolve(), json.loads((path / "metadata.json").read_text()))
                        for path in args.fixture]
    for _, meta in fixture_metadata:
        if (meta["schema"] != "ltx-qkv-replay-v1" or meta["dim"] != 128 or
                meta["qkv_layout"] != "row-head-dim" or meta["rope_layout"] != "head-row-halfdim" or
                meta["dtype"] != "bf16"):
            parser.error("unsupported fixture")
    args.output.mkdir(parents=True)
    variants = {}
    for label, binary, shader in (
            ("float", args.baseline_directory / "ltx-attention-probe",
             args.baseline_directory / "ltx_shaders.metal"),
            ("packed", root / "build/native/ltx-attention-probe",
             root / "native/models/ltx_runtime/ltx_shaders.metal")):
        directory = args.output.resolve() / label
        directory.mkdir()
        for source in (binary, shader):
            shutil.copy2(source, directory / source.name)
        variants[label] = directory
    report = dict(schema="ltx-route-layout-abba-v1", complete=False,
                  scope="attention core; not full Stage-2 or decoded video",
                  runner_sha256=digest(Path(__file__)),
                  sources={name: {p.name: digest(p) for p in directory.iterdir()}
                           for name, directory in variants.items()},
                  fixtures=[dict(path=str(path), metadata=meta, sha256={
                      name: digest(path / name) for name in ("query.bf16", "key.bf16", "value.bf16",
                                                           "cosine.bf16", "sine.bf16", "gate.bf16")})
                      for path, meta in fixture_metadata], boundaries=[], runs=[], comparisons=[])
    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    def run(variant, n, heads, mode, radius, keep, runs, fixture=None):
        directory = variants[variant]
        command = [str(directory / "ltx-attention-probe"), str(directory / "ltx_shaders.metal"),
                   str(n), str(heads), str(runs), ".5", str(mode), str(radius), "0", "0", str(keep)]
        if fixture:
            command.append(str(fixture))
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            report["failure"] = dict(command=command, stderr=result.stderr, stdout=result.stdout)
            save()
            raise RuntimeError(result.stderr or result.stdout)
        row = json.loads(result.stdout)
        validate_head_metrics(row)
        if variant == "packed" and row.get("route_layout") != "packed-u32-v1":
            raise RuntimeError("rebuild the packed-route probe")
        return dict(variant=variant, command=command, metrics=row)
    save()
    for n in (2048, 2049, 8192, 8193, 16384):
        for mode in range(6):
            rows = [run(name, n, 2, mode, 0, 32, 1) for name in variants]
            same = equal_metrics(rows[0]["metrics"], rows[1]["metrics"])
            report["boundaries"].append(dict(rows=n, mode=mode, runs=rows, equal_reported_metrics=same))
            save()
            if not same:
                raise RuntimeError("float/packed boundary numerical metrics differ")
    configs = [(f"mode-{mode}", mode, 1 if mode < 4 else 0, 32) for mode in range(6)]
    configs += [("all-exact", 1, 256, 0), ("pooled-topk64", 5, 0, 64)]
    for fixture, meta in fixture_metadata:
        for label, mode, radius, keep in configs:
            rows = []
            for name in ("float", "packed", "packed", "float"):
                row = run(name, meta["rows"], meta["heads"], mode, radius, keep, args.runs, fixture)
                row.update(label=label, fixture=str(fixture))
                rows.append(row)
                report["runs"].append(row)
                save()
            same = all(equal_metrics(rows[0]["metrics"], r["metrics"]) for r in rows)
            times = {name: statistics.median(r["metrics"]["sol_seconds"] for r in rows
                                            if r["variant"] == name) for name in variants}
            summary = dict(fixture=str(fixture), label=label, seconds=times,
                           float_over_packed=times["float"] / times["packed"],
                           equal_reported_metrics=same,
                           method="ABBA processes, alternating dense/sparse per probe, two warmup pairs")
            report["comparisons"].append(summary)
            save()
            print(json.dumps(summary), flush=True)
            if not same:
                raise RuntimeError("float/packed real-QKV numerical metrics differ")
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
