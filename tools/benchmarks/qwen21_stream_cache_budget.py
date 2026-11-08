#!/usr/bin/env python3
"""Matched forward/reverse budget sweep, exact outputs, and sampled process RAM.

Each budget gets a fresh CLI process, a correctness/warmup request and two timed
prompts. Increasing then decreasing order reduces ordering bias. The budget is
managed weight capacity, never a whole-process limit. All timed requests retain
the original eight-step schedule and the selected ConvRot activation route.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from qwen21_convrot_w8a8_benchmark import (
    ROOT, BenchmarkMemoryGuard, arm_environment, decode_cli_records, image_metrics,
    request_specs, run_sampled, sha256, source_snapshot, tensor_metrics, validate_record, write_json,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen-Image-2.1-Turbo-INT8")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets-mib", type=int, nargs="+", default=[0, 1024, 2048, 4096, 6144, 8192])
    parser.add_argument("--dimension", type=int, choices=(256, 512), default=512)
    parser.add_argument("--route", choices=("11", "00"), default="11", help="11=M5 W8A8; 00=forced Float-A")
    parser.add_argument("--single-order", action="store_true", help="correctness smoke only, no reverse-order repeat")
    args = parser.parse_args()
    require(args.budgets_mib[0] == 0 and len(set(args.budgets_mib)) == len(args.budgets_mib) and
            all(n == 0 or 1024 <= n <= 16384 for n in args.budgets_mib), "budgets must start at 0 and be unique, then 1024...16384 MiB")
    cli, model = args.cli.resolve(strict=True), args.model.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    library = cli.with_name("libturbocider.dylib")
    identity = json.loads((cli.parent / "runtime-build/runtime-build-manifest.json").read_text())
    limit = 12 << 30
    report = dict(status="running", route=args.route, dimension=args.dimension, budget_order_mib=[], batches=[],
        runtime_build_id=identity["runtime_build_id"], cli_sha256=sha256(cli), library_sha256=sha256(library),
        script_sha256=sha256(Path(__file__)), model_snapshot=source_snapshot(model, required=True),
        scope="single machine; one warmup then short/long; forward/reverse order; weight cap excludes activations/VAE")
    path = output / "report.json"
    controls = None
    try:
        require(all(sha256(ROOT / entry["path"]) == entry["sha256"] for entry in identity["inputs"]["sources"]),
                "native build does not match the current source tree")
        order = args.budgets_mib if args.single_order else args.budgets_mib + args.budgets_mib[::-1]
        report["budget_order_mib"] = order
        for index, mib in enumerate(order):
            folder = output / f"{index:02d}-{mib}MiB"
            folder.mkdir()
            specs = request_specs(argparse.Namespace(warmup=1, repeats=1, smoke=False,
                dimension=args.dimension, seed=73, dump_tensors=False, residency="streamed"), folder)
            for spec in specs:
                if mib:
                    spec["request"]["qwen21_stream_weight_budget_bytes"] = mib << 20
                if spec["warmup"]:
                    spec["request"]["dump_tensors"] = str(folder / "correctness")
                write_json(Path(spec["path"]), spec["request"])
            settings = arm_environment(args.route)
            settings["MLX_ENABLE_TF32"] = "0"
            settings["TURBOCIDER_SOURCE_REGISTRATION_CACHE"] = str(folder / "source-registration")
            batch = dict(budget_mib=mib, directory=str(folder), status="running", runs=[])
            report["batches"].append(batch)
            write_json(path, report)
            print(f"{index + 1}/{len(order)}: {mib} MiB, route {args.route}", flush=True)
            guard = BenchmarkMemoryGuard(folder / "process-memory.jsonl", limit)
            with (folder / "stdout.json").open("x") as stdout, (folder / "stderr.log").open("x") as stderr:
                batch["memory"] = run_sampled([str(cli), "batch", str(model), *(s["path"] for s in specs)],
                    repo=ROOT, output=folder, stem="process", env=settings, stdout=stdout, stderr=stderr,
                    timeout=1800, interval_ms=100, max_gap_ms=500, observer=guard, async_evidence=True)
            memory = batch["memory"]["verified"]
            require(not guard.reason and memory["complete"] and memory["command_exit_code"] == 0 and
                    memory["tree_peak_phys_footprint_bytes"] <= limit and memory["swap_out_bytes"] == 0,
                    "physical memory guard, incomplete process sample, nonzero exit or new swap-out")
            records = decode_cli_records((folder / "stdout.json").read_text())
            require(len(records) == len(specs), "batch omitted a request")
            for spec, record in zip(specs, records):
                validate_record(record, spec, args.route)
                execution = record["qwen_convrot_w8a8"]
                require(execution["encoder_projections"] == (252 if args.route == "11" else 0) and
                        execution["dit_projections"] == (1536 if args.route == "11" else 0), "incomplete projection coverage")
                write_json(folder / f"{spec['index']:02d}-result.json", record)
                batch["runs"].append(dict(case=spec["case"], request=spec["request"], result=record))
            if controls is None:
                controls = batch
            batch["correctness"] = dict(
                tensors={name: tensor_metrics(Path(controls["directory"]) / "correctness" / f"{name}.safetensors",
                    folder / "correctness" / f"{name}.safetensors") for name in ("qwen21_text", "qwen21_initial", "qwen21_latents")},
                pixels=[image_metrics(Path(a["request"]["output"]), Path(b["request"]["output"]))
                    for a, b in zip(controls["runs"], batch["runs"])])
            require(all(v["exact"] and v["finite"] for v in batch["correctness"]["tensors"].values()) and
                    all(v["mae_255"] == 0 for v in batch["correctness"]["pixels"]), "cache changed exact tensor values or pixels")
            require(source_snapshot(model, required=True) == report["model_snapshot"] and
                    sha256(cli) == report["cli_sha256"] and sha256(library) == report["library_sha256"], "model/executable mutated")
            batch["status"] = "complete"
            write_json(path, report)
        report["summary"] = []
        for mib in args.budgets_mib:
            batches = [b for b in report["batches"] if b["budget_mib"] == mib]
            runs = [r for b in batches for r in b["runs"] if r["case"] != "warmup"]
            stages = [r["result"]["qwen_convrot_streaming"]["denoiser"] for r in runs]
            report["summary"].append(dict(budget_mib=mib, cached_layers=[s["prefix"] for s in stages],
                budget_target_layers=[s["cache_budget_prefix"] for s in stages],
                cache_evictions=[s.get("cache_evictions",0) for s in stages],
                prefix_per_pass=[s.get("prefix_per_pass",[s["prefix"]]*8) for s in stages],
                managed_weight_capacity_bytes=[s["managed_weight_capacity_bytes"] + s["derived_weight_capacity_bytes"] for s in stages],
                logical_dit_read_bytes=[s["resident_source_bytes"] + s["streamed_source_bytes"] for s in stages],
                sampled_process_peak_bytes=max(b["memory"]["verified"]["tree_peak_phys_footprint_bytes"] for b in batches),
                timings={case: {phase: statistics.median(r["result"]["timings_seconds"][phase]
                    for r in runs if r["case"] == case) for phase in ("request_wall", "text_encode", "denoise", "vae_decode")}
                    for case in ("short", "long")}))
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        if report["batches"]:
            batch = report["batches"][-1]
            batch["status"] = "failed"
            evidence = Path(batch["directory"]) / "process-memory-report.json"
            if evidence.is_file():
                batch["failed_process_evidence"] = json.loads(evidence.read_text())
        raise
    finally:
        write_json(path, report)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
