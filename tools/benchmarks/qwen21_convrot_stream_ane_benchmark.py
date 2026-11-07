#!/usr/bin/env python3
"""Serial ABBA/BAAB comparison of streamed GPU W8A8 and the fixed private ANE split.

Requires an explicitly private diagnostic build. Production packaging remains
GPU-only. Timed requests exclude the initial correctness/warmup tensor dumps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

from qwen21_convrot_w8a8_benchmark import (
    ROOT, BenchmarkMemoryGuard, arm_environment, decode_cli_records, image_metrics,
    image_passes, request_specs, run_sampled, sha256, source_snapshot, tensor_metrics,
    validate_record, write_json,
)

ANE_ENV = {
    "TURBOCIDER_QWEN21_CONVROT_STREAM_ANE": "1",
    "TURBOCIDER_ANE_BACKEND": "private",
    "TURBOCIDER_ALLOW_PRIVATE_ANE": "1",
    "TURBOCIDER_PRIVATE_ANE_DATA_PATH": "convrot_w8a8",
    "TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM": "1",
    "TURBOCIDER_RUNTIME_ANE_CHUNKS": "1",
    "TURBOCIDER_RUNTIME_ANE_FIXED_ASYNC": "1",
    "TURBOCIDER_RUNTIME_ANE_FP32_CHANNEL_JOIN": "1",
    "TURBOCIDER_PRIVATE_ANE_GPU_IO": "1",
    "TURBOCIDER_PRIVATE_ANE_PREFETCH": "0",
    "TURBOCIDER_PRIVATE_ANE_SCALE_CACHE": "0",
    "TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE": "1",
}
DESCRIPTOR = dict(backend="private_runtime_shape", biases=False, descriptor_version=1,
    hidden=4096, kind="swiglu", layout="out_in", lora_inputs=False, rows=1024,
    schema_version=1, tile_k=2048, tile_n=1024, width=12288)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_route(record, candidate):
    """Require completed calls and exact source/split policy, not toggle intent."""
    hybrid = record.get("hybrid") or {}
    if not candidate:
        require(not hybrid.get("runtime_calls_session_total", 0), "GPU control used Runtime ANE")
        return
    runtime = hybrid.get("runtime_weight") or {}
    expected = {
        "executor_backend": "private_ane", "data_path": "w8a8_convrot", "io_path": "gpu_iosurface",
        "partition_axis": "intermediate_channels", "ane_channels": 4096, "gpu_channels": 8192,
        "fp32_channel_join_enabled": True, "fixed_program_enabled": True, "program_disabled": False,
        "program_instances_session_total": 1, "fallback_blocks_session_total": 0,
        "prefetch_submissions_session_total": 0, "channel_blocks_session_total": 224,
        "untimed_hybrid_blocks_session_total": 224, "convrot_stage_submissions_session_total": 224,
    }
    mismatches = {key: {"actual": runtime.get(key), "expected": value}
                  for key, value in expected.items() if runtime.get(key) != value}
    require(not mismatches and hybrid.get("runtime_failed") is False,
            "Runtime ANE did not complete: " + repr(mismatches) + " " + str(runtime.get("failure_reason")))
    calls = hybrid.get("runtime_calls_session_total", 0)
    require(calls >= 224 and runtime.get("device_io_calls_session_total") == calls and
            runtime.get("capability_calls_session_total", 0) > 0 and
            0 < runtime.get("estimated_bytes", 0) <= 640 << 20 and
            runtime.get("source_recipe", "").endswith("+fp32-partial-join-v1"),
            "missing real Runtime ANE predictions or signed-source proof")
    phases = record.get("qwen_ffn_phases") or {}
    require(phases.get("prefill", {}).get("steps_this_request") == 1 and
            phases.get("decode", {}).get("steps_this_request") == 7 and
            phases.get("decode", {}).get("actual_rows") == 1024,
            "Runtime ANE changed prefill/decode geometry")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen-Image-2.1-Turbo-INT8")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--order", choices=("ABBA", "BAAB"), default="ABBA", help="A=GPU, B=GPU+ANE")
    args = parser.parse_args()
    cli, model = args.cli.resolve(strict=True), args.model.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    limit = 8 << 30
    library = cli.with_name("libturbocider.dylib")
    identity = json.loads((cli.parent / "runtime-build/runtime-build-manifest.json").read_text())
    manifest = output / "private-r1024.json"
    write_json(manifest, DESCRIPTOR)
    report = dict(status="running", scope="single-machine paired order; warmup excluded; two prompts, seed 73",
        cli=str(cli), cli_sha256=sha256(cli), library_sha256=sha256(library),
        runtime_build_id=identity["runtime_build_id"], script_sha256=sha256(Path(__file__)),
        manifest_sha256=sha256(manifest), max_footprint_bytes=limit, order=args.order, batches=[], comparisons=[])
    report_path = output / "report.json"
    try:
        require(all(sha256(ROOT / entry["path"]) == entry["sha256"] for entry in identity["inputs"]["sources"]),
                "native build does not match the current source tree")
        report["model_snapshot"] = source_snapshot(model, required=True)
        for index, letter in enumerate(args.order):
            arm = "gpu" if letter == "A" else "ane"
            folder = output / f"{index}-{arm}"
            folder.mkdir()
            settings = arm_environment("11")
            registration = folder / "source-registration"
            registration.mkdir(mode=0o700)
            settings["TURBOCIDER_SOURCE_REGISTRATION_CACHE"] = str(registration)
            if arm == "ane":
                settings.update(ANE_ENV)
                settings["TURBOCIDER_PRIVATE_ANE_CACHE_DIR"] = str(output / "program-cache")
            specs = request_specs(argparse.Namespace(warmup=1, repeats=args.repeats, smoke=False,
                dimension=512, seed=73, dump_tensors=False, residency="streamed"), folder)
            for spec in specs:
                request = spec["request"]
                if arm == "ane":
                    request.update(execution="gpu_ane", hybrid_mlp_mode="runtime", ane_manifest=str(manifest))
                if spec["case"] == "warmup":
                    request["dump_tensors"] = str(folder / "correctness")
                write_json(Path(spec["path"]), request)
            command = [str(cli), "batch", str(model), *(row["path"] for row in specs)]
            batch = dict(arm=arm, directory=str(folder), status="running", command=command, environment={
                k: v for k, v in settings.items() if k.startswith(("TURBOCIDER_", "MLX_"))})
            report["batches"].append(batch)
            write_json(report_path, report)
            print(f"{index}: {arm}, {len(specs)} Turbo8 requests", flush=True)
            guard = BenchmarkMemoryGuard(folder / "process-memory.jsonl", limit)
            start = time.perf_counter()
            try:
                with (folder / "stdout.json").open("x") as stdout, (folder / "stderr.log").open("x") as stderr:
                    batch["memory"] = run_sampled(command, repo=ROOT, output=folder, stem="process", env=settings,
                        stdout=stdout, stderr=stderr, timeout=1800, interval_ms=100, max_gap_ms=500,
                        observer=guard, async_evidence=True)
            finally:
                if guard.reason:
                    batch["memory_stop"] = guard.reason
            batch["harness_seconds"] = time.perf_counter() - start
            require(not guard.reason, "physical-memory guard stopped the batch")
            records = decode_cli_records((folder / "stdout.json").read_text())
            require(len(records) == len(specs), "batch did not complete every request")
            batch["runs"] = []
            for spec, record in zip(specs, records):
                validate_record(record, spec, "11")
                validate_route(record, arm == "ane")
                write_json(folder / f"{spec['index']:02d}-result.json", record)
                batch["runs"].append(dict(case=spec["case"], result=record,
                    png=spec["request"]["output"], png_sha256=sha256(Path(spec["request"]["output"]))))
            memory = batch["memory"]["verified"]
            require(memory["complete"] and memory["command_exit_code"] == 0 and
                memory["tree_peak_phys_footprint_bytes"] <= limit and memory["swap_out_bytes"] == 0,
                "incomplete process measurement or new swap-out")
            require(source_snapshot(model, required=True) == report["model_snapshot"] and
                sha256(cli) == report["cli_sha256"] and sha256(library) == report["library_sha256"],
                "model or executable changed during measurement")
            batch["status"] = "complete"
            write_json(report_path, report)
        # Every candidate has a matched GPU request with the same prompt and seed.
        control_batch = next(batch for batch in report["batches"] if batch["arm"] == "gpu")
        controls = control_batch["runs"]
        for batch in report["batches"]:
            if batch is control_batch:
                continue
            for control, candidate in zip(controls, batch["runs"]):
                pixels = image_metrics(Path(control["png"]), Path(candidate["png"]))
                report["comparisons"].append(dict(arm=batch["arm"], case=candidate["case"], pixels=pixels,
                    passed=image_passes(pixels, min_correlation=.99, min_cosine=.995, max_mae_255=5.)))
        report["tensor_comparisons"] = []
        for index, batch in enumerate(report["batches"]):
            if batch is control_batch:
                continue
            tensors = {name: tensor_metrics(Path(control_batch["directory"]) / "correctness" / f"{name}.safetensors",
                Path(batch["directory"]) / "correctness" / f"{name}.safetensors")
                for name in ("qwen21_text", "qwen21_initial", "qwen21_latents")}
            require(all(value["finite"] for value in tensors.values()) and
                tensors["qwen21_text"]["exact"] and tensors["qwen21_initial"]["exact"],
                "conditioning/noise differed or candidate returned nonfinite latents")
            report["tensor_comparisons"].append(dict(batch=index, tensors=tensors))
        def median(arm, case, phase):
            return statistics.median(row["result"]["timings_seconds"][phase]
                for batch in report["batches"] if batch["arm"] == arm for row in batch["runs"] if row["case"] == case)
        report["summary"] = {case: {phase: {"gpu_seconds": median("gpu", case, phase),
            "ane_seconds": median("ane", case, phase),
            "gpu_over_ane": median("gpu", case, phase) / median("ane", case, phase)}
            for phase in ("request_wall", "text_encode", "denoise", "vae_decode")} for case in ("short", "long")}
        report["status"] = "complete" if all(row["passed"] for row in report["comparisons"]) else "quality-gate-failed"
    except Exception as error:
        report.update(status="failed", error=str(error))
    write_json(report_path, report)
    print(json.dumps(dict(status=report["status"], report=str(report_path), error=report.get("error"))), flush=True)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
