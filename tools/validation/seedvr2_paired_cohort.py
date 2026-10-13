#!/usr/bin/env python3
"""Balanced ABBA full-request speed evidence for one immutable SR workload.

Each cohort has its own process/server, one cold run, and at least three
genuinely executed warm requests. This proves only the stated speed gate;
full media quality, other resolutions and GPU+ANE are separate requirements.
Never contacts a user's existing ComfyUI service.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

from seedvr2_framework_screen import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("model", "source", "library", "comfy", "comfy-python", "comfy-extra", "output"):
        parser.add_argument("--"+field, type=Path, required=True)
    parser.add_argument("--kind", choices=("image", "video"), required=True)
    parser.add_argument("--size", type=int, choices=(512, 1024), required=True)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--comfy-route", choices=("comfy-default", "comfy-sdpa-gpu"), default="comfy-sdpa-gpu")
    args = parser.parse_args()
    if not 4 <= args.count <= 10:
        parser.error("at least three warm repeats per cohort are required (count 4...10)")
    if args.kind == "image" and args.frames != 1:
        parser.error("image workload must have one frame")
    if args.output.exists():
        parser.error("a fresh cohort directory is required")
    for name in ("model", "source", "library", "comfy", "comfy_python", "comfy_extra", "output"):
        setattr(args, name, getattr(args, name).absolute())
    # The observer needs aiohttp/YAML/Pillow, independently of the model's
    # server environment. Do not require those packages in the native venv.
    subprocess.run([str(args.comfy_python), "-c", "import aiohttp, yaml; from PIL import Image"], check=True)
    args.output.mkdir(parents=True)
    source_sha, library_sha = sha256(args.source), sha256(args.library)
    record = dict(schema="seedvr2-framework-paired-cohort-v1", status="running",
        order=["native-gpu", args.comfy_route, args.comfy_route, "native-gpu"],
        workload=dict(kind=args.kind, size=args.size, frames=args.frames, fps=args.fps,
                      seed=args.seed, count=args.count),
        source_sha256=source_sha, library_sha256=library_sha, cohorts=[],
        speed_gate_passed=False, overall_qualification_passed=False,
        scope="balanced full-request warm speed for this workload/reference only; not a complete quality/ANE gate")
    artifact_sha = None
    baseline_identity = None
    model_identity = None
    try:
        for index, route in enumerate(record["order"]):
            if sha256(args.source) != source_sha or sha256(args.library) != library_sha:
                raise ValueError("source or native library changed during the paired cohort")
            destination = args.output / f"{index}-{route}"
            observer_python = str(args.comfy_python) if route.startswith("comfy-") else sys.executable
            command = [observer_python, str(Path(__file__).with_name("seedvr2_framework_screen.py")),
                "--route", route, "--model", str(args.model), "--source", str(args.source),
                "--kind", args.kind, "--width", str(args.size), "--height", str(args.size),
                "--frames", str(args.frames), "--fps", str(args.fps), "--seed", str(args.seed),
                "--count", str(args.count), "--output", str(destination), "--library", str(args.library),
                "--comfy", str(args.comfy), "--comfy-python", str(args.comfy_python),
                "--comfy-extra", str(args.comfy_extra)]
            write_json(args.output / f"command-{index}.json", dict(argv=command,
                environment={key:value for key,value in os.environ.items() if key.startswith("TURBOCIDER_SEEDVR2_")}))
            print(f"cohort {index+1}/4 {route}: {args.kind} {args.size} t{args.frames}", flush=True)
            subprocess.run(command, check=True)
            observed = json.loads((destination / "report.json").read_text())
            if observed["status"] != "complete" or len(observed["runs"]) != args.count or observed["source_sha256"] != source_sha:
                raise ValueError("incomplete/mismatched full request evidence")
            current_model = observed["model_provenance"]
            if model_identity is not None and current_model != model_identity:
                raise ValueError("original checkpoint provenance changed")
            model_identity = current_model
            if any(run["index"] != offset or not math.isfinite(run["request_wall_including_export"]) or
                   run["request_wall_including_export"] <= 0 for offset,run in enumerate(observed["runs"])):
                raise ValueError("invalid/missing ordered timing evidence")
            if route == "native-gpu":
                identity = observed["native_artifact_identity"]
                if identity["library_sha256"] != library_sha:
                    raise ValueError("native library identity changed")
                current = identity["runtime_build_manifest_sha256"]
                if artifact_sha is not None and current != artifact_sha:
                    raise ValueError("native build identity changed")
                artifact_sha = current
            else:
                if not observed.get("server_terminal"):
                    raise ValueError("owned ComfyUI server did not exit")
                # Bind both Comfy cohorts to the exact version/local patch.
                current = {key:value for key,value in observed.items()
                           if key.startswith("comfy_") or key in ("model_provenance",)}
                system = observed["system_stats"]["system"]
                current["runtime_versions"] = {key:system[key] for key in (
                    "os", "comfyui_version", "python_version", "pytorch_version", "comfy_package_versions")}
                if baseline_identity is not None and current != baseline_identity:
                    raise ValueError("Comfy reference identity changed")
                baseline_identity = current
            warm = [run["request_wall_including_export"] for run in observed["runs"][1:]]
            record["cohorts"].append(dict(index=index, route=route, report=str(destination / "report.json"),
                cold_request_seconds=observed["runs"][0]["request_wall_including_export"],
                warm_seconds=warm, warm_median_seconds=statistics.median(warm)))
            write_json(args.output / "report.json", record)
        native = [v for c in record["cohorts"] if c["route"] == "native-gpu" for v in c["warm_seconds"]]
        reference = [v for c in record["cohorts"] if c["route"] != "native-gpu" for v in c["warm_seconds"]]
        native_median, reference_median = statistics.median(native), statistics.median(reference)
        record.update(status="complete", native_warm_median_seconds=native_median,
            reference_warm_median_seconds=reference_median, speedup=reference_median/native_median,
            speed_gate_passed=native_median <= reference_median,
            native_runtime_build_manifest_sha256=artifact_sha)
        print(json.dumps({key:record[key] for key in ("status","native_warm_median_seconds",
            "reference_warm_median_seconds","speedup","speed_gate_passed")}), flush=True)
    except BaseException as error:
        record.update(status="failed", failure=str(error))
        raise
    finally:
        write_json(args.output / "report.json", record)


if __name__ == "__main__":
    main()
