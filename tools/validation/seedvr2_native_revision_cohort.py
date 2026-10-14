#!/usr/bin/env python3
"""Balanced full-request before/after native revision observations.

Includes media import/export, not just model components. Active MLX peaks
exclude allocator cache/non-MLX allocations; no physical budget is proved.
Output equivalence/quality must be checked separately.
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
    for name in ("model","source","before-library","after-library","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--kind",choices=("image","video"),required=True)
    parser.add_argument("--size",type=int,choices=(512,1024),required=True)
    parser.add_argument("--frames",type=int,default=1)
    parser.add_argument("--fps",type=int,default=24)
    parser.add_argument("--count",type=int,default=4)
    parser.add_argument("--residency",choices=("resident","component_staged"),default="resident")
    parser.add_argument("--manifest",type=Path)
    args = parser.parse_args()
    if not 4 <= args.count <= 10 or (args.kind == "image" and args.frames != 1):
        parser.error("one cold and at least three genuine warm requests per cohort required")
    for name in ("model","source","before_library","after_library"):
        setattr(args,name,getattr(args,name).resolve(strict=True))
    if args.manifest: args.manifest = args.manifest.resolve(strict=True)
    args.output = args.output.absolute()
    if args.output.exists(): parser.error("fresh revision cohort directory required")
    libraries = dict(before=args.before_library,after=args.after_library)
    paths = {"source":args.source}
    for label, library in libraries.items():
        paths[label+"_library"] = library
        paths[label+"_build"] = library.parent / "runtime-build/runtime-build-manifest.json"
    if args.manifest: paths["manifest"] = args.manifest
    identities = {key:sha256(path) for key,path in paths.items()}
    args.output.mkdir(parents=True)
    route = "native-hybrid" if args.manifest else "native-gpu"
    order = ["before","after","after","before"]
    report = dict(schema="seedvr2-native-revision-request-abba-v1",status="running",order=order,
        identities=identities,cohorts=[],route=route,
        workload=dict(kind=args.kind,size=args.size,frames=args.frames,fps=args.fps,residency=args.residency),
        scope="matched full-request warm timing/active MLX peak; separate output quality required; no hard physical budget")
    environment = {key:value for key,value in os.environ.items() if not key.startswith("TURBOCIDER_SEEDVR2_")}
    environment["TURBOCIDER_SEEDVR2_ALLOCATOR_CACHE_BYTES"] = str(4 << 30)
    try:
        model_identity = None
        for index, label in enumerate(order):
            if identities != {key:sha256(path) for key,path in paths.items()}:
                raise ValueError("source/library/build/bank identity changed")
            destination = args.output / f"{index}-{label}"
            command = [sys.executable,str(Path(__file__).with_name("seedvr2_framework_screen.py")),
                "--route",route,"--model",str(args.model),"--source",str(args.source),
                "--library",str(libraries[label]),"--kind",args.kind,"--width",str(args.size),
                "--height",str(args.size),"--frames",str(args.frames),"--fps",str(args.fps),
                "--seed","42","--count",str(args.count),"--residency",args.residency,"--output",str(destination)]
            if args.manifest: command += ["--ane-manifest",str(args.manifest)]
            write_json(args.output / f"command-{index}.json",dict(argv=command,allocator_cache_bytes=4 << 30))
            print(f"revision cohort {index+1}/4: {label}",flush=True)
            subprocess.run(command,env=environment,check=True)
            observed = json.loads((destination / "report.json").read_text())
            if (observed.get("status") != "complete" or len(observed.get("runs",[])) != args.count or
                observed.get("source_sha256") != identities["source"] or
                observed["native_artifact_identity"]["library_sha256"] != identities[label+"_library"] or
                observed["native_artifact_identity"]["runtime_build_manifest_sha256"] != identities[label+"_build"]):
                raise ValueError("incomplete or identity-mismatched revision observer")
            current = observed["model_provenance"]
            if model_identity is not None and current != model_identity:
                raise ValueError("original model provenance changed")
            model_identity = current
            warm = observed["runs"][1:]
            times = [row["request_wall_including_export"] for row in warm]
            peaks = [row["metrics"]["mlx_peak_bytes"] for row in warm]
            if any(not math.isfinite(value) or value <= 0 for value in times+peaks):
                raise ValueError("invalid warm timing/active peak")
            report["cohorts"].append(dict(index=index,revision=label,report=str(destination / "report.json"),
                cold_seconds=observed["runs"][0]["request_wall_including_export"],
                warm_seconds=times,warm_active_peak_bytes=peaks))
            write_json(args.output / "report.json",report)
        if identities != {key:sha256(path) for key,path in paths.items()}:
            raise ValueError("revision identities changed before finalization")
        for label in libraries:
            rows = [row for row in report["cohorts"] if row["revision"] == label]
            report[label+"_warm_median_seconds"] = statistics.median(t for row in rows for t in row["warm_seconds"])
            report[label+"_active_peak_bytes"] = max(t for row in rows for t in row["warm_active_peak_bytes"])
        report.update(status="complete",
            speed_gate_passed=report["after_warm_median_seconds"] <= report["before_warm_median_seconds"],
            speedup=report["before_warm_median_seconds"]/report["after_warm_median_seconds"],
            active_peak_reduction_bytes=report["before_active_peak_bytes"]-report["after_active_peak_bytes"])
        print(json.dumps({key:report[key] for key in ("status","before_warm_median_seconds",
            "after_warm_median_seconds","speedup","speed_gate_passed","active_peak_reduction_bytes")}),flush=True)
    except BaseException as error:
        report.update(status="failed",failure=str(error))
        raise
    finally:
        write_json(args.output / "report.json",report)


if __name__ == "__main__": main()
