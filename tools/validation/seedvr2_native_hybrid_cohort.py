#!/usr/bin/env python3
"""Identity-matched GPU/CPU+NE complete-request ABBA warm speed evidence.

Scope is one workload and residency, not physical ANE occupancy, universal
acceleration, a hard memory budget or ComfyUI speed comparison.
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
    for name in ("model", "source", "library", "manifest", "output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--kind",choices=("image","video"),required=True)
    parser.add_argument("--size",type=int,choices=(512,1024),required=True)
    parser.add_argument("--frames",type=int,default=1)
    parser.add_argument("--fps",type=int,default=24)
    parser.add_argument("--count",type=int,default=4)
    parser.add_argument("--residency",choices=("resident","component_staged"),default="resident")
    args = parser.parse_args()
    if not 4 <= args.count <= 10 or (args.kind == "image" and args.frames != 1):
        parser.error("one cold plus at least three warm requests per cohort required")
    for name in ("model","source","library","manifest"):
        setattr(args,name,getattr(args,name).resolve(strict=True))
    args.output = args.output.absolute()
    if args.output.exists(): parser.error("a fresh complete-request cohort directory is required")
    build = args.library.parent / "runtime-build/runtime-build-manifest.json"
    identities = dict(source=sha256(args.source),library=sha256(args.library),manifest=sha256(args.manifest),build=sha256(build))
    args.output.mkdir(parents=True)
    order = ["native-gpu","native-hybrid","native-hybrid","native-gpu"]
    record = dict(schema="seedvr2-native-hybrid-request-abba-v1",status="running",order=order,
        identities=identities,workload=dict(kind=args.kind,size=args.size,frames=args.frames,fps=args.fps,residency=args.residency),
        cohorts=[],speed_gate_passed=False,overall_qualification_passed=False,
        scope="matched complete-request warm wall time including media and export; no physical ANE placement/budget claim")
    environment = {key:value for key,value in os.environ.items() if not key.startswith("TURBOCIDER_SEEDVR2_")}
    environment["TURBOCIDER_SEEDVR2_ALLOCATOR_CACHE_BYTES"] = str(4 << 30)
    try:
        provenance = None
        for index, route in enumerate(order):
            for key,path in (("source",args.source),("library",args.library),("manifest",args.manifest),("build",build)):
                if sha256(path) != identities[key]: raise ValueError("hybrid cohort identities changed")
            output = args.output / f"{index}-{route}"
            command = [sys.executable,str(Path(__file__).with_name("seedvr2_framework_screen.py")),
                "--route",route,"--model",str(args.model),"--source",str(args.source),"--library",str(args.library),
                "--kind",args.kind,"--width",str(args.size),"--height",str(args.size),"--frames",str(args.frames),
                "--fps",str(args.fps),"--seed","42","--count",str(args.count),"--residency",args.residency,"--output",str(output)]
            if route == "native-hybrid": command += ["--ane-manifest",str(args.manifest)]
            write_json(args.output / f"command-{index}.json",dict(argv=command,allocator_cache_bytes=4 << 30))
            print(f"native/hybrid ABBA {index+1}/4: {route}",flush=True)
            subprocess.run(command,env=environment,check=True)
            observed = json.loads((output / "report.json").read_text())
            if observed.get("status") != "complete" or len(observed.get("runs",[])) != args.count:
                raise ValueError("incomplete complete-request hybrid cohort")
            if observed.get("source_sha256") != identities["source"] or observed["native_artifact_identity"]["library_sha256"] != identities["library"]:
                raise ValueError("observer source/library identity mismatch")
            if provenance is not None and observed["model_provenance"] != provenance:
                raise ValueError("original models changed")
            provenance = observed["model_provenance"]
            warm = [row["request_wall_including_export"] for row in observed["runs"][1:]]
            if any(not math.isfinite(v) or v <= 0 for v in warm): raise ValueError("invalid warm timings")
            record["cohorts"].append(dict(index=index,route=route,report=str(output / "report.json"),
                cold_seconds=observed["runs"][0]["request_wall_including_export"],warm_seconds=warm,
                active_peak_bytes=max(row["metrics"]["mlx_peak_bytes"] for row in observed["runs"][1:])))
            write_json(args.output / "report.json",record)
        for key,path in (("source",args.source),("library",args.library),("manifest",args.manifest),("build",build)):
            if sha256(path) != identities[key]: raise ValueError("identity changed before finalizing")
        for route,label in (("native-gpu","gpu"),("native-hybrid","hybrid")):
            values = [v for row in record["cohorts"] if row["route"] == route for v in row["warm_seconds"]]
            record[label+"_warm_median_seconds"] = statistics.median(values)
        record.update(status="complete",speed_gate_passed=record["hybrid_warm_median_seconds"] <= record["gpu_warm_median_seconds"],
                      speedup=record["gpu_warm_median_seconds"]/record["hybrid_warm_median_seconds"])
        print(json.dumps({key:record[key] for key in ("status","gpu_warm_median_seconds","hybrid_warm_median_seconds","speed_gate_passed","speedup")}),flush=True)
    except BaseException as error:
        record.update(status="failed",failure=str(error))
        raise
    finally:
        write_json(args.output / "report.json",record)


if __name__ == "__main__": main()
