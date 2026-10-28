#!/usr/bin/env python3
"""Exact component output screen for owned post-encoder input release.

Run only after the identity-bound preceding full-request benchmark ends.
This is not an App, paired warm-speed, physical-memory or ANE qualification.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from seedvr2_framework_screen import sha256, write_json


def completed_cohort(path, library_sha):
    report = json.loads(path.read_text())
    order = ["native-gpu","comfy-default","comfy-default","native-gpu"]
    rows = report.get("cohorts",[])
    if (report.get("status") != "complete" or report.get("speed_gate_passed") is not True or
        report.get("order") != order or report.get("library_sha256") != library_sha or
        len(rows) != 4 or [row.get("route") for row in rows] != order or
        [row.get("index") for row in rows] != list(range(4))):
        raise ValueError("preceding cohort is not a complete identity-bound speed gate")
    for row in rows:
        child = json.loads(Path(row["report"]).read_text())
        if child.get("status") != "complete" or len(child.get("runs",[])) < 4:
            raise ValueError("preceding cohort has unfinished child requests")
        if row["route"] == "comfy-default" and child.get("server_terminal") is not True:
            raise ValueError("preceding owned reference server is not terminal")
        if row["route"] == "native-gpu" and child.get("native_artifact_identity",{}).get("library_sha256") != library_sha:
            raise ValueError("preceding child has another native library")
    return report


def exact_tensors(reference, candidate):
    import numpy as np
    from safetensors.numpy import load_file
    left, right = load_file(str(reference)), load_file(str(candidate))
    expected = {"condition","velocity","latent","pixels"}
    if set(left) != expected or set(right) != expected:
        raise ValueError("restoration tensor set changed")
    rows = []
    for key in sorted(expected):
        x, y = left[key], right[key]
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError("restoration tensor geometry/precision changed")
        finite = exact = True
        # Keep temporary validation buffers bounded, even on t25 videos.
        a, b = x.reshape(-1), y.reshape(-1)
        for first in range(0,a.size,1 << 18):
            c, d = a[first:first+(1 << 18)], b[first:first+(1 << 18)]
            finite &= bool(np.isfinite(c).all() and np.isfinite(d).all())
            exact &= bool(np.array_equal(c.view(np.uint8),d.view(np.uint8)))
        rows.append(dict(tensor=key,shape=list(x.shape),dtype=str(x.dtype),finite=finite,bitwise_equal=exact))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probe","model","after-cohort","baseline-library","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--workload",nargs=2,action="append",metavar=("NAME","INPUT"),required=True)
    args = parser.parse_args()
    for name in ("probe","model","after_cohort","baseline_library"):
        setattr(args,name,getattr(args,name).resolve(strict=True))
    preceding = completed_cohort(args.after_cohort,sha256(args.baseline_library))
    names = [name for name,_ in args.workload]
    if len(names) != len(set(names)) or any(not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in name) for name in names):
        parser.error("unique simple workload names required")
    args.output = args.output.absolute()
    if args.output.exists(): parser.error("fresh owned-input screen directory required")
    import numpy
    from safetensors.numpy import load_file
    sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "coreml"))
    from export_ltx_gemma_mlp import read_header
    inputs = {name:Path(raw).resolve(strict=True) for name,raw in args.workload}
    identities = dict(probe=sha256(args.probe),preceding_cohort=sha256(args.after_cohort),
        baseline_library=sha256(args.baseline_library),inputs={name:sha256(path) for name,path in inputs.items()})
    provenance = json.loads((args.model / "source.json").read_text())
    qualified_model = json.loads(Path(preceding["cohorts"][0]["report"]).read_text())["model_provenance"]
    if provenance != qualified_model: raise ValueError("model provenance differs from qualified baseline")
    for item in provenance["files"]:
        path = args.model / item["path"]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError("original checkpoint changed before ownership screen")
    args.output.mkdir(parents=True)
    report = dict(schema="seedvr2-owned-input-component-screen-v1",status="running",identities=identities,
        workloads=[],preceding_speed_gate=preceding["speed_gate_passed"],model_provenance=provenance,
        scope="full component restoration exact tensors and one observed active peak per route; no App/warm/physical-memory claim")
    environment = {key:value for key,value in os.environ.items() if not key.startswith("TURBOCIDER_SEEDVR2_")}
    environment["TURBOCIDER_SEEDVR2_ALLOCATOR_CACHE_BYTES"] = str(4 << 30)
    try:
        for name, path in inputs.items():
            _, header = read_header(path)
            source = header.get("pixels",{})
            if source.get("dtype") != "F32": raise ValueError("FP32 source input required")
            logical_bytes = math.prod(source["shape"])*4
            observations = {}
            for command, label in (("restore","borrowed"),("restore-owned","owned")):
                destination = args.output / f"{name}-{label}"
                argv = [str(args.probe),command,str(args.model),str(path),str(destination),"fp16","no-dumps"]
                write_json(args.output / f"{name}-{label}-command.json",dict(argv=argv))
                print(f"owned input screen: {name} {label}",flush=True)
                with (args.output / f"{name}-{label}.log").open("w") as log:
                    subprocess.run(argv,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
                observed = json.loads((destination / "report.json").read_text())
                if (observed.get("input_release_recipe") != ("after_encode_owned" if label == "owned" else "borrowed") or
                    observed.get("released_input_logical_bytes") != (logical_bytes if label == "owned" else 0) or
                    observed.get("intermediates_dumped") is not False):
                    raise ValueError("probe did not execute its declared release/dump policy")
                observations[label] = observed
            rows = exact_tensors(args.output / f"{name}-borrowed/restoration.safetensors",
                                 args.output / f"{name}-owned/restoration.safetensors")
            passed = all(row["finite"] and row["bitwise_equal"] for row in rows)
            report["workloads"].append(dict(name=name,observations=observations,tensors=rows,exact_passed=passed))
            write_json(args.output / "report.json",report)
            if not passed: raise ValueError("owned restoration exact output gate failed")
            if identities["inputs"][name] != sha256(path) or identities["probe"] != sha256(args.probe):
                raise ValueError("input/probe changed during screen")
        if identities["preceding_cohort"] != sha256(args.after_cohort) or identities["baseline_library"] != sha256(args.baseline_library):
            raise ValueError("preceding qualification identity changed")
        for item in provenance["files"]:
            path = args.model / item["path"]
            if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                raise ValueError("original checkpoint changed during ownership screen")
        report.update(status="complete",all_exact_passed=True)
    except BaseException as error:
        report.update(status="failed",failure=str(error))
        raise
    finally:
        write_json(args.output / "report.json",report)


if __name__ == "__main__": main()
