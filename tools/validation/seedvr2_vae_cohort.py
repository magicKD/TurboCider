#!/usr/bin/env python3
"""Balanced original/optimized/original VAE component observations (ABBA).

Every iteration encodes and decodes the same pixels anew; only weights,
allocator buffers and compiled kernels stay warm. This is not a complete
media request, an independent Comfy quality witness or a hard memory budget.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = {
    "TURBOCIDER_SEEDVR2_VAE_NORM_RECIPE": "metal",
    "TURBOCIDER_SEEDVR2_VAE_CONV_TILE_ROWS": "16384",
    "TURBOCIDER_SEEDVR2_ALLOCATOR_CACHE_BYTES": str(4 << 30),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parity(reference, candidate):
    import numpy as np
    from safetensors.numpy import load_file
    expected, actual = load_file(str(reference)), load_file(str(candidate))
    if set(expected) != {"latent", "decoded"} or set(actual) != set(expected):
        raise ValueError("VAE tensor set changed")
    result = []
    for key in sorted(expected):
        if expected[key].shape != actual[key].shape:
            raise ValueError("VAE output shape changed")
        x, y = expected[key].astype(np.float64).ravel(), actual[key].astype(np.float64).ravel()
        finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
        if not finite:
            result.append(dict(tensor=key,finite=False,relative_l2=None,cosine=None,
                bitwise_equal=False,passed=False))
            continue
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        l2 = float(np.linalg.norm(x-y) / max(nx, 1e-20))
        cosine = float(np.dot(x,y) / max(nx*ny, 1e-20))
        result.append(dict(tensor=key, finite=finite, relative_l2=l2, cosine=cosine,
            bitwise_equal=expected[key].dtype == actual[key].dtype and expected[key].tobytes() == actual[key].tobytes(),
            passed=finite and l2 <= .05 and cosine >= .999))
    return result


def validate_recipes(raw,settings):
    for field in ("conv_recipe","norm_recipe","history_recipe","pointwise_recipe"):
        if raw[field] != settings["TURBOCIDER_SEEDVR2_VAE_"+field.upper()]:
            raise ValueError("VAE observation did not execute its declared recipe")
    # Older completed observers predate this selector and necessarily use
    # native upsampling. They can never qualify a metal_fused candidate.
    expected=settings.get("TURBOCIDER_SEEDVR2_VAE_UPSCALE_RECIPE","native")
    if raw.get("upscale_recipe","native")!=expected:
        raise ValueError("VAE observation did not execute its declared upscale recipe")


def run(args):
    # Fail before any expensive GPU work if this validation interpreter lacks
    # the independent host-side tensor reader.
    import numpy
    from safetensors.numpy import load_file
    for name in ("probe", "model", "input"):
        setattr(args, name, getattr(args,name).resolve(strict=True))
    if args.finalize_only:
        return finalize_existing(args)
    if args.output.exists():
        raise ValueError("a fresh VAE cohort directory is required")
    args.output.mkdir(parents=True)
    weight = args.model / "vae/seedvr2_ema_vae_fp16.safetensors"
    identities = {"probe": sha256(args.probe), "input": sha256(args.input), "vae": sha256(weight)}
    if identities["vae"] != "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1":
        raise ValueError("original SeedVR2 VAE identity required")
    baseline = {**SETTINGS, "TURBOCIDER_SEEDVR2_VAE_CONV_RECIPE": "singleton2d",
        "TURBOCIDER_SEEDVR2_VAE_HISTORY_RECIPE": getattr(args,"baseline_history","concat"),
        "TURBOCIDER_SEEDVR2_VAE_POINTWISE_RECIPE": getattr(args,"baseline_pointwise","conv"),
        "TURBOCIDER_SEEDVR2_VAE_UPSCALE_RECIPE":getattr(args,"baseline_upscale","native")}
    candidate = {**SETTINGS, "TURBOCIDER_SEEDVR2_VAE_CONV_RECIPE": args.candidate,
        "TURBOCIDER_SEEDVR2_VAE_HISTORY_RECIPE": getattr(args,"candidate_history","compact"),
        "TURBOCIDER_SEEDVR2_VAE_POINTWISE_RECIPE": "matmul",
        "TURBOCIDER_SEEDVR2_VAE_UPSCALE_RECIPE":getattr(args,"candidate_upscale","native")}
    order = ["baseline","candidate","candidate","baseline"]
    report = dict(schema="seedvr2-vae-paired-cohort-v1", status="running", identities=identities,
        order=order, settings=dict(baseline=baseline,candidate=candidate), cohorts=[],
        scope="VAE encode+decode only; input import/export excluded; no full-model/Comfy/ANE qualification",
        peak_scope="active MLX allocation peak; not whole-process/system usage or an enforced budget")
    path = args.output / "report.json"
    write_json(path,report)
    try:
        for index, route in enumerate(order):
            output = args.output / f"{index}-{route}"
            settings = baseline if route == "baseline" else candidate
            environment = {key:value for key,value in os.environ.items()
                if not key.startswith("TURBOCIDER_SEEDVR2_")}
            environment.update(settings)
            command = [str(args.probe), "vae",str(args.model),str(args.input),str(output),str(args.repeats)]
            write_json(args.output/f"command-{index}.json",dict(command=command,settings=settings))
            print(f"running {index}: {route}",flush=True)
            with (args.output/f"stderr-{index}.txt").open("w") as stderr:
                subprocess.run(command,env=environment,stdout=subprocess.PIPE,stderr=stderr,check=True,text=True)
            raw = json.loads((output/"report.json").read_text())
            validate_recipes(raw,settings)
            iterations = raw["iterations"]
            if len(iterations) != args.repeats or any(row["iteration"] != n for n,row in enumerate(iterations)):
                raise ValueError("incomplete VAE repetitions")
            warm = iterations[1:]
            report["cohorts"].append(dict(index=index,route=route,report=str(output/"report.json"),
                cold_seconds=iterations[0]["encode_seconds"]+iterations[0]["decode_seconds"],
                warm_seconds=[row["encode_seconds"]+row["decode_seconds"] for row in warm],
                warm_peak_bytes=[max(row["encode_peak_bytes"],row["decode_peak_bytes"]) for row in warm]))
            write_json(path,report)
        if identities != {"probe":sha256(args.probe),"input":sha256(args.input),"vae":sha256(weight)}:
            raise ValueError("VAE cohort identity changed")
        reference = args.output/"0-baseline/vae.safetensors"
        report["parity"] = {f"{i}-{route}":parity(reference,args.output/f"{i}-{route}/vae.safetensors")
            for i,route in enumerate(order[1:],1)}
        report["parity_passed"] = all(row["passed"] for rows in report["parity"].values() for row in rows)
        for route in ("baseline","candidate"):
            rows = [row for row in report["cohorts"] if row["route"] == route]
            report[f"{route}_warm_median_seconds"] = statistics.median(t for row in rows for t in row["warm_seconds"])
            report[f"{route}_active_peak_bytes"] = max(t for row in rows for t in row["warm_peak_bytes"])
        report["speedup"] = report["baseline_warm_median_seconds"]/report["candidate_warm_median_seconds"]
        report["speed_gate_passed"] = report["candidate_warm_median_seconds"] <= report["baseline_warm_median_seconds"]
        report["bitwise_equal"] = all(row["bitwise_equal"] for rows in report["parity"].values() for row in rows)
        report["active_peak_reduction_bytes"] = report["baseline_active_peak_bytes"]-report["candidate_active_peak_bytes"]
        report["status"] = "complete"
        write_json(path,report)
        if not report["parity_passed"]:
            raise ValueError("VAE numerical gate failed")
        print(json.dumps(report),flush=True)
    except BaseException as error:
        report.update(status="failed",error=str(error))
        write_json(path,report)
        raise


def finalize_existing(args):
    """Recover only completed GPU observations from a failed host finalizer."""
    path = args.output/"report.json"
    report = json.loads(path.read_text())
    if report["status"] != "failed" or report["error"] != "No module named 'safetensors'":
        raise ValueError("only the completed-observer/missing-reader failure can be finalized")
    identities = dict(probe=sha256(args.probe),input=sha256(args.input),
        vae=sha256(args.model/"vae/seedvr2_ema_vae_fp16.safetensors"))
    if report["identities"] != identities or report["order"] != ["baseline","candidate","candidate","baseline"]:
        raise ValueError("completed cohort identity/order changed")
    if len(report["cohorts"]) != 4:
        raise ValueError("incomplete GPU observations cannot be finalized")
    for index,route in enumerate(report["order"]):
        observed = report["cohorts"][index]
        output = args.output/f"{index}-{route}"
        raw = json.loads((output/"report.json").read_text())
        if observed["index"] != index or observed["route"] != route or len(raw["iterations"]) != args.repeats:
            raise ValueError("cohort count/order changed")
        validate_recipes(raw,report["settings"][route])
        iterations = raw["iterations"]
        if any(row["iteration"] != n for n,row in enumerate(iterations)) or observed["warm_seconds"] != [
                row["encode_seconds"]+row["decode_seconds"] for row in iterations[1:]] or observed["warm_peak_bytes"] != [
                max(row["encode_peak_bytes"],row["decode_peak_bytes"]) for row in iterations[1:]]:
            raise ValueError("completed raw observations changed")
    reference = args.output/"0-baseline/vae.safetensors"
    report["parity"] = {f"{i}-{route}":parity(reference,args.output/f"{i}-{route}/vae.safetensors")
        for i,route in enumerate(report["order"][1:],1)}
    report["parity_passed"] = all(row["passed"] for rows in report["parity"].values() for row in rows)
    for route in ("baseline","candidate"):
        rows = [row for row in report["cohorts"] if row["route"] == route]
        report[f"{route}_warm_median_seconds"] = statistics.median(t for row in rows for t in row["warm_seconds"])
        report[f"{route}_active_peak_bytes"] = max(t for row in rows for t in row["warm_peak_bytes"])
    report["speedup"] = report["baseline_warm_median_seconds"]/report["candidate_warm_median_seconds"]
    report["speed_gate_passed"] = report["candidate_warm_median_seconds"] <= report["baseline_warm_median_seconds"]
    report["bitwise_equal"] = all(row["bitwise_equal"] for rows in report["parity"].values() for row in rows)
    report["active_peak_reduction_bytes"] = report["baseline_active_peak_bytes"]-report["candidate_active_peak_bytes"]
    failed_record = args.output/"host-finalization-failure.json"
    if failed_record.exists():
        raise ValueError("prior finalization-failure record already exists")
    write_json(failed_record,json.loads(path.read_text()))
    report.update(status="complete",finalization_interpreter=sys.executable,
        recovered_failure="completed GPU cohorts; original host interpreter lacked safetensors; no GPU observations rerun")
    del report["error"]
    write_json(path,report)
    if not report["parity_passed"]:
        raise ValueError("VAE numerical gate failed")
    print(json.dumps(report),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe",type=Path,default=ROOT/"build/seedvr2/seedvr2-probe")
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--candidate",choices=("singleton2d","mixed3d"),default="singleton2d")
    parser.add_argument("--baseline-history",choices=("concat","compact","frames"),default="concat")
    parser.add_argument("--candidate-history",choices=("concat","compact","frames"),default="compact")
    parser.add_argument("--baseline-pointwise",choices=("conv","matmul"),default="conv")
    parser.add_argument("--baseline-upscale",choices=("native","metal_fused"),default="native")
    parser.add_argument("--candidate-upscale",choices=("native","metal_fused"),default="native")
    parser.add_argument("--repeats",type=int,default=4)
    parser.add_argument("--finalize-only",action="store_true",help="verify/recover completed GPU cohorts after a missing-reader host failure")
    args = parser.parse_args()
    if not 4 <= args.repeats <= 20:
        parser.error("at least one cold and three warm repeats per cohort are required")
    run(args)


if __name__ == "__main__":
    main()
