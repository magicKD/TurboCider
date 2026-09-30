#!/usr/bin/env python3
"""Compare two runtime-LoRA schedules using matched resident whole-image runs.

The first request in each fresh CLI process is cold and excluded. Requests and
PNGs live in a temporary directory; the source request is never overwritten.
This is an exploratory screen, not an ANE-residency or image-quality proof.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, default=ROOT / "build/native/turbocider")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--flag", required=True,
                        help="experimental environment flag: unset versus 1")
    parser.add_argument("--warm-repeats", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.warm_repeats < 2 or args.timeout <= 0:
        parser.error("need at least two warm runs and a positive timeout")
    prototype = json.loads(args.request.read_text())
    if (prototype.get("execution") != "gpu_ane" or
            prototype.get("hybrid_mlp_mode") != "lora_fused" or
            prototype.get("lora_strategy") != "inference_time" or
            not prototype.get("loras") or
            prototype.get("residency") != "resident" or
            (prototype.get("width"), prototype.get("height")) != (512, 512)):
        parser.error("provide an explicit resident 512px runtime-LoRA fused-base request")
    model_id = prototype["model"]
    cases = {
        "qwen-image-2.1": (160, 227), "z-image-turbo": (256, 238)
    }
    if model_id not in cases or prototype.get("steps") != {
            "qwen-image-2.1": 6, "z-image-turbo": 8}.get(model_id):
        parser.error("screen only supports Qwen 6-step or Z-Image 8-step 512px requests")
    calls_per_request, expected_projections = cases[model_id]
    results = {"off": [], "on": []}
    image_hashes = {"off": set(), "on": set()}
    with tempfile.TemporaryDirectory(prefix="turbocider-lora-schedule-") as scratch:
        temporary = Path(scratch)
        for trial, state in enumerate(("off", "on", "on", "off")):
            requests = []
            outputs = []
            for index in range(args.warm_repeats + 1):
                request = dict(prototype)
                output = temporary / f"{trial}-{index}.png"
                request["output"] = str(output)
                target = temporary / f"{trial}-{index}.json"
                target.write_text(json.dumps(request))
                requests.append(target)
                outputs.append(output)
            env = os.environ.copy()
            env.pop(args.flag, None)
            if state == "on":
                env[args.flag] = "1"
            process = subprocess.run(
                [str(args.cli.resolve()), "batch", str(args.model.resolve()),
                 *(str(path) for path in requests)], cwd=ROOT, env=env,
                capture_output=True, text=True, timeout=args.timeout)
            if process.returncode != 0:
                raise RuntimeError(f"{state} trial {trial} failed: {process.stderr[-4000:]}")
            rows = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
            if len(rows) != len(requests):
                raise AssertionError("batch returned the wrong number of requests")
            for index, (row, output) in enumerate(zip(rows, outputs)):
                if (row.get("lora_applied_projections") != expected_projections or
                        row["hybrid"]["runtime_calls_session_total"] !=
                        (index + 1) * calls_per_request or
                        row["hybrid"]["runtime_failures_session_total"] != 0):
                    raise AssertionError("runtime adapter or Core ML coverage changed")
                digest = hashlib.sha256(output.read_bytes()).hexdigest()
                image_hashes[state].add(digest)
                if index:
                    results[state].append(row["timings_seconds"]["request_wall"])
            print(json.dumps({"trial": trial, "flag": state,
                              "warm_wall_seconds": results[state][-args.warm_repeats:]},
                             sort_keys=True), flush=True)
    if len(image_hashes["off"] | image_hashes["on"]) != 1:
        raise AssertionError("schedule changed output PNG; reject the timing comparison")
    print(json.dumps({"model": model_id, "flag": args.flag, "order": "off/on/on/off",
                      "warm_wall_seconds": results,
                      "median_seconds": {key: statistics.median(values)
                                         for key, values in results.items()},
                      "matching_png_sha256": next(iter(image_hashes["off"]))},
                     sort_keys=True))


if __name__ == "__main__":
    main()
