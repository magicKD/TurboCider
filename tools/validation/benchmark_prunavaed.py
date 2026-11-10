#!/usr/bin/env python3
"""Time Conv VAE and PrunaVAED on identical normalized BF16 LTX latents."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--conv", type=Path, required=True)
    parser.add_argument("--pruna", type=Path, required=True)
    parser.add_argument("--latent", type=Path, required=True)
    parser.add_argument("--shape", type=int, nargs=3, required=True, metavar=("F", "H", "W"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("at least two measured samples required")
    args.output.mkdir(parents=True, exist_ok=True)
    variants = ["conv", "pruna_unfused", "pruna_fused"]
    rows = []
    report = dict(method="One warm-up per variant, then alternating order; new process per decode; decode compute excludes loading and output IO",
                  shape=args.shape, latent_sha256=hashlib.sha256(args.latent.read_bytes()).hexdigest(),
                  binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                  samples=args.samples, runs=rows, complete=False)

    def save():
        (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    for iteration in range(-1, args.samples):
        order = variants if iteration % 2 == 0 else list(reversed(variants))
        for variant in order:
            env = os.environ.copy()
            for key in list(env):
                if key.startswith("TURBOCIDER_LTX_VAE_") or key == "TURBOCIDER_LTX_VIDEO_VAE_CHECKPOINT_FD":
                    env.pop(key)
            env["TURBOCIDER_PRUNAVAED_FUSED_SILU"] = "1" if variant == "pruna_fused" else "0"
            timing = args.output / f"{variant}-{iteration}.json"
            checkpoint = args.conv if variant == "conv" else args.pruna
            subprocess.run([str(args.binary.resolve()), str(checkpoint), str(args.latent),
                            *map(str, args.shape), str(args.output / f"{variant}.bf16"),
                            str(timing)], env=env, check=True)
            row = dict(variant=variant, warmup=iteration < 0, **json.loads(timing.read_text()))
            if row["ltx_video_vae"] != ("conv" if variant == "conv" else "prunavaed"):
                raise ValueError("decoder selection mismatch")
            rows.append(row)
            save()
            print(json.dumps(row), flush=True)
    medians = {variant: statistics.median(row["decode_compute_seconds"] for row in rows
                                        if row["variant"] == variant and not row["warmup"])
               for variant in variants}
    report.update(median_decode_seconds=medians,
                  fused_speedup=medians["pruna_unfused"] / medians["pruna_fused"],
                  speedup_vs_conv=medians["conv"] / medians["pruna_fused"], complete=True)
    save()
    print(json.dumps({key: value for key, value in report.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
