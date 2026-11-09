#!/usr/bin/env python3
"""Repeat the best Core ML row split across real LTX FFN blocks.

Each run retains the strict runtime self-test. Synthetic activations, omitted
biases, and unknown physical ANE placement preclude end-to-end claims.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess

from ltx098_comfy_reference import MODEL_SHA256, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    checkpoint, manifest, probe = (p.resolve(strict=True) for p in (args.checkpoint, args.manifest, args.probe))
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory must not already exist")
    if sha256(checkpoint) != MODEL_SHA256:
        raise ValueError("checkpoint SHA-256 differs from the pinned source")
    spec = json.loads(manifest.read_text())
    if (spec.get("kind"), spec.get("hidden"), spec.get("width"), spec.get("biases")) != ("gelu", 2048, 8192, False):
        raise ValueError("incompatible runtime graph")
    from safetensors import safe_open
    from safetensors.torch import save_file
    output.mkdir(parents=True)
    report = {"schema": "ltx098-ane-blocks-screen-v1", "status": "running", "checkpoint_sha256": MODEL_SHA256,
              "manifest_sha256": sha256(manifest), "probe_sha256": sha256(probe),
              "scope": "CPU_AND_NE scheduling vs MLX GPU; real block 0/14/27 matrices, synthetic inputs, no biases; no placement or end-to-end claim",
              "rows": args.rows, "chunks": args.chunks, "runs": []}
    try:
        bundles = {}
        with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
            for block in (0, 14, 27):
                prefix = f"model.diffusion_model.transformer_blocks.{block}.ff.net."
                bundle = output / f"block-{block}.safetensors"
                save_file({"wu": source.get_tensor(prefix + "0.proj.weight").contiguous(),
                           "wd": source.get_tensor(prefix + "2.weight").contiguous()}, str(bundle))
                bundles[block] = bundle
        for block in (0, 14, 27, 27, 14, 0):
            command = [str(probe), str(manifest), str(args.repeats), str(args.rows), "ne", str(bundles[block]),
                       str(args.chunks), "simd", "-", str(block)]
            result = subprocess.run(command, capture_output=True, text=True, timeout=1200)
            if result.returncode:
                raise RuntimeError(f"block {block}: {result.stderr[-4000:]}")
            value = json.loads(result.stdout)
            row = {"block": block, "bundle_sha256": sha256(bundles[block]), "gpu_seconds": value["gpu_seconds"],
                   "split_with_stage_seconds": value["split_with_stage_seconds"], "speedup_with_stage": value["speedup_with_stage"],
                   "relative_l2": value["relative_l2"], "max_abs": value["max_abs"],
                   "observed_ane_residency": value["observed_ane_residency"], "samples": value["samples"]}
            report["runs"].append(row)
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({key: item for key, item in row.items() if key != "samples"}), flush=True)
        report["median_component_speedup"] = statistics.median(row["speedup_with_stage"] for row in report["runs"])
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
