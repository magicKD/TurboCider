#!/usr/bin/env python3
"""Screen runtime-weight Core ML row splits with real LTX FFN matrices.

Synthetic activations and omitted biases; CPU_AND_NE does not prove physical
ANE execution. Measurements are component-only and are not end-to-end claims.
"""
import argparse
import json
from pathlib import Path
import subprocess

from ltx098_comfy_reference import MODEL_SHA256, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--chunks", default="1,2,4,6,8,10,12,15")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    checkpoint, manifest, probe, bundle = (p.resolve(strict=True) for p in
        (args.checkpoint, args.manifest, args.probe, args.bundle))
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory must not already exist")
    if sha256(checkpoint) != MODEL_SHA256:
        raise ValueError("checkpoint SHA-256 differs from the pinned source")
    spec = json.loads(manifest.read_text())
    chunk_rows = spec.get("rows")
    if (spec.get("kind"), spec.get("hidden"), spec.get("width"), spec.get("biases")) != ("gelu", 2048, 8192, False) or not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise ValueError("incompatible runtime graph")
    chunks = [int(value) for value in args.chunks.split(",")]
    if not chunks or len(set(chunks)) != len(chunks) or any(value < 1 or value * chunk_rows >= args.rows for value in chunks):
        parser.error("chunks must be unique positive integers leaving some GPU rows")
    from safetensors import safe_open
    from safetensors.torch import load_file
    import torch
    values = load_file(str(bundle))
    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        for key, suffix in (("wu", "0.proj.weight"), ("wd", "2.weight")):
            name = "model.diffusion_model.transformer_blocks.0.ff.net." + suffix
            if key not in values or not torch.equal(values[key], source.get_tensor(name)):
                raise ValueError(f"bundle tensor {key} does not equal checkpoint block 0")
    del values
    output.mkdir(parents=True)
    report = {"schema": "ltx098-ane-chunk-sweep-v1", "status": "running", "checkpoint_sha256": MODEL_SHA256,
              "manifest_sha256": sha256(manifest), "bundle_sha256": sha256(bundle),
              "scope": "CPU_AND_NE scheduling vs MLX GPU; real block-0 matrices, synthetic inputs, no biases; no placement or end-to-end claim",
              "rows": args.rows, "runs": []}
    try:
        for count in chunks:
            command = [str(probe), str(manifest), str(args.repeats), str(args.rows), "ne", str(bundle), str(count), "simd", "-", "0"]
            run = subprocess.run(command, capture_output=True, text=True, timeout=1200)
            if run.returncode:
                row = {"chunks": count, "status": "failed", "error": run.stderr[-4000:]}
            else:
                measurements = json.loads(run.stdout)
                row = {"chunks": count, "status": "complete", "ane_rows": measurements["ane_rows"],
                       "gpu_rows": measurements["gpu_rows"], "gpu_seconds": measurements["gpu_seconds"],
                       "split_with_stage_seconds": measurements["split_with_stage_seconds"],
                       "speedup_with_stage": measurements["speedup_with_stage"],
                       "relative_l2": measurements["relative_l2"], "max_abs": measurements["max_abs"],
                       "observed_ane_residency": measurements["observed_ane_residency"],
                       "samples": measurements["samples"]}
            report["runs"].append(row)
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
        report["status"] = "complete" if all(row["status"] == "complete" for row in report["runs"]) else "some_failed"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
