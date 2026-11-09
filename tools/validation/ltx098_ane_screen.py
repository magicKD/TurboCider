#!/usr/bin/env python3
"""Screen a runtime-weight Core ML GELU graph with real LTX 0.9.8 FFN weights.

This intentionally omits both biases and uses synthetic activations. It is a
component feasibility test, not a model-quality, end-to-end speed, or proof of
physical ANE residency. CPU_AND_NE can fall back; CPU_ONLY is a control.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from ltx098_comfy_reference import MODEL_SHA256


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", type=int, choices=range(28), default=0)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--ane-chunks", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--compute", choices=("ne", "cpu"), default="ne")
    args = parser.parse_args()
    checkpoint, manifest, probe = (path.resolve(strict=True) for path in (args.checkpoint, args.manifest, args.probe))
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory must not exist")
    if sha256(checkpoint) != MODEL_SHA256:
        raise ValueError("checkpoint SHA-256 differs from the pinned LTX 0.9.8 source")
    spec = json.loads(manifest.read_text())
    if (spec.get("kind"), spec.get("hidden"), spec.get("width"), spec.get("biases")) != ("gelu", 2048, 8192, False) or not isinstance(spec.get("rows"), int) or spec["rows"] < 1:
        raise ValueError("expected H2048/F8192, bias-free GELU runtime graph with positive rows")
    from safetensors import safe_open
    from safetensors.torch import save_file
    prefix = f"model.diffusion_model.transformer_blocks.{args.block}.ff.net."
    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        up = source.get_tensor(prefix + "0.proj.weight")
        down = source.get_tensor(prefix + "2.weight")
    if tuple(up.shape) != (8192, 2048) or tuple(down.shape) != (2048, 8192):
        raise ValueError("checkpoint FFN geometry mismatch")
    output.mkdir(parents=True)
    bundle = output / "weights.safetensors"
    save_file({"wu": up.contiguous(), "wd": down.contiguous()}, str(bundle))
    command = [str(probe), str(manifest), str(args.repeats), str(args.rows), args.compute, str(bundle),
               str(args.ane_chunks), "simd", "-", str(args.block)]
    report = {"schema": "ltx098-ane-component-screen-v1", "status": "running", "checkpoint_sha256": MODEL_SHA256,
              "manifest_sha256": sha256(manifest), "bundle_sha256": sha256(bundle), "command": command,
              "compute_units": "CPU_AND_NE" if args.compute == "ne" else "CPU_ONLY",
              "scope": "Real BF16 FFN matrices, synthetic activations, no biases; FP16 runtime-weight Core ML vs MLX GPU. Not end-to-end or proof of physical ANE placement."}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    started = time.perf_counter()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=1200)
        (output / "stdout.txt").write_text(result.stdout)
        (output / "stderr.txt").write_text(result.stderr)
        report.update(returncode=result.returncode, elapsed_seconds=time.perf_counter() - started,
                      status="complete" if result.returncode == 0 else "failed")
        if result.returncode == 0:
            report["measurements"] = json.loads(result.stdout)
        else:
            report["error"] = result.stderr[-4000:]
    except BaseException as error:
        report.update(status="failed", error=str(error), elapsed_seconds=time.perf_counter() - started)
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
