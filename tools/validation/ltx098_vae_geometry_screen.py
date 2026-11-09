#!/usr/bin/env python3
"""Byte-exact VAE candidate checks on small and non-square latent geometries."""

import argparse
import json
from pathlib import Path
import subprocess

from ltx098_comfy_reference import MODEL_SHA256, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "input", "baseline", "candidate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    checkpoint, source, baseline, candidate = (p.resolve(strict=True) for p in
                                                (args.checkpoint, args.input, args.baseline, args.candidate))
    output = args.output.resolve()
    if output.exists() or baseline == candidate:
        parser.error("new output directory and distinct binaries required")
    if sha256(checkpoint) != MODEL_SHA256:
        raise ValueError("checkpoint SHA-256 differs from pinned source")
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    import torch

    inputs = load_file(str(source))
    if set(inputs) != {"latents", "noise"} or any(
            x.dtype != torch.bfloat16 or tuple(x.shape) != (1, 2, 2, 2, 128) for x in inputs.values()):
        raise ValueError("expected the saved BF16 two-frame 2x2 latent fixture")
    identities = {"checkpoint_sha256": MODEL_SHA256, "source_sha256": sha256(source),
                  "baseline_sha256": sha256(baseline), "candidate_sha256": sha256(candidate)}
    output.mkdir(parents=True)
    report = {"schema": "ltx098-vae-geometry-screen-v1", "status": "running", **identities,
              "scope": "Full GPU VAE decode of saved synthetic inputs, matching BF16 pixel files; not timing or quality against Comfy",
              "cases": []}
    try:
        for name, transform, expected in (
                ("one-frame-64-square", lambda x: x[:, :1].contiguous(), (1, 1, 64, 64, 3)),
                ("nine-frame-64-square", lambda x: x.contiguous(), (1, 9, 64, 64, 3)),
                ("nine-frame-64x96", lambda x: torch.cat((x, x[:, :, :, :1]), dim=3).contiguous(), (1, 9, 64, 96, 3)),
                ("full-length-128x1024", lambda x: x.repeat(1, 8, 2, 16, 1).contiguous(), (1, 121, 128, 1024, 3)),
                ("full-length-1024x128", lambda x: x.repeat(1, 8, 16, 2, 1).contiguous(), (1, 121, 1024, 128, 3))):
            fixture = output / f"{name}-input.safetensors"
            save_file({key: transform(value) for key, value in inputs.items()}, str(fixture))
            hashes = {}
            for mode, executable in (("baseline", baseline), ("candidate", candidate)):
                path = output / f"{name}-{mode}.safetensors"
                run = subprocess.run([str(executable), str(checkpoint), str(fixture), str(path)],
                                     capture_output=True, text=True, timeout=600)
                (output / f"{name}-{mode}.stdout.txt").write_text(run.stdout)
                (output / f"{name}-{mode}.stderr.txt").write_text(run.stderr)
                if run.returncode:
                    raise RuntimeError(f"{name} {mode} failed: {run.stderr[-4000:]}")
                with safe_open(str(path), framework="pt", device="cpu") as result:
                    if list(result.keys()) != ["pixels"] or tuple(result.get_slice("pixels").get_shape()) != expected:
                        raise ValueError(f"{name} {mode} output geometry mismatch")
                hashes[mode] = sha256(path)
            if hashes["baseline"] != hashes["candidate"]:
                raise ValueError(f"{name} decoded BF16 pixels differ")
            row = {"case": name, "shape": expected, "fixture_sha256": sha256(fixture),
                   "baseline_sha256": hashes["baseline"], "candidate_sha256": hashes["candidate"], "identical": True}
            report["cases"].append(row)
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
        if (sha256(checkpoint) != identities["checkpoint_sha256"] or sha256(source) != identities["source_sha256"] or
                sha256(baseline) != identities["baseline_sha256"] or sha256(candidate) != identities["candidate_sha256"]):
            raise ValueError("source or binary identity changed during measurement")
        report.update(status="complete", all_identical=True)
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
