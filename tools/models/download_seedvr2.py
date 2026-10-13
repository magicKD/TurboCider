#!/usr/bin/env python3
"""Download and verify the original Comfy-Org SeedVR2 FP16 files from ModelScope.

Partial files are resumable. Nothing is published at the final filename until
both its byte count and the ModelScope SHA-256 match. Proxy settings are passed
to curl through the environment; this tool never reads shell configuration.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


REPOSITORY = "Comfy-Org/SeedVR2"
REVISION = "8d30a39cd60f07cfff11306406f97021a50ff0eb"
FILES = (
    ("diffusion_models/seedvr2_3b_fp16.safetensors", 6784268336,
     "98669fd2c06df5eca88baf68cd5c478775c8e61fc110e598c52b350145ea2660"),
    ("vae/seedvr2_ema_vae_fp16.safetensors", 501324814,
     "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1"),
)


def verify(path, size, sha256):
    if path.stat().st_size != size:
        raise ValueError(f"wrong size: {path} ({path.stat().st_size}, expected {size})")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != sha256:
        raise ValueError(f"SHA-256 mismatch: {path}: {actual}")
    return actual


def download(root, verify_only=False):
    records = []
    for relative, size, sha256 in FILES:
        target = root / relative
        partial = target.with_suffix(target.suffix + ".part")
        if target.exists():
            verify(target, size, sha256)
        else:
            if not verify_only:
                target.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run([
                    "curl", "--fail", "--location", "--retry", "4", "--retry-delay", "2",
                    "--continue-at", "-", "--output", str(partial),
                    f"https://modelscope.cn/models/{REPOSITORY}/resolve/{REVISION}/{relative}",
                ], check=True)
            verify(partial, size, sha256)
            partial.rename(target)
        print(f"verified {relative}: {sha256}", flush=True)
        records.append(dict(path=relative, bytes=size, sha256=sha256))
    provenance = dict(schema="turbocider-seedvr2-source-v1", source="modelscope",
                      repository=REPOSITORY, revision=REVISION, files=records)
    temporary = root / "source.json.tmp"
    temporary.write_text(json.dumps(provenance, indent=2) + "\n")
    temporary.replace(root / "source.json")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/Comfy-Org-SeedVR2"))
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify already downloaded final/partial files; no network access.")
    args = parser.parse_args()
    download(args.output, args.verify_only)


if __name__ == "__main__":
    main()
