#!/usr/bin/env python3
"""Download and verify PrunaVAED from ModelScope; optionally install for LTX."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from modelscope_ranges import download_ranges

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "PrunaAI/PrunaVAED"
# Pinned source files: this decoder's graph is validated by the native loader.
FILES = [
    ("vae/diffusion_pytorch_model.safetensors", "16dbad8c7c3fa0c2da35464a34dc1aaf592e7390",
     1327909418, "4cbf0cbe6c185514d62c6c58c35dc42d7ea15924f34391e08be12f44bfccdf1d"),
    ("vae/config.json", "2249d0f2ce3080f9dbea1223ef4a3f91ddf7b573",
     1475, "b636e95d41d93027529fa726ff75fe6eed55751a8cbe602e614ef7a600d287e7"),
    ("patch_diffusers.py", "ace541926f00f545b77a099afbd45784636b2370",
     8796, "dfac5ca02891f3d7061f54b1410db65d1dfb91e558bee93ca571ee1ed066dbf3"),
    ("README.md", "0dd0d6944ec14b2ed4a2e08d3cac12db1f67eca5",
     16886, "be447ec28032441c59f60ecf7728e5b21c2e2a635df9e7d716fe29a57e748344"),
]


def install(source: Path, model_root: Path) -> Path:
    destination = model_root / "vae/prunavaed.safetensors"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        with destination.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != FILES[0][3]:
                raise ValueError(f"Refusing to replace different weights: {destination}")
        return destination
    # Hard-link when possible; no additional 1.33 GB allocation on the same disk.
    try:
        os.link(source, destination)
    except OSError as error:
        import errno
        if error.errno != errno.EXDEV:
            raise
        temporary = destination.with_suffix(".safetensors.part")
        with source.open("rb") as src, temporary.open("xb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        os.link(temporary, destination)  # Publish without replacing another file.
        temporary.unlink()
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "models/PrunaVAED")
    parser.add_argument("--install-root", type=Path, help="LTX model directory")
    parser.add_argument("--connections", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    receipts = []
    for name, revision, size, sha256 in FILES:
        asset = dict(repository=REPOSITORY, revision=revision, filename=name,
                     relative_path=name, bytes=size, sha256=sha256)
        receipts.append(download_ranges(asset, args.root, connections=args.connections))
    (args.root / "native-download-receipt.json").write_text(
        json.dumps(dict(repository=REPOSITORY, assets=receipts), indent=2) + "\n")
    if args.install_root:
        print(install(args.root / FILES[0][0], args.install_root))


if __name__ == "__main__":
    main()
