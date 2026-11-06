#!/usr/bin/env python3
"""Download the original MiniMax-H3 FL2VA VAE assets from ModelScope.

The vpipe MiniMax-H3 Metal decoders consume the released H3 layout:
``FL2VA/video_vae/source`` and ``FL2VA/audio_vae``.  The FastH3 preview
repository contains equivalent-looking diffusers files, but its video VAE
does not carry the source/wrapper envelope expected by the vpipe loader.

Files are written directly to the final tree using resumable ``.part``
siblings.  The manifest records the ModelScope endpoint, exact sizes, and
SHA-256 digests after a successful download; it is deliberately separate from
the transformer downloader so a partial VAE fetch cannot make the VDN base
look complete.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path


# Keep direct CLI invocation and file-based tooling imports working.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h3_download_common import download_file, sha256_file  # noqa: E402


MODEL_ID = "MiniMax/MiniMax-H3"
REVISION = "master"
ENDPOINT = "https://modelscope.cn"
FILES = {
    "FL2VA/video_vae/config.json": 1807,
    "FL2VA/video_vae/source/config.json": 1164,
    "FL2VA/video_vae/source/model.safetensors": 10_415_548_320,
    "FL2VA/audio_vae/config.json": 1973,
    "FL2VA/audio_vae/metadata.json": 440,
    "FL2VA/audio_vae/model.safetensors": 605_429_308,
}


@contextmanager
def download_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".modelscope_vae_download.lock"
    with path.open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another H3 VAE ModelScope download is active for {root}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="MiniMax-H3 model root, e.g. models/MiniMax-H3-ModelScope")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--min-free-gib", type=float, default=4.0)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.revision != REVISION:
        raise SystemExit(f"this pinned VAE downloader accepts revision {REVISION!r}")
    root = args.root.expanduser().resolve()
    needed = sum(size for relative, size in FILES.items()
                 if not (root / relative).is_file() or
                 (root / relative).stat().st_size != size)
    free = shutil.disk_usage(root.parent).free / 2**30
    if not args.verify_only and free < needed / 2**30 + args.min_free_gib:
        raise RuntimeError(
            f"H3 VAE assets need {needed / 2**30:.2f} GiB and only {free:.2f} GiB is free")
    with download_lock(root):
        records = {}
        base = f"{ENDPOINT}/models/{MODEL_ID}/resolve/{args.revision}"
        for relative, expected_size in FILES.items():
            target = root / relative
            if target.is_file() and target.stat().st_size == expected_size:
                print(f"[skip] {relative}", flush=True)
            elif args.verify_only:
                raise RuntimeError(f"missing or wrong-sized H3 VAE file: {relative}")
            else:
                print(f"[download] {relative}", flush=True)
                download_file(f"{base}/{relative}?download=true", target)
            if target.stat().st_size != expected_size:
                raise RuntimeError(f"size mismatch for {relative}: {target.stat().st_size} != {expected_size}")
            digest = sha256_file(target)
            print(f"[verify] {relative} sha256={digest}", flush=True)
            records[relative] = {"bytes": expected_size, "sha256": digest}
        manifest = {
            "schema": "turbocider-modelscope-h3-fl2va-vae-v1",
            "repository": MODEL_ID,
            "revision": REVISION,
            "endpoint": ENDPOINT,
            "partition": "FL2VA",
            "files": records,
            "purpose": "original MiniMax-H3 VAE layout for VDN/vpipe-compatible decode",
        }
        path = root / "modelscope_vae_download.json"
        if not args.verify_only:
            path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        else:
            if not path.is_file():
                raise RuntimeError(f"missing VAE manifest: {path}")
            old = json.loads(path.read_text())
            if old.get("schema") != manifest["schema"] or old.get("files") != records:
                raise RuntimeError("VAE manifest identity or digest mismatch")
    print(f"[ready] {root} FL2VA VAE assets={len(FILES)}", flush=True)


if __name__ == "__main__":
    main()
