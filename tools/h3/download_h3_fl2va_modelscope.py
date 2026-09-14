#!/usr/bin/env python3
"""Download the ordinary MiniMax-H3 FL2VA transformer from ModelScope.

VDN stage-DMD is an auxiliary branch attached to the *ordinary* FL2VA
backbone.  It must not be built from the already merged LightX2V/Turbo
checkpoint or from the pruned ConvRot checkpoint, because those checkpoints
replace the 2688-dimensional time embedding with a different AdaLN table.

This downloader intentionally fetches only the 13 transformer shards and the
transformer config/index.  Text encoder, VAE and tokenizer assets are already
provided by the ModelScope FastH3 snapshot used by TurboCider.  Files are
written directly to their final location through resumable ``.part`` siblings
and are verified by the pinned ModelScope SHA-256 values.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path


MODEL_ID = "MiniMax/MiniMax-H3"
REVISION = "master"
ENDPOINT = "https://modelscope.cn"
PREFIX = "FL2VA/transformer"

# These values are from the ModelScope repository tree at the pinned revision.
# Keeping the list explicit prevents a changed upstream tree from silently
# becoming a different VDN base.
FILES = {
    f"{PREFIX}/config.json": {
        "bytes": 604,
        "sha256": "f619093a231fcfbcc3d035bec26c50ad864e7331a500d5c519f5045dc1e50458",
    },
    f"{PREFIX}/model.safetensors.index.json": {
        "bytes": 38_323,
        "sha256": "fb457a26ffa6294660e249b0ddd03a337f2e5393f770b5c34c8b8f90a29a7efb",
    },
    f"{PREFIX}/model-00001-of-00013.safetensors": {
        "bytes": 5_227_812_968,
        "sha256": "0b3386565e476bfdea287e9ea9f269d036e5c649ed14bb9b4afac1dc4661bd2a",
    },
    f"{PREFIX}/model-00002-of-00013.safetensors": {
        "bytes": 5_164_578_856,
        "sha256": "9c98fd4579c9bc96d1b1bbf65c1243f9256df60bccd229570cecc61897773003",
    },
    f"{PREFIX}/model-00003-of-00013.safetensors": {
        "bytes": 5_164_578_872,
        "sha256": "fa484c940d4170199fb69f5016739df2daf6ecd1150631b12a297e06e3964730",
    },
    f"{PREFIX}/model-00004-of-00013.safetensors": {
        "bytes": 5_164_578_896,
        "sha256": "4a851df37cce2d14b39cb58df41def7b591b1f11b8520a32b421a5fa3e997c9f",
    },
    f"{PREFIX}/model-00005-of-00013.safetensors": {
        "bytes": 5_164_578_896,
        "sha256": "3fe6ff94dc9d3107c6776a25277d2b72b5443f88cf0f4ed835fd5b65a94ea5d4",
    },
    f"{PREFIX}/model-00006-of-00013.safetensors": {
        "bytes": 5_164_578_896,
        "sha256": "79b47e1b9ff03a2c6f06dd15a28972acf32c0867172ee34e5aa6e96b99494127",
    },
    f"{PREFIX}/model-00007-of-00013.safetensors": {
        "bytes": 5_164_578_896,
        "sha256": "6ac4d6ce639786b722a8dd99843ed56dcab48a0d174a4b0f9b8cdcbe5160fb0f",
    },
    f"{PREFIX}/model-00008-of-00013.safetensors": {
        "bytes": 5_164_578_896,
        "sha256": "03531812243fb0a333e01efa7e3d39750580bca8cbf2ba0529da419278eab736",
    },
    f"{PREFIX}/model-00009-of-00013.safetensors": {
        "bytes": 5_164_578_896,
        "sha256": "7e82a80b0d0d267026eb841b81523069cc7b956c1ebd8a18c07e462ffb262487",
    },
    f"{PREFIX}/model-00010-of-00013.safetensors": {
        "bytes": 5_164_578_896,
        "sha256": "ef0a1f6b65145232543dd2232d58d91a4d31f44b0da5084acf22cea123255481",
    },
    f"{PREFIX}/model-00011-of-00013.safetensors": {
        "bytes": 5_164_578_896,
        "sha256": "bf885b8f2f078c5499b75fded21094a26bfe0e6e00bee276e499443ecd2f298d",
    },
    f"{PREFIX}/model-00012-of-00013.safetensors": {
        "bytes": 5_164_578_896,
        "sha256": "dc717572d014ebf8527e35af815d10eb2f5a6bc87253b8a7d2f456965192db69",
    },
    f"{PREFIX}/model-00013-of-00013.safetensors": {
        "bytes": 4_242_305_176,
        "sha256": "8bfd852d5817e9836de1d3ec8dbac1c5446b167568b717371f08282b22291aa2",
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def download_lock(root: Path):
    """Prevent two resumable downloaders from racing on the same .part file."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".modelscope_download.lock"
    with path.open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another FL2VA ModelScope download is active for {root}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def download_file(url: str, target: Path, *, retries: int = 5) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    for attempt in range(1, retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                status = getattr(response, "status", response.getcode())
                if offset and status != 206:
                    offset = 0
                    part.unlink(missing_ok=True)
                mode = "ab" if offset else "wb"
                content_length = int(response.headers.get("Content-Length", "0"))
                total = offset + content_length if content_length else 0
                downloaded = offset
                last_report = time.monotonic()
                with part.open(mode) as stream:
                    while True:
                        chunk = response.read(16 * 1024 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 10:
                            suffix = f"/{total / 2**30:.2f} GiB" if total else ""
                            print(f"  {target.name}: {downloaded / 2**30:.2f} GiB{suffix}",
                                  flush=True)
                            last_report = now
                if total and downloaded != total:
                    raise IOError(f"short response ({downloaded} != {total} bytes)")
            part.replace(target)
            return
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            print(f"[retry {attempt}/{retries}] {target.name}: {error}", flush=True)
            if attempt == retries:
                raise


def validate_index(root: Path) -> None:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map")
    require(isinstance(weight_map, dict) and weight_map,
            "FL2VA transformer index has no weight_map")
    expected_shards = {Path(path).name for path in FILES
                       if path.endswith(".safetensors")}
    require(set(weight_map.values()) == expected_shards,
            "FL2VA transformer index does not reference the pinned 13 shards")
    require(len(weight_map) > 500,
            "FL2VA transformer index is unexpectedly small")


def validate_manifest(root: Path) -> None:
    manifest = json.loads((root / "modelscope_download.json").read_text())
    require(manifest.get("schema") == "turbocider-modelscope-h3-fl2va-v1",
            "invalid FL2VA ModelScope manifest schema")
    require(manifest.get("repository") == MODEL_ID and
            manifest.get("revision") == REVISION and
            manifest.get("files") == FILES,
            "FL2VA ModelScope manifest identity mismatch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="destination model root, e.g. models/MiniMax-H3-ModelScope")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--min-free-gib", type=float, default=12.0)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.revision != REVISION:
        raise SystemExit(f"this pinned base accepts revision {REVISION!r}")
    root = args.root.expanduser().resolve()
    with download_lock(root):
        needed = sum(int(record["bytes"])
                     for path, record in FILES.items()
                     if not (root / path).is_file())
        free = shutil.disk_usage(root.parent).free / 2**30
        if not args.verify_only and free < needed / 2**30 + args.min_free_gib:
            raise RuntimeError(
                f"FL2VA transformer needs {needed / 2**30:.2f} GiB and only {free:.2f} GiB is free")
        base = f"{ENDPOINT}/models/{MODEL_ID}/resolve/{args.revision}"
        for relative, record in FILES.items():
            target = root / relative
            expected_size = int(record["bytes"])
            if target.is_file() and target.stat().st_size == expected_size:
                print(f"[skip] {relative}", flush=True)
            elif args.verify_only:
                raise RuntimeError(f"missing or wrong-sized FL2VA file: {relative}")
            else:
                print(f"[download] {relative}", flush=True)
                download_file(f"{base}/{relative}?download=true", target)
            require(target.stat().st_size == expected_size,
                    f"size mismatch for {relative}")
            digest = sha256_file(target)
            print(f"[verify] {relative} sha256={digest}", flush=True)
            require(digest == record["sha256"], f"SHA-256 mismatch for {relative}")
        validate_index(root / PREFIX)
        manifest = {
            "schema": "turbocider-modelscope-h3-fl2va-v1",
            "repository": MODEL_ID,
            "revision": REVISION,
            "endpoint": ENDPOINT,
            "partition": "FL2VA",
            "files": FILES,
            "purpose": "VDN ordinary FL2VA base; not LightX2V/Turbo or pruned ConvRot",
        }
        manifest_path = root / "modelscope_download.json"
        if not args.verify_only:
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        validate_manifest(root)
    print(f"[ready] {root}", flush=True)


if __name__ == "__main__":
    main()
