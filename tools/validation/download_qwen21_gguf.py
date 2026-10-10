#!/usr/bin/env python3
"""Fetch pinned Qwen21 Q4_K_M assets without a second Hub cache copy.

Weights are published only after a full LFS SHA256 check. Partial files are
resumable, servers must honor Range exactly, and the existing BF16 VAE and
processor are linked read-only rather than duplicated or converted.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import urllib.error
import urllib.request

ASSETS = (
    {
        "repository": "unsloth/Qwen-Image-2.1-GGUF",
        "revision": "2c31ccd392b367a6637841a143813320a02dff55",
        "filename": "qwen-image-2.1-Q4_K_M.gguf",
        "relative_path": "diffusion_models/qwen-image-2.1-Q4_K_M.gguf",
        "bytes": 4199565024,
        "sha256": "631d532e7ca71e8d90a87c71d3699761a812039d22e3370e87498d87754660fe",
    },
    {
        "repository": "unsloth/Qwen3-VL-8B-Instruct-GGUF",
        "revision": "b93a7ee713758252c555be4210c00540df954dc2",
        "filename": "Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
        "relative_path": "text_encoders/Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
        "bytes": 5027785568,
        "sha256": "108e7ff92b78eefd3db4741885104acba514255c11b617d3c7b197a5f46efe89",
    },
)
DEFAULT_ENDPOINTS = ("https://hf-mirror.com", "https://huggingface.co")


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def local_size(path):
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("download target must be a regular non-symlink file")
    return path.stat().st_size if path.exists() else 0


def check_response(response, offset, size):
    status = response.status
    if status == 206:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if not match or tuple(map(int, match.groups())) != (offset, size - 1, size):
            raise ValueError("server returned a different Range or source size")
    elif status != 200 or offset:
        raise ValueError("server did not honor the requested byte range")
    length = response.headers.get("Content-Length")
    if length is not None and int(length) != size - offset:
        raise ValueError("server Content-Length differs from the pinned source")
    if response.headers.get("Content-Encoding", "identity") not in ("", "identity"):
        raise ValueError("compressed transfer cannot safely resume binary model bytes")


def download_asset(asset, root, endpoints=DEFAULT_ENDPOINTS, retries=8):
    target = root / asset["relative_path"]
    partial = target.with_name(target.name + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if local_size(target) != asset["bytes"] or digest(target) != asset["sha256"]:
            raise ValueError("existing published asset does not match the pinned source")
        return {**asset, "status": "verified_existing"}
    size = asset["bytes"]
    for attempt in range(retries):
        offset = local_size(partial)
        if offset > size:
            raise ValueError("partial file is larger than the pinned source")
        if offset == size:
            break
        endpoint = endpoints[attempt % len(endpoints)].rstrip("/")
        url = f"{endpoint}/{asset['repository']}/resolve/{asset['revision']}/{asset['filename']}"
        headers = {"User-Agent": "TurboCider-pinned-GGUF-download", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        print(json.dumps({"downloading": asset["filename"], "endpoint": endpoint, "resume_bytes": offset}), flush=True)
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                check_response(response, offset, size)
                # Refuse path aliases before opening a resumable publication.
                local_size(partial)
                with partial.open("ab") as output:
                    received = offset
                    last_progress = time.monotonic()
                    while chunk := response.read(4 << 20):
                        if len(chunk) > size - received:
                            raise ValueError("server sent bytes beyond the pinned source")
                        output.write(chunk)
                        received += len(chunk)
                        if time.monotonic() - last_progress >= 5:
                            output.flush()
                            print(json.dumps({"asset": asset["filename"], "bytes": received, "total_bytes": size}), flush=True)
                            last_progress = time.monotonic()
                    if received != size:
                        raise OSError("transfer ended before the pinned source size")
            break
        except (OSError, urllib.error.URLError) as error:
            print(json.dumps({"retry": attempt + 1, "asset": asset["filename"], "error_type": type(error).__name__}), flush=True)
            if attempt + 1 == retries:
                raise
            time.sleep(min(attempt + 1, 3))
    if local_size(partial) != size or digest(partial) != asset["sha256"]:
        raise ValueError("complete partial asset failed the pinned LFS SHA256 check; not published")
    if target.exists() or target.is_symlink():
        raise ValueError("publication target appeared during the download")
    partial.rename(target)
    print(json.dumps({"verified": asset["filename"], "bytes": size, "sha256": asset["sha256"]}), flush=True)
    return {**asset, "status": "downloaded_verified"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bf16-root", type=Path, required=True)
    parser.add_argument("--endpoint", action="append")
    parser.add_argument("--reserve-bytes", type=int, default=1 << 30)
    args = parser.parse_args()
    if args.reserve_bytes < 1 << 30:
        parser.error("keep at least 1GiB free for runtime/build evidence")
    source = args.bf16_root.resolve(strict=True)
    vae = source / "vae/qwen_image_2.1_vae_bf16.safetensors"
    processor = source / "processor"
    if not vae.is_file() or not (processor / "tokenizer.json").is_file():
        parser.error("existing original BF16 VAE and processor required")
    root = args.output.absolute()
    root.mkdir(parents=True, exist_ok=True)
    remaining = 0
    for asset in ASSETS:
        target = root / asset["relative_path"]
        if target.exists():
            local_size(target)
        else:
            partial = target.with_name(target.name + ".part")
            remaining += max(0, asset["bytes"] - local_size(partial))
    if shutil.disk_usage(root).free < remaining + args.reserve_bytes:
        parser.error("insufficient free space for both pinned assets plus runtime reserve")
    results = [download_asset(asset, root, args.endpoint or DEFAULT_ENDPOINTS) for asset in ASSETS]
    for relative, original in (("vae/qwen_image_2.1_vae_bf16.safetensors", vae), ("processor", processor)):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve(strict=True) != original:
                raise ValueError("BF16 companion publication target conflicts with the original source")
        else:
            target.symlink_to(original)
    receipt = {"schema": "tc-qwen21-pinned-gguf-assets-v1", "status": "verified_assets", "assets": results,
               "vae": "original_bf16_read_only_link", "processor": "original_read_only_link",
               "runtime_support_verified": False, "performance_verified": False}
    (root / "download-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
