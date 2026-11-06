"""Shared transport for the H3 VAE and ordinary FL2VA downloaders.

Component manifests, pinned identities and download locks stay in each entry
point. This module only streams resumable files and computes their digest.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time
import urllib.error
import urllib.request


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
                            print(f"  {target.name}: {downloaded / 2**30:.2f} GiB{suffix}", flush=True)
                            last_report = now
                if total and downloaded != total:
                    raise IOError(f"short response ({downloaded} != {total} bytes)")
            part.replace(target)
            return
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            print(f"[retry {attempt}/{retries}] {target.name}: {error}", flush=True)
            if attempt == retries:
                raise
