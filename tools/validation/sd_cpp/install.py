#!/usr/bin/env python3
"""Install a pinned sd.cpp reference for developer-only comparisons.

Never required by TurboCider build/package or shipped to users.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

TAG = "master-813-bfbef5b-u13b9d92"
ARCHIVE = f"sd-{TAG}-bin-Darwin-macOS-arm64.zip"
URL = f"https://github.com/unslothai/stable-diffusion.cpp/releases/download/{TAG}/{ARCHIVE}"
SHA256 = "1778b6da9529f74a77a056abfddbd00a6d06efa4f5865ee51a2509d36b9aa38e"
EXPECTED = {"sd-cli", "sd-server", "LICENSE", "UNSLOTH_BUILD.txt"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def download(destination: Path) -> None:
    request = urllib.request.Request(URL, headers={"User-Agent": "TurboCider dependency installer"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 << 20)


def extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        selected: dict[str, str] = {}
        for name in bundle.namelist():
            parts = PurePosixPath(name).parts
            if len(parts) == 2 and parts[-1] in EXPECTED:
                selected[parts[-1]] = name
        missing = EXPECTED - selected.keys()
        if missing:
            raise SystemExit(f"sd.cpp archive is missing: {', '.join(sorted(missing))}")
        destination.mkdir(parents=True)
        for basename, member in selected.items():
            target = destination / basename
            with bundle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 << 20)
            if basename in {"sd-cli", "sd-server"}:
                target.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, help="use an already downloaded verified zip")
    parser.add_argument("--offline", action="store_true", help="fail instead of downloading")
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("the pinned GGUF runtime currently supports Apple Silicon macOS")
    root = Path(__file__).resolve().parents[3]
    parent = root / ".deps" / "stable-diffusion-cpp"
    destination = parent / TAG
    current = parent / "current"
    marker = destination / "turbocider-install.json"
    installed = destination.joinpath("sd-server").is_file() and marker.is_file()
    if installed:
        record = json.loads(marker.read_text())
        if record.get("archive_sha256") == SHA256:
            if current.is_symlink() and current.resolve() == destination:
                print(destination)
                return
            if current.exists() and not current.is_symlink():
                raise SystemExit(f"refusing to replace non-symlink: {current}")
            current.unlink(missing_ok=True)
            current.symlink_to(TAG, target_is_directory=True)
            print(destination)
            return
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="turbocider-sd-cpp-", dir=parent) as temporary:
        temporary = Path(temporary)
        archive = args.archive.resolve() if args.archive else temporary / ARCHIVE
        if args.archive:
            if not archive.is_file():
                raise SystemExit(f"archive does not exist: {archive}")
        elif args.offline:
            raise SystemExit("offline install requires --archive")
        else:
            download(archive)
        actual = digest(archive)
        if actual != SHA256:
            raise SystemExit(f"sd.cpp archive SHA-256 mismatch: {actual}")
        staged = temporary / "runtime"
        extract(archive, staged)
        (staged / "turbocider-install.json").write_text(
            json.dumps({"owner": "TurboCider", "tag": TAG, "archive_sha256": actual}, indent=2)
            + "\n"
        )
        if destination.exists():
            shutil.rmtree(destination)
        staged.rename(destination)
    if current.is_symlink() or current.exists():
        if current.is_dir() and not current.is_symlink():
            raise SystemExit(f"refusing to replace non-symlink: {current}")
        current.unlink()
    current.symlink_to(TAG, target_is_directory=True)
    print(destination)


if __name__ == "__main__":
    main()
