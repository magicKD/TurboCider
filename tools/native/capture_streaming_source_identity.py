#!/usr/bin/env python3
"""Capture reproducible source provenance for a streaming campaign build."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = 1
EXCLUDED_PARTS = {
    ".git", ".deps", ".venv", ".pytest_cache", ".mypy_cache",
    "__pycache__", "build", "dist", "models", "Python", "results",
}


class IdentityError(RuntimeError):
    pass


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def run_git(root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True, capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        message = getattr(exc, "stderr", b"")
        if isinstance(message, bytes):
            message = message.decode(errors="replace")
        raise IdentityError(f"git {' '.join(arguments)} failed: {message}") from exc
    return completed.stdout


def included(relative: Path) -> bool:
    return bool(relative.parts) and not any(
        part in EXCLUDED_PARTS for part in relative.parts
    )


def file_record(root: Path, relative: Path) -> dict:
    path = root / relative
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return {"path": relative.as_posix(), "kind": "deleted"}
    executable = bool(metadata.st_mode & stat.S_IXUSR)
    if path.is_symlink():
        target = os.readlink(path)
        payload = target.encode()
        return {
            "path": relative.as_posix(),
            "kind": "symlink",
            "target": target,
            "sha256": digest_bytes(payload),
            "executable": executable,
        }
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        return {
            "path": relative.as_posix(),
            "kind": "file",
            "bytes": metadata.st_size,
            "sha256": digest.hexdigest(),
            "executable": executable,
        }
    return {"path": relative.as_posix(), "kind": "unsupported"}


def manifest(records: Iterable[dict]) -> tuple[str, list[dict]]:
    ordered = sorted(records, key=lambda item: item["path"])
    encoded = json.dumps(
        ordered, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return digest_bytes(encoded), ordered


def archive_paths(root: Path) -> list[Path]:
    return sorted(
        path.relative_to(root)
        for path in root.rglob("*")
        if (path.is_file() or path.is_symlink()) and
        included(path.relative_to(root))
    )


def zero_split(value: bytes) -> list[Path]:
    return [
        Path(item.decode()) for item in value.split(b"\0")
        if item and included(Path(item.decode()))
    ]


def capture_git(root: Path) -> dict:
    commit = run_git(root, "rev-parse", "HEAD").decode().strip()
    tracked = zero_split(run_git(root, "ls-files", "-z"))
    untracked = zero_split(
        run_git(root, "ls-files", "--others", "--exclude-standard", "-z")
    )
    diff = run_git(root, "diff", "--binary", "--no-ext-diff", "HEAD", "--")
    tracked_digest, tracked_records = manifest(
        file_record(root, path) for path in tracked
    )
    untracked_digest, untracked_records = manifest(
        file_record(root, path) for path in untracked
    )
    source_digest, source_records = manifest(
        [*tracked_records, *untracked_records]
    )
    clean = not diff and not untracked_records
    return {
        "format": "turbocider-streaming-source-identity-v1",
        "schema_version": SCHEMA_VERSION,
        "commit": commit,
        "clean": clean,
        "source_manifest_sha256": source_digest,
        "tracked_worktree_sha256": tracked_digest,
        "tracked_file_count": len(tracked_records),
        "dirty_diff_sha256": digest_bytes(diff),
        "dirty_diff_bytes": len(diff),
        "untracked_sources_sha256": untracked_digest,
        "untracked_source_count": len(untracked_records),
        "untracked_sources": [record["path"] for record in untracked_records],
        "excluded_directory_names": sorted(EXCLUDED_PARTS),
    }


def capture_archive(root: Path, commit: str) -> dict:
    paths = archive_paths(root)
    source_digest, records = manifest(file_record(root, path) for path in paths)
    return {
        "format": "turbocider-streaming-source-identity-v1",
        "schema_version": SCHEMA_VERSION,
        "commit": commit,
        "clean": True,
        "source_manifest_sha256": source_digest,
        "tracked_worktree_sha256": source_digest,
        "tracked_file_count": len(records),
        "dirty_diff_sha256": digest_bytes(b""),
        "dirty_diff_bytes": 0,
        "untracked_sources_sha256": digest_bytes(b"[]"),
        "untracked_source_count": 0,
        "untracked_sources": [],
        "excluded_directory_names": sorted(EXCLUDED_PARTS),
    }


def capture(root: Path, commit: str | None = None) -> dict:
    root = root.resolve()
    if not root.is_dir():
        raise IdentityError(f"source root is not a directory: {root}")
    if (root / ".git").exists():
        if commit:
            raise IdentityError("--commit is only for an exported source tree")
        return capture_git(root)
    if not commit or len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit.lower()
    ):
        raise IdentityError("an exported source tree requires a 40-hex commit")
    return capture_archive(root, commit)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--commit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = capture(args.source_root, args.commit)
    except IdentityError as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2))
        return 2
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
