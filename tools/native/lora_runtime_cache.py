#!/usr/bin/env python3
"""Content-addressed runtime LoRA bake/cache support.

The user-owned model tree contains only the base checkpoint and the adapter.
When a native backend needs a merged checkpoint, this module builds it once in
``~/Library/Caches/TurboCider/lora`` (or ``TURBOCIDER_LORA_CACHE_DIR``), under
an identity derived from the complete input files and merge contract.  The
cache is disposable; it is never treated as a source model or edited in place.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable


ALGORITHM = {
    "h3": "turbocider-h3-runtime-lora-bake-v3",
    "ltx": "turbocider-ltx-refiner-bake-v2",
}
SCHEMA = "turbocider-runtime-lora-cache-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class HashMemo:
    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            value = json.loads(path.read_text())
            self.values = value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            self.values = {}

    def digest(self, path: Path, stat: os.stat_result) -> str:
        key = str(path)
        cached = self.values.get(key)
        if (
            isinstance(cached, dict)
            and cached.get("bytes") == stat.st_size
            and cached.get("mtime_ns") == stat.st_mtime_ns
            and isinstance(cached.get("sha256"), str)
        ):
            return cached["sha256"]
        digest = _sha256(path)
        self.values[key] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest,
        }
        return digest

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        temporary.write_text(json.dumps(self.values, sort_keys=True) + "\n")
        os.replace(temporary, self.path)


def file_identity(path: Path, memo: HashMemo | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": (memo.digest(path, stat) if memo else _sha256(path))
        if path.is_file() else None,
    }


def _tree_identity(path: Path, memo: HashMemo | None = None) -> dict[str, Any]:
    """Identity for a sharded checkpoint directory without copying it."""
    path = path.expanduser().resolve()
    files = []
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and not candidate.name.startswith("."):
            files.append(file_identity(candidate, memo))
    return {"path": str(path), "files": files}


def cache_root(value: str | Path | None = None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    configured = os.environ.get("TURBOCIDER_LORA_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / "Library" / "Caches" / "TurboCider" / "lora"


def _lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def identity(
    model: str,
    base: Path,
    adapter: Path,
    strength: float,
    role: str,
    *,
    hardware: str = "apple-silicon",
    algorithm: str | None = None,
    memo: HashMemo | None = None,
) -> tuple[str, dict[str, Any]]:
    base_id = _tree_identity(base, memo) if base.is_dir() else file_identity(base, memo)
    adapter_id = file_identity(adapter, memo)
    material = {
        "schema": SCHEMA,
        "model": model,
        "base": base_id,
        "adapter": adapter_id,
        "strength": float(strength),
        "role": role,
        "algorithm": algorithm or ALGORITHM.get(model, "runtime-bake-v1"),
        "hardware": hardware,
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), material


def _script_path() -> Path:
    # A test/developer can point at an explicitly controlled implementation;
    # this is never inferred from a workspace or sibling checkout.
    override = os.environ.get("TURBOCIDER_LORA_TOOL_DIR")
    if override:
        candidate = Path(override).expanduser().resolve()
        if candidate.is_dir():
            return candidate
    # The merge implementations ship beside this cache helper in both the
    # source tree and packaged App/CLI. Never discover a sibling checkout:
    # a portable TurboCider installation must be sufficient on its own.
    return Path(__file__).resolve().parent


def _run_merge(model: str, base: Path, adapter: Path, output: Path,
               strength: float, profile: str) -> None:
    tools = _script_path()
    if model == "h3":
        script = tools / "merge_h3_lora.py"
        command = [
            os.environ.get("TURBOCIDER_PREPARE_PYTHON", os.sys.executable),
            str(script), str(base), str(adapter), str(output),
            "--profile", profile, "--strength", str(strength),
            "--device", os.environ.get("TURBOCIDER_LORA_MERGE_DEVICE", "cpu"),
        ]
    elif model == "ltx":
        script = tools / "merge_ltx_refiner.py"
        command = [
            os.environ.get("TURBOCIDER_PREPARE_PYTHON", os.sys.executable),
            str(script), str(base), str(adapter), str(output),
            "--device", os.environ.get("TURBOCIDER_LORA_MERGE_DEVICE", "auto"),
        ]
    else:
        raise ValueError(f"no offline merge implementation for {model!r}")
    if not script.is_file():
        raise FileNotFoundError(f"LoRA merge tool is unavailable: {script}")
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(detail[-16_384:] or f"merge tool exited {completed.returncode}")


def ensure_cache(
    model: str,
    base: str | Path,
    adapter: str | Path,
    strength: float,
    *,
    role: str = "transformer",
    cache_dir: str | Path | None = None,
    profile: str = "auto",
    hardware: str = "apple-silicon",
) -> dict[str, Any]:
    base_path = Path(base).expanduser().resolve()
    adapter_path = Path(adapter).expanduser().resolve()
    if not base_path.exists():
        raise FileNotFoundError(f"LoRA base does not exist: {base_path}")
    if not adapter_path.is_file():
        raise FileNotFoundError(f"LoRA adapter does not exist: {adapter_path}")
    cache_base = cache_root(cache_dir)
    cache_base.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(_lock(cache_base / ".hashes.lock")):
        memo = HashMemo(cache_base / "hashes-v1.json")
        key, material = identity(
            model, base_path, adapter_path, strength, role,
            hardware=hardware, memo=memo,
        )
        memo.save()
    root = cache_base / model / key
    marker = root / "cache.json"
    lock_path = root.parent / f".{key}.lock"
    root.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(_lock(lock_path)):
        if marker.is_file():
            try:
                value = json.loads(marker.read_text())
                artifact = Path(value["artifact"])
                manifest = Path(value["manifest"])
                if value.get("identity") == material and artifact.exists() and manifest.exists():
                    value["cache_hit"] = True
                    return value
            except (OSError, KeyError, TypeError, ValueError):
                pass
        staging = Path(tempfile.mkdtemp(prefix=f".{key}.staging-", dir=root.parent))
        try:
            if model == "h3":
                output = staging / "transformer"
                output.mkdir()
                _run_merge(model, base_path, adapter_path, output, strength, profile)
                component = base_path.parent
                source_model = component.parent
                if base_path.name != "transformer" or component == source_model:
                    raise ValueError(
                        "H3 runtime bake base must be MODEL/COMPONENT/transformer"
                    )
                model_tree = staging / "model"
                model_tree.mkdir()
                for source in source_model.iterdir():
                    if source.name.startswith(".") or source == component:
                        continue
                    (model_tree / source.name).symlink_to(source)
                cached_component = model_tree / component.name
                cached_component.mkdir()
                for source in component.iterdir():
                    if source.name.startswith(".") or source == base_path:
                        continue
                    (cached_component / source.name).symlink_to(source)
                (cached_component / "transformer").symlink_to(
                    Path("..") / ".." / "transformer"
                )
                artifact = model_tree
                manifest = cached_component / "transformer" / "h3-turbo-merge-manifest.json"
            elif model == "ltx":
                output = staging / "ltx-2.5-22b-dev-refiner-lora-0.8-comfy-int8-convrot.safetensors"
                _run_merge(model, base_path, adapter_path, output, strength, profile)
                # The upstream manifest records plain filenames and the native
                # validator deliberately requires those records to resolve in
                # the cache entry. Symlinks cost no extra checkpoint space.
                for source in (base_path, adapter_path):
                    link = staging / source.name
                    if not link.exists():
                        link.symlink_to(source)
                artifact = output
                manifest = Path(str(output) + ".manifest.json")
            else:
                raise ValueError(f"no offline merge implementation for {model!r}")
            if not artifact.exists() or not manifest.is_file():
                raise RuntimeError("LoRA merge completed without a manifest-bound artifact")
            value = {
                "schema": SCHEMA,
                "model": model,
                "identity": material,
                "cache_key": key,
                "artifact": str(artifact),
                "manifest": str(manifest),
                "cache_hit": False,
                "created_unix": time.time(),
                "evictable": True,
            }
            (staging / "cache.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
            if root.exists():
                shutil.rmtree(root)
            staging.replace(root)
            value["artifact"] = str(root / artifact.relative_to(staging))
            value["manifest"] = str(root / manifest.relative_to(staging))
            (root / "cache.json").write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
            return value
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def _main(argv: list[str] | None = None) -> int:
    arguments = list(os.sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] not in {"ensure", "prune"}:
        arguments.insert(0, "ensure")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("ensure", "prune"), nargs="?", default="ensure")
    parser.add_argument("model", choices=("h3", "ltx"), nargs="?")
    parser.add_argument("base", nargs="?")
    parser.add_argument("adapter", nargs="?")
    parser.add_argument("--strength", type=float)
    parser.add_argument("--role", default="transformer")
    parser.add_argument("--profile", default="auto")
    parser.add_argument("--cache-dir")
    parser.add_argument("--hardware", default="apple-silicon")
    parser.add_argument("--max-bytes", type=int, default=0)
    args = parser.parse_args(arguments)
    if args.command == "prune":
        if args.max_bytes <= 0:
            parser.error("prune requires a positive --max-bytes budget")
        root = cache_root(args.cache_dir)
        removed = 0
        reclaimed = 0
        if root.is_dir():
            entries = []
            for marker in root.glob("*/*/cache.json"):
                try:
                    value = json.loads(marker.read_text())
                    entry = marker.parent
                    size = sum(
                        path.stat().st_size for path in entry.rglob("*")
                        if path.is_file() and not path.is_symlink()
                    )
                    entries.append((float(value.get("created_unix", 0.0)), size, entry))
                except (OSError, ValueError):
                    continue
            total = sum(size for _, size, _ in entries)
            for _, size, entry in sorted(entries):
                if total <= args.max_bytes:
                    break
                shutil.rmtree(entry, ignore_errors=True)
                total -= size
                removed += 1
                reclaimed += size
        print(json.dumps({"root": str(root), "removed": removed, "reclaimed_bytes": reclaimed},
                         sort_keys=True, separators=(",", ":")))
        return 0
    if not args.model or not args.base or not args.adapter or args.strength is None:
        parser.error("ensure requires MODEL BASE ADAPTER --strength")
    value = ensure_cache(
        args.model, args.base, args.adapter, args.strength,
        role=args.role, cache_dir=args.cache_dir, profile=args.profile,
        hardware=args.hardware,
    )
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
