#!/usr/bin/env python3
"""Fingerprint native build inputs; verify they stayed fixed during compilation.

This is a compatibility key, not a signature, artifact proof, or binary hash.
Apple SDK identity uses SDKSettings.json; the entire Apple SDK is not hashed.
Final App/bundle and runtime dependency verification remain packaging concerns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

DOMAIN = "tc-runtime-build-v1"
SOURCE_SUFFIXES = {".h", ".hpp", ".c", ".cpp", ".m", ".mm", ".metal", ".inc"}
BUILD_SCRIPTS = (
    "tools/native/build.sh", "tools/native/dependencies.sh",
    "tools/native/generate_runtime_build_identity.py",
    "tools/native/generate_bundled_streaming_catalog.py",
    "tools/native/build_streaming_catalog.py",
    "tools/native/verify_streaming_campaign.py",
)
SEARCH_ENV = (
    "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH",
    "LIBRARY_PATH", "COMPILER_PATH",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entries(root: Path, paths: list[Path]) -> list[dict]:
    return [{"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
            for path in sorted(set(paths))]


def source_inputs(root: Path) -> list[dict]:
    paths = []
    # Include native CLI/service glue as well as all library source/headers and
    # runtime shaders. Exclude docs/tests/build outputs and Swift frontend code.
    for directory in ("native", "bindings/c", "apps/cli", "services/turbociderd"):
        folder = root / directory
        if not folder.is_dir():
            raise ValueError(f"missing source directory: {directory}")
        paths.extend(p for p in folder.rglob("*")
                     if p.is_file() and p.suffix in SOURCE_SUFFIXES)
    paths.extend(root / name for name in BUILD_SCRIPTS)
    return entries(root, paths)


def mlx_inputs(root: Path) -> list[dict]:
    required = [root / "include/mlx/mlx.h", root / "lib/libmlx.dylib",
                root / "lib/libjaccl.dylib", root / "lib/mlx.metallib"]
    for path in required:
        if not path.is_file():
            raise ValueError(f"missing MLX build dependency: {path}")
    return entries(root, [p for p in (root / "include").rglob("*") if p.is_file()]
                   + required[1:])


def seal(inputs: dict) -> dict:
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True).encode()
    digest = hashlib.sha256(DOMAIN.encode() + b"\0" + canonical).hexdigest()
    return {"schema": DOMAIN, "runtime_build_id": f"{DOMAIN}-{digest}",
            "inputs": inputs}


def manifest(args: argparse.Namespace) -> dict:
    active = [name for name in SEARCH_ENV if os.environ.get(name)]
    if active:
        raise ValueError("untracked compiler search environment: " + ", ".join(active))
    toolchain = {}
    for name in ("clang", "clang++", "ld", "ar"):
        tool = args.toolchain / name
        if not tool.is_file():
            raise ValueError(f"missing build tool: {tool}")
        toolchain[name] = sha256_file(tool)
    compiler = args.toolchain / "clang++"
    version = subprocess.check_output([str(compiler), "--version"], text=True)
    # InstalledDir is location metadata, not compiler behavior.
    version = "\n".join(line for line in version.splitlines()
                        if not line.startswith("InstalledDir:"))
    replacements = [(str(args.root), "<SOURCE>"), (str(args.mlx_root), "<MLX>"),
                    (str(args.sdk), "<SDK>"), (str(args.output), "<OUTPUT>")]
    # Replace longer paths first (MLX/output commonly live below source).
    def normalize(value: str) -> str:
        for path, marker in sorted(replacements, key=lambda item: -len(item[0])):
            value = value.replace(path, marker)
        return value
    flags = args.flags[1:] if args.flags[:1] == ["--"] else args.flags
    return seal({
        "sources": source_inputs(args.root), "mlx": mlx_inputs(args.mlx_root),
        "toolchain": toolchain, "compiler_version": version,
        "sdk_settings_sha256": sha256_file(args.sdk / "SDKSettings.json"),
        "host_architecture": platform.machine(),
        "policy": {"deployment_target": args.deployment_target,
                   "test_hooks": args.test_hooks, "audit_counters": args.audit_counters,
                   "experimental_probes": args.experimental_probes,
                   "common_flags": [normalize(flag) for flag in flags]},
    })


def verify_existing(path: Path, expected: dict) -> None:
    if json.loads(path.read_text()) != expected:
        raise ValueError("native build inputs changed during compilation; rebuild required")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "mlx-root", "sdk", "toolchain", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--deployment-target", required=True)
    for name in ("test-hooks", "audit-counters", "experimental-probes"):
        parser.add_argument("--" + name, choices=("0", "1"), required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("flags", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        value = manifest(args)
        path = args.output / "runtime-build-manifest.json"
        if args.verify:
            verify_existing(path, value)
        else:
            args.output.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
            (args.output / "turbocider_runtime_build_generated.hpp").write_text(
                "#pragma once\n#define TURBOCIDER_RUNTIME_BUILD_ID "
                + json.dumps(value["runtime_build_id"]) + "\n")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"runtime build identity: {error}\n")


if __name__ == "__main__":
    main()
