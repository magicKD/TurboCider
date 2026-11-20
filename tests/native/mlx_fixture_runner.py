"""Compile and run the linked Qwen21 MLX test fixtures."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def native_test_directory() -> Path:
    return Path(os.environ.get("TURBOCIDER_TEST_NATIVE_DIR", ROOT / "build/native")).resolve()


def run_mlx_fixture(
    name: str, title: str, *, native: Path | None = None,
    build_only: bool = False, runtime_skip: bool = False,
) -> None:
    mlx_root = Path(os.environ.get(
        "MLX_ROOT", str(Path(sysconfig.get_paths()["purelib"]) / "mlx"),
    ))
    include = mlx_root / "include"
    library = mlx_root / "lib"
    if not (include / "mlx/mlx.h").is_file() or not (
            library / "libmlx.dylib").is_file():
        raise FileNotFoundError(
            "MLX C++ headers/libraries are unavailable; run make setup or "
            "set MLX_ROOT"
        )
    if native is None:
        native = native_test_directory()
    if not (native / "libturbocider.dylib").is_file():
        raise FileNotFoundError("native dylib is unavailable; run make build")
    compiler = subprocess.check_output(
        ["xcrun", "--find", "clang++"], text=True,
    ).strip()
    sdk = subprocess.check_output(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True,
    ).strip()
    load_commands = subprocess.check_output(
        ["otool", "-l", str(library / "libmlx.dylib")], text=True,
    )
    minimum = re.search(
        r"cmd LC_BUILD_VERSION.*?\n(?:.*\n)*?\s+minos\s+(\S+)",
        load_commands,
    )
    deployment = minimum.group(1) if minimum else "15.0"
    label = name.replace("_", "-")
    with tempfile.TemporaryDirectory(prefix=f"tc-{label}-") as raw:
        binary = Path(raw) / f"{label}-test"
        subprocess.run([
            compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
            "-isysroot", sdk, "-mmacosx-version-min=" + deployment,
            "-I", str(ROOT / "native"),
            "-I", str(ROOT / "native/core"),
            "-isystem", str(include),
            str(ROOT / f"tests/native/{name}_test.cpp"),
            "-L", str(native), "-lturbocider",
            "-L", str(library), "-lmlx", "-ljaccl", "-licucore",
            "-framework", "Foundation", "-framework", "Metal",
            "-Wl,-rpath," + str(native),
            "-Wl,-rpath," + str(library),
            "-o", str(binary),
        ], check=True, cwd=ROOT)
        if build_only:
            print(f"BUILT {title} fixture (GPU execution deferred)")
            return
        result = subprocess.run(
            [str(binary)], text=True, capture_output=True, timeout=120, cwd=ROOT,
        )
        combined = result.stdout + result.stderr
        if any(message in combined for message in (
                "No Metal device available", "no Metal device",
                "Failed to get the default Metal device", "Metal is not supported")):
            raise unittest.SkipTest(f"{title} fixture requires Metal access")
        if result.returncode:
            print(result.stdout, end="")
            print(result.stderr, end="", file=sys.stderr)
            raise RuntimeError(f"{title} fixture failed: {result.returncode}")
        if runtime_skip and result.stdout.startswith("SKIP "):
            raise unittest.SkipTest(result.stdout.strip())
        if not result.stdout.startswith(f"PASS {title}:"):
            print(result.stdout, end="")
            print(result.stderr, end="", file=sys.stderr)
            raise RuntimeError(f"{title} test exited without its PASS marker")
        print(result.stdout, end="")
