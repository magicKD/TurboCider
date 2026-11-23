"""Compile and run native MLX test fixtures."""

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


def _mlx_build_flags(
    native: Path | None, *, linked: bool = True,
) -> tuple[list[str], list[str]]:
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
    if linked and native is None:
        native = native_test_directory()
    if native is not None and not (native / "libturbocider.dylib").is_file():
        raise FileNotFoundError("native dylib is unavailable; run make build")
    compiler = subprocess.check_output(
        ["xcrun", "--find", "clang++"], text=True,
    ).strip()
    sdk = subprocess.check_output(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True,
    ).strip()
    sanitizer = "" if linked else os.environ.get("TC_STREAMING_SANITIZER", "")
    if sanitizer not in ("", "address,undefined", "thread"):
        raise ValueError("unsupported sanitizer")
    load_commands = subprocess.check_output(
        ["otool", "-l", str(library / "libmlx.dylib")], text=True,
    )
    minimum = re.search(
        r"cmd LC_BUILD_VERSION.*?\n(?:.*\n)*?\s+minos\s+(\S+)",
        load_commands,
    )
    deployment = minimum.group(1) if minimum else "15.0"
    compile_flags = [compiler, "-std=c++20"]
    if linked:
        compile_flags.append("-O2")
    compile_flags += [
        "-Wall", "-Wextra", "-Werror",
        "-isysroot", sdk, "-mmacosx-version-min=" + deployment,
        "-I", str(ROOT / "native"), "-I", str(ROOT / "native/core"),
        "-isystem", str(include),
    ]
    if sanitizer:
        compile_flags += ["-O1", "-g", "-fno-omit-frame-pointer",
                          "-fsanitize=" + sanitizer]
    elif not linked:
        compile_flags.append("-O2")
    link_flags = ["-L", str(library), "-lmlx", "-ljaccl", "-licucore",
                  "-framework", "Foundation", "-framework", "Metal",
                  "-Wl,-rpath," + str(library)]
    if native is not None:
        link_flags[:0] = ["-L", str(native), "-lturbocider"]
        link_flags.insert(-1, "-Wl,-rpath," + str(native))
    return compile_flags, link_flags


def _metal_unavailable(result: subprocess.CompletedProcess[str]) -> bool:
    combined = result.stdout + result.stderr
    return any(message in combined for message in (
        "No Metal device available", "no Metal device",
        "Failed to get the default Metal device", "Metal is not supported",
    ))


def run_mlx_fixture(
    name: str, title: str, *, native: Path | None = None,
    build_only: bool = False, runtime_skip: bool = False,
) -> None:
    compile_flags, link_flags = _mlx_build_flags(native)
    label = name.replace("_", "-")
    with tempfile.TemporaryDirectory(prefix=f"tc-{label}-") as raw:
        binary = Path(raw) / f"{label}-test"
        subprocess.run([
            *compile_flags, str(ROOT / f"tests/native/{name}_test.cpp"),
            *link_flags, "-o", str(binary),
        ], check=True, cwd=ROOT)
        if build_only:
            print(f"BUILT {title} fixture (GPU execution deferred)")
            return
        result = subprocess.run(
            [str(binary)], text=True, capture_output=True, timeout=120, cwd=ROOT,
        )
        if _metal_unavailable(result):
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


def run_mlx_directory_fixture(
    name: str, title: str, *, native: Path | None,
    sources: tuple[str, ...] = (), create_fixture: bool = False,
) -> None:
    compile_flags, link_flags = _mlx_build_flags(native, linked=native is not None)
    label = title.lower().replace(" ", "-")
    with tempfile.TemporaryDirectory(prefix=f"tc-{label}-") as raw:
        temporary = Path(raw)
        binary = temporary / f"{label}-test"
        subprocess.run([
            *compile_flags, str(ROOT / f"tests/native/{name}_test.cpp"),
            *(str(ROOT / source) for source in sources),
            *link_flags, "-o", str(binary),
        ], check=True)
        fixture = temporary / "fixture"
        if create_fixture:
            fixture.mkdir()
        result = subprocess.run(
            [str(binary), str(fixture)], text=True, capture_output=True, timeout=120,
        )
        if _metal_unavailable(result):
            print(f"SKIP: {title} fixture requires Metal access")
            return
        if result.returncode or not result.stdout.startswith(f"PASS {title}:"):
            print(result.stdout, end="")
            print(result.stderr, end="", file=sys.stderr)
            if result.returncode:
                raise SystemExit(result.returncode)
            raise RuntimeError(f"{title} test exited without its PASS marker")
        print(result.stdout, end="")
