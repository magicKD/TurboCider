#!/usr/bin/env python3
"""Build the zero-copy Core ML bridge for the active Python interpreter."""

from __future__ import annotations

import argparse
import subprocess
import sys
import sysconfig
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=root / "build")
    parser.add_argument("--xcode", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    include = Path(str(sysconfig.get_config_var("INCLUDEPY") or ""))
    fallback_include = Path(sys.base_prefix) / "include" / (
        "python%d.%d" % (sys.version_info.major, sys.version_info.minor)
    )
    if not (include / "Python.h").is_file() and (fallback_include / "Python.h").is_file():
        include = fallback_include
    if not suffix or not include:
        raise RuntimeError("active Python does not expose extension build metadata")
    output = args.output_dir / f"_flux2_ane_bridge{suffix}"
    if args.xcode:
        compiler = args.xcode / "Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++"
        sdk = args.xcode / "Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
    else:
        compiler = Path(
            subprocess.check_output(["xcrun", "--find", "clang++"], text=True).strip()
        )
        sdk = Path(
            subprocess.check_output(
                ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True
            ).strip()
        )
    command = [
        str(compiler),
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        "-fobjc-arc",
        "-Wall",
        "-Wextra",
        "-Wno-missing-field-initializers",
        "-isysroot",
        str(sdk),
        "-mmacosx-version-min=15.0",
        "-I",
        str(include),
        "-bundle",
        "-undefined",
        "dynamic_lookup",
        str(root / "native" / "flux2_ane_bridge.mm"),
        "-o",
        str(output),
        "-framework",
        "Foundation",
        "-framework",
        "CoreML",
    ]
    subprocess.run(command, check=True)
    print(output.resolve())


if __name__ == "__main__":
    main()
