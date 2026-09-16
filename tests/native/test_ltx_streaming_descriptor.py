#!/usr/bin/env python3
"""Real LTX metadata projection: no GPU, no full-weight materialization."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
from test_ltx_streaming_layout import fixture

ROOT = Path(__file__).resolve().parents[2]
LTX = ROOT / "native/models/ltx_runtime"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    cc = subprocess.check_output(["xcrun", "--find", "clang"], text=True).strip()
    cxx = subprocess.check_output(["xcrun", "--find", "clang++"], text=True).strip()
    sdk = subprocess.check_output(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True
    ).strip()
    flags = ["-Wall", "-Wextra", "-Werror", "-isysroot", sdk,
             "-I", str(LTX)]
    sanitizer = os.environ.get("TC_STREAMING_SANITIZER", "")
    if sanitizer:
        if sanitizer not in ("address,undefined", "thread"):
            raise ValueError("unsupported sanitizer")
        flags += ["-O1", "-g", "-fno-omit-frame-pointer", "-fsanitize=" + sanitizer]
    else:
        flags += ["-O2"]
    sources = ["native/runtime/streaming/config.cpp", "native/runtime/streaming/layout.cpp",
               "native/runtime/memory_manifest.cpp", "native/runtime/memory_policy.cpp",
               "native/core/common.cpp", "native/models/ltx_runtime/ltx_streaming_descriptor.cpp",
               "native/models/ltx_runtime/ltx_streaming_plan.cpp"]
    with tempfile.TemporaryDirectory(prefix="tc-ltx-descriptor-") as raw:
        directory = Path(raw)
        checkpoint = directory / "fixture.safetensors"
        fixture(checkpoint, blocks=48)
        objects = []
        for name in ("ltx.c", "ltx_streaming_layout.c", "ltx_safetensors.m", "ltx_weights.m"):
            source = LTX / name
            target = directory / (source.stem + ".o")
            subprocess.run([cc, "-std=c11", *flags, *(["-fobjc-arc"] if source.suffix == ".m" else []),
                            "-c", str(source), "-o", str(target)], check=True)
            objects.append(str(target))
        binary = directory / "descriptor-test"
        subprocess.run([cxx, "-std=c++20", *flags,
                        str(ROOT / "tests/native/ltx_streaming_descriptor_test.cpp"),
                        *[str(ROOT / source) for source in sources], *objects,
                        "-framework", "Foundation", "-o", str(binary)], check=True)
        subprocess.run([str(binary), str(checkpoint), "--mutable-fixture"], check=True, timeout=60)
        if args.checkpoint:
            subprocess.run([str(binary), str(args.checkpoint.resolve())], check=True, timeout=120)


if __name__ == "__main__":
    main()
