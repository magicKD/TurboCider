#!/usr/bin/env python3
"""Opt-in real checkpoint/Metal integration test. Never applies memory pressure."""
import argparse
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--upsampler", required=True, type=Path)
    parser.add_argument("--video-vae", required=True, type=Path)
    parser.add_argument("--slots", type=int, choices=(1, 2, 3), default=3)
    # The legacy comparison path intentionally exercises V1/V2.  The split
    # Stage-1/Stage-2 V3 ABI is exercised by run_split below and does not
    # replace the historical single-stage comparison selector.
    parser.add_argument("--exact-api", type=int, choices=(1, 2), default=2)
    parser.add_argument("--conditioning", choices=("synthetic", "connector"), default="synthetic")
    args = parser.parse_args()
    for path in (args.checkpoint, args.upsampler, args.video_vae):
        if not path.is_file():
            parser.error(f"missing checkpoint: {path}")
    compiler = subprocess.check_output(["xcrun", "--find", "clang++"], text=True).strip()
    sdk = subprocess.check_output(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True
    ).strip()
    with tempfile.TemporaryDirectory(prefix="tc-ltx-model-") as raw:
        binary = Path(raw) / "test"
        subprocess.run([compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-isysroot", sdk,
                        "-I", str(ROOT / "native/runtime"), "-I", str(ROOT / "native/models/ltx_runtime"),
                        str(ROOT / "tests/native/ltx_streaming_model_test.cpp"),
                        "-L", str(ROOT / "build/native"), "-lturbocider", "-Wl,-rpath," + str(ROOT / "build/native"),
                        "-o", str(binary)], check=True)
        subprocess.run([str(binary), str(args.checkpoint.resolve()),
                        str(ROOT / "native/models/ltx_runtime/ltx_shaders.metal"),
                        str(args.upsampler.resolve()), str(args.video_vae.resolve()),
                        str(args.slots), str(args.exact_api), args.conditioning], check=True, timeout=600)


if __name__ == "__main__":
    main()
