#!/usr/bin/env python3
"""Isolated synthetic ConvRot FFN comparison; not an end-to-end model test."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def command(args):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {args!r}\n"
                           f"{result.stdout}\n{result.stderr}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1056, 4128])
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True,
                        help="new JSON receipt path; existing files are never replaced")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists")
    if not 4 <= args.iterations <= 100 or any(not 1 <= row <= 8192 for row in args.rows):
        parser.error("iterations must be 4..100 and rows 1..8192")
    # Shared with other Z GPU diagnostics; a busy device run is not queued.
    with open("/tmp/turbocider-z-image-gpu-benchmark.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another Z-image GPU diagnostic holds the benchmark lock")
        command(["bash", "tools/native/build_convrot_ffn_probe.sh"])
        binary = ROOT / "build/native/convrot-ffn-probe"
        sources = ["native/models/z_image/ffn.hpp", "native/backends/mlx.cpp",
                   "tools/native/convrot_ffn_probe.cpp"]
        receipt = {
            "schema_version": 1,
            "kind": "synthetic_convrot_ffn_shared_rotation",
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "os": platform.platform(),
            "hardware": command(["sysctl", "-n", "machdep.cpu.brand_string"]).strip(),
            "base_commit": command(["git", "rev-parse", "HEAD"]).strip(),
            "library_sha256": digest(ROOT / "build/native/libturbocider.dylib"),
            "probe_sha256": digest(binary),
            "source_sha256": {name: digest(ROOT / name) for name in sources},
            "scope": "production inline helper + existing Weights primitives; synthetic inputs/weights",
            "excludes": ["weight load/packing", "attention", "full request", "image quality",
                         "low-memory certification", "ANE", "M5 qualification"],
            "parity": json.loads(command([str(binary)])),
            "samples": [],
        }
        for rows in args.rows:
            receipt["samples"].append(json.loads(command(
                [str(binary), str(rows), "3840", "10240", str(args.iterations)])))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
