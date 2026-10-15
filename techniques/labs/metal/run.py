#!/usr/bin/env python3
"""Compile/run the isolated Metal learning probe, never a production library."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))
from common import emit, file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("vector", "fusion", "gemm", "norm-rope", "attention", "graph", "overlap"))
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--count", type=int, default=65539)
    parser.add_argument("--m", type=int, default=33)
    parser.add_argument("--n", type=int, default=41)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--steps", type=int, default=16, help="1..32 repeated elementwise stages in graph/overlap")
    parser.add_argument("--diagnostic", choices=("none", "counter-stage", "counter-dispatch", "capture"), default="none")
    parser.add_argument("--capture-output", type=Path, help="fresh ignored .gputrace bundle, only with --diagnostic capture")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("actual Metal probe requires macOS; use labs/run.py for CPU references")
    if args.output and (args.output.exists() or args.output.is_symlink()):
        parser.error("output already exists; use a new report path")
    if (args.diagnostic == "capture") != (args.capture_output is not None):
        parser.error("--diagnostic capture requires --capture-output; other modes reject it")
    if args.capture_output:
        if args.capture_output.exists() or args.capture_output.is_symlink():
            parser.error("capture output already exists")
        args.capture_output.parent.mkdir(parents=True, exist_ok=True)
        if args.output and args.output.resolve().is_relative_to(args.capture_output.resolve()):
            parser.error("report cannot be inside the capture bundle")
        if shutil.disk_usage(args.capture_output.parent).free < 512 * 1024 * 1024:
            parser.error("capture requires at least 512 MiB of free disk headroom")
    root = Path(__file__).resolve().parent
    source_names = ("bench.mm", "kernels.metal", "run.py", "mlx_capture_probe.py")
    source_hashes = {name: file_sha(root / name) for name in source_names}
    snapshots = []
    if args.output:
        snapshot_root = args.output.parent / (args.output.stem + "-source")
        snapshot_root.mkdir(parents=True, exist_ok=False)
        for name in source_names:
            shutil.copy2(root / name, snapshot_root / name)
            snapshots.append({"path": str((snapshot_root / name).relative_to(args.output.parent)),
                              "sha256": file_sha(snapshot_root / name)})
    with tempfile.TemporaryDirectory(prefix="tc-technique-metal-") as directory:
        binary = Path(directory) / "metal-lab"
        command = ["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-Wall", "-Wextra",
                   str(root / "bench.mm"), "-framework", "Foundation", "-framework", "Metal", "-o", str(binary)]
        built = subprocess.run(command, text=True, capture_output=True, timeout=120)
        if built.returncode:
            emit({"status": "build_failed", "stderr": built.stderr, "production_qualified": False}, args.output)
            return built.returncode
        environment = os.environ.copy()
        if args.diagnostic == "capture":
            environment["MTL_CAPTURE_ENABLED"] = "1"
        result = subprocess.run([str(binary), str(root / "kernels.metal"), args.case, str(args.repeats), str(args.count),
                                 str(args.m), str(args.n), str(args.k), str(args.threads), str(args.seed), args.diagnostic,
                                 str(args.capture_output.resolve()) if args.capture_output else "-", str(args.steps)],
                                text=True, capture_output=True, timeout=120, env=environment)
        try:
            value = json.loads(result.stdout)
        except ValueError:
            value = {"status": "invalid_native_report", "stdout": result.stdout}
        value.update(arguments={name: getattr(args, name) for name in ("case", "repeats", "count", "m", "n", "k", "threads", "seed", "steps")},
                     variant_threads={"baseline": args.threads, "candidate": 32 if args.case == "gemm" else args.threads},
                     binary_sha256=file_sha(binary), source_sha256=source_hashes, source_snapshots=snapshots,
                     source_inputs_unchanged=source_hashes == {name: file_sha(root / name) for name in source_names},
                     os=platform.mac_ver()[0], build_warnings=built.stderr, stderr=result.stderr, exit_code=result.returncode)
        if args.diagnostic == "capture" and result.returncode == 0:
            from mlx_capture_probe import capture_artifacts
            value["capture_artifacts"] = capture_artifacts(args.capture_output.resolve(), args.capture_output.resolve().parent)
        if not value["source_inputs_unchanged"]:
            value.update(status="source_changed_during_run", production_qualified=False)
            emit(value, args.output)
            return 3
        emit(value, args.output)
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
