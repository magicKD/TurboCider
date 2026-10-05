#!/usr/bin/env python3
"""Serial component screen with continuous host-load evidence, not E2E qualification.

Build first, then run this wrapper. The child must serialize its own timing
arms. Observation includes its process tree, but is not a physical GPU trace.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def observe():
    started = time.monotonic()
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,%cpu=,comm="], capture_output=True,
        text=True, check=True, timeout=5)
    return {"monotonic": started, "load_average": list(os.getloadavg()),
            "processes": result.stdout.splitlines(),
            "observation_seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=.5)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not .1 <= args.interval <= 2 or not 1 <= args.timeout <= 3600:
        parser.error("provide command, interval .1..2 and timeout 1..3600")
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists")
    binary = Path(command[0]).resolve(strict=True)
    if not binary.is_file():
        parser.error("command must name a binary file")
    with binary.open("rb") as binary_file:
        binary_sha256 = hashlib.file_digest(binary_file, "sha256").hexdigest()
    receipt = {"schema": "tc-observed-gpu-component-screen-v1",
               "time_utc": datetime.now(timezone.utc).isoformat(),
               "command": command, "binary_sha256": binary_sha256,
               "interval_seconds": args.interval, "observations": [], "observation_errors": [],
               "scope": "host-span component diagnostic, observer included; not E2E or physical GPU/ANE trace"}
    stop = threading.Event()

    def sample():
        while True:
            try:
                receipt["observations"].append(observe())
            except (OSError, subprocess.SubprocessError) as error:
                receipt["observation_errors"].append(str(error))
            if stop.wait(args.interval):
                break

    with open("/tmp/turbocider-z-image-gpu-benchmark.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt["observations"].append(observe())
        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        start = time.monotonic()
        try:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as child:
                receipt["pid"] = child.pid
                try:
                    stdout, stderr = child.communicate(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    # Only this wrapper's direct child; never kill by process name.
                    child.kill()
                    stdout, stderr = child.communicate()
                    receipt["timed_out"] = True
                receipt.update(exit_code=child.returncode, stdout=stdout, stderr=stderr)
        finally:
            stop.set()
            sampler.join(timeout=6)
        receipt["wall_seconds"] = time.monotonic() - start
        receipt["observations"].append(observe())
    times = [row["monotonic"] for row in receipt["observations"]]
    receipt["maximum_observation_gap_seconds"] = max((b-a for a,b in zip(times,times[1:])), default=0)
    receipt["observation_thread_joined"] = not sampler.is_alive()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(receipt, output, indent=2, allow_nan=False)
        output.write("\n")
    print(receipt.get("stdout", ""), end="")
    print(receipt.get("stderr", ""), end="")
    print(f"evidence: {args.output}")
    raise SystemExit(receipt.get("exit_code", 1) or bool(receipt["observation_errors"]) or not receipt["observation_thread_joined"])


if __name__ == "__main__":
    main()
