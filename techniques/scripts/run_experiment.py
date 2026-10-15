#!/usr/bin/env python3
"""Explicit owned-process experiment supervisor; never invoked by retrieval.

Use a fresh ignored output directory. Child stdout/stderr and failure receipts
survive timeouts; only the process group created by this invocation is stopped.
No environment dump, system cache change, implicit dependency or shell command.
"""
import argparse
import hashlib
import json
from pathlib import Path
import os
import signal
import subprocess
import time


def terminate_owned(process):
    for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            return process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
    return None


def run(command, output, timeout):
    if not command or not all(isinstance(value, str) and value for value in command):
        raise ValueError("explicit nonempty argument vector required")
    if type(timeout) is not int or not 1 <= timeout <= 900:
        raise ValueError("timeout must be 1..900 seconds")
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "tc-owned-experiment-v1", "status": "starting", "command": command,
              "timeout_seconds": timeout, "started_monotonic_ns": time.monotonic_ns(),
              "supervisor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "production_qualified": False, "physical_device_drain": "not_observed; no buffers reused across child processes"}

    def save():
        (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    process = None
    try:
        with (output / "stdout.log").open("x") as stdout, (output / "stderr.log").open("x") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
            report.update(status="running", owned_pid=process.pid)
            save()
            try:
                report["returncode"] = process.wait(timeout=timeout)
                report["status"] = "succeeded" if report["returncode"] == 0 else "failed"
            except subprocess.TimeoutExpired:
                report.update(status="timed_out", returncode=terminate_owned(process))
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error=type(error).__name__ + ": " + str(error))
        if process is not None:
            report["returncode"] = terminate_owned(process)
    report["finished_monotonic_ns"] = time.monotonic_ns()
    report["owned_process_exit_confirmed"] = process is not None and process.poll() is not None
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        report = run(command, args.output_dir.resolve(), args.timeout)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))
    return int(report["status"] != "succeeded")


if __name__ == "__main__":
    raise SystemExit(main())
