#!/usr/bin/env python3
"""Fail-closed GPU-idle admission for a bounded, explicitly selected experiment.

Only IORegistry utilization is collected: no process args, addresses or IDs.
This is a coarse preflight, not proof of per-process GPU exclusivity.
"""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import plistlib
import subprocess
import sys
import time


def utilization(nodes):
    """Handle ioreg's nested plist; missing/invalid counters must fail closed."""
    found = []
    def visit(value):
        if isinstance(value, dict):
            stats = value.get("PerformanceStatistics")
            if isinstance(stats, dict) and "Device Utilization %" in stats:
                percent = stats["Device Utilization %"]
                if (isinstance(percent, bool) or not isinstance(percent, (int, float))
                        or not math.isfinite(percent) or not 0 <= percent <= 100):
                    raise ValueError("invalid GPU utilization counter")
                found.append(float(percent))
            for child in value.values():
                if isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(nodes)
    if not found:
        raise ValueError("GPU utilization unavailable; experiment not admitted")
    return max(found)


def sample():
    result = subprocess.run(["ioreg", "-a", "-r", "-d", "1", "-c", "IOAccelerator"],
                            capture_output=True, timeout=10, check=True)
    return utilization(plistlib.loads(result.stdout))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 1 <= args.timeout <= 300:
        parser.error("explicit command and timeout 1..300 seconds required")
    output = args.output_dir
    if output.exists() or output.is_symlink():
        parser.error("use a new output directory")
    output.mkdir(parents=True)
    report = {"schema": "gpu-ane-idle-admission-v1", "status": "checking",
              "threshold_percent": 5, "preflight_percent": [], "during_percent": [],
              "postflight_percent": None, "command": command, "returncode": None,
              "physical_overlap_proven": False, "production_qualified": False,
              "limitation": "coarse total utilization, not per-process attribution or exclusion of future background activity",
              "guard_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def save():
        (output / "admission.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    # Shared by this tool's invocations, not a lock on other applications.
    with (output.parent / ".gpu-ane-survey.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            report["status"] = "refused_owned_suite_busy"
            save()
            return 2
        try:
            for index in range(5):
                report["preflight_percent"].append(sample())
                save()
                if report["preflight_percent"][-1] >= report["threshold_percent"]:
                    report["status"] = "deferred_gpu_busy"
                    save()
                    return 2
                if index < 4:
                    time.sleep(1)
            report["status"] = "admitted"
            save()
            # Existing supervisor owns timeout, child process group and logs.
            supervisor = Path(__file__).resolve().parents[1] / "scripts/run_experiment.py"
            started = time.monotonic()
            runner = subprocess.Popen([sys.executable, str(supervisor), "--timeout", str(args.timeout),
                "--output-dir", str(output / "child"), "--", *command],
                stdout=subprocess.DEVNULL)
            while runner.poll() is None:
                try:
                    report["during_percent"].append(sample())
                except (ValueError, OSError, subprocess.SubprocessError, plistlib.InvalidFileException):
                    report["during_percent"].append(None)
                time.sleep(1)
            report.update(returncode=runner.returncode,
                status="finished" if runner.returncode == 0 else "child_failed",
                elapsed_seconds=time.monotonic() - started)
            report["postflight_percent"] = sample()
        except (ValueError, OSError, subprocess.SubprocessError, plistlib.InvalidFileException) as error:
            report.update(status="admission_or_observation_failed", error=type(error).__name__ + ": " + str(error))
        finally:
            save()
    print(json.dumps({k: report[k] for k in ("status", "preflight_percent", "returncode")}))
    return 0 if report["status"] == "finished" else 2


if __name__ == "__main__":
    raise SystemExit(main())
