"""Opt-in, independent memory evidence for resident GPU/ANE comparisons.

The sample scope includes load, cold/warm requests and process exit. It is not
per-request GPU/ANE memory attribution. No model work occurs on import.
"""
import json
import os
import signal
import subprocess
import sys
import threading
import uuid

# Keep the existing public helper while sharing the host-only implementation.
from runtime_ane_common import sha256_file as sha256


def signal_owned_group(pid, signum):
    # Only called for Popen(start_new_session=True) created below. Never
    # resolve a target by executable name or touch competing inference.
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass


def run_owned(command, **kwargs):
    """Keep a sampler timeout/interrupt from orphaning its inference child."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("owned memory sampling requires the main thread for signal cleanup")
    timeout = kwargs.pop("timeout")
    process = subprocess.Popen(command, start_new_session=True, **kwargs)

    def interrupted(signum, _frame):
        raise InterruptedError(f"memory sampling interrupted by signal {signum}")

    previous_term = signal.signal(signal.SIGTERM, interrupted)
    try:
        returncode = process.wait(timeout=timeout)
    except BaseException as error:
        try:
            signal_owned_group(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            # The wrapper can exit while a descendant ignores TERM. Kill
            # any remaining members of this owned group, then reap it.
            signal_owned_group(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        except BaseException as cleanup_error:
            error.add_note(f"owned sampler group cleanup failed: {cleanup_error}")
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    if returncode:
        # A crashed/killed wrapper might not have reached its child cleanup.
        signal_owned_group(process.pid, signal.SIGKILL)
    return subprocess.CompletedProcess(command, returncode)


def run_sampled(command, *, repo, output, stem, env, stdout, stderr, timeout,
                interval_ms, max_gap_ms):
    """Run then independently verify; incomplete evidence never returns success."""
    raw = output / f"{stem}-memory.jsonl"
    report = output / f"{stem}-memory-report.json"
    sampler = repo / "tools/native/process_tree_sampler.py"
    verifier = repo / "tools/native/verify_process_tree_samples.py"
    sources = {str(path.relative_to(repo)): sha256(path) for path in (sampler, verifier)}
    correlation = uuid.uuid4().hex
    wrapped = [sys.executable, str(sampler), "--launch", "--output", str(raw.resolve()),
               "--correlation-id", correlation, "--root-role", "inference",
               "--interval-ms", str(interval_ms), "--max-gap-ms", str(max_gap_ms),
               "--", *command]
    result = run_owned(wrapped, cwd=repo, env=env, stdout=stdout, stderr=stderr, timeout=timeout)
    verification = subprocess.run(
        [sys.executable, str(verifier), str(raw.resolve()), "--output", str(report.resolve())],
        cwd=repo, env=env, stdout=stderr, stderr=stderr, timeout=60)
    # Keep raw JSONL/report even for an inconclusive sample or failed command.
    if result.returncode or verification.returncode:
        raise RuntimeError(f"memory sampling/verification failed ({result.returncode}/"
                           f"{verification.returncode}); inspect {stem} evidence")
    verified = json.loads(report.read_text())
    if (verified.get("complete") is not True or verified.get("correlation_id") != correlation or
            type(verified.get("command_exit_code")) is not int or
            verified["command_exit_code"] != 0 or
            verified.get("allowed_max_gap_ns") != max_gap_ms * 1_000_000):
        raise ValueError("memory report is incomplete or does not match this trial")
    if any(sha256(repo / name) != digest for name, digest in sources.items()):
        raise ValueError("memory sampler/verifier changed during the trial")
    return {
        "scope": "process tree; load+cold+warm+exit; excludes external services/driver attribution",
        "correlation_id": correlation, "interval_ms": interval_ms, "max_gap_ms": max_gap_ms,
        "evidence": raw.name, "evidence_sha256": sha256(raw),
        "report": report.name, "report_sha256": sha256(report),
        "tool_sha256": sources, "verified": verified,
    }
