"""Cancellation-aware engine process runner."""

from __future__ import annotations

import os
import re
import selectors
import signal
import subprocess
import threading
import time
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from turbocider.adapters.base import CommandSpec


ProgressCallback = Callable[[str, float, str], None]


_PROGRESS_PATTERNS = (
    re.compile(
        r"^\s*([A-Za-z][A-Za-z0-9_. -]*?)\s+(\d+)\s*/\s*(\d+)",
        re.I,
    ),
    re.compile(r"(?:phase|stage)[=: ]+([A-Za-z0-9_.-]+).*?(\d+)\s*/\s*(\d+)", re.I),
    re.compile(r"([A-Za-z0-9_.-]+).*?(\d+)\s+of\s+(\d+)", re.I),
)


class ProcessResult:
    def __init__(
        self,
        return_code: int,
        wall_seconds: float,
        log_path: Path,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        self.return_code = return_code
        self.wall_seconds = wall_seconds
        self.log_path = log_path
        self.metrics = metrics or {}


class PersistentProcess:
    """One line-delimited worker retained across heavyweight generation jobs."""

    def __init__(
        self,
        argv,
        cwd: Path,
        environment: Dict[str, str],
    ):
        self.argv = list(argv)
        self.cwd = cwd
        self.environment = environment
        self.lock = threading.Lock()
        self.process: Optional[subprocess.Popen] = None

    def _start(self) -> subprocess.Popen:
        if self.process is not None and self.process.poll() is None:
            return self.process
        if self.process is not None:
            self._close_streams(self.process)
        self.process = subprocess.Popen(
            self.argv,
            cwd=str(self.cwd),
            env=self.environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        return self.process

    def run(
        self,
        payload: Dict[str, Any],
        log,
        cancel_event: threading.Event,
        progress: Optional[ProgressCallback],
    ) -> tuple[int, Dict[str, Any]]:
        with self.lock:
            process = self._start()
            assert process.stdin is not None
            assert process.stdout is not None
            assert process.stderr is not None
            process.stdin.write(json.dumps(payload) + "\n")
            process.stdin.flush()
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            response: Optional[Dict[str, Any]] = None
            try:
                while response is None:
                    if cancel_event.is_set() and process.poll() is None:
                        self.terminate()
                    for key, _ in selector.select(timeout=0.25):
                        line = key.fileobj.readline()
                        if not line:
                            continue
                        line = line.rstrip("\n")
                        log.write(line + "\n")
                        log.flush()
                        if key.data == "stdout" and line.startswith(
                            "turbocider_worker_result="
                        ):
                            response = json.loads(line.split("=", 1)[1])
                        elif progress:
                            parsed = ProcessRunner._parse_progress(line)
                            if parsed:
                                progress(parsed[0], parsed[1], line)
                            else:
                                progress("", -1.0, line)
                    if process.poll() is not None and response is None:
                        return int(process.returncode or 1), {
                            "worker_error": "persistent worker exited before responding"
                        }
            finally:
                selector.close()
            if not response.get("ok", False):
                return 1, {"worker_error": str(response.get("error", "worker failed"))}
            metrics = dict(response.get("metrics", {}))
            metrics["persistent_worker_pid"] = process.pid
            metrics["persistent_worker_reused"] = bool(response.get("worker_request_index", 1) > 1)
            return 0, metrics

    def terminate(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if process is not None:
            self._close_streams(process)
        self.process = None

    @staticmethod
    def _close_streams(process: subprocess.Popen) -> None:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass


class ProcessRunner:
    def __init__(self):
        self.persistent: Dict[str, PersistentProcess] = {}
        self.persistent_lock = threading.Lock()

    def run(
        self,
        spec: CommandSpec,
        log_path: Path,
        cancel_event: threading.Event,
        progress: Optional[ProgressCallback] = None,
    ) -> ProcessResult:
        environment: Dict[str, str] = os.environ.copy()
        for prefix in spec.unset_environment_prefixes:
            for key in list(environment):
                if key.startswith(prefix):
                    environment.pop(key, None)
        environment.update(spec.environment)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        persistent = spec.metadata.get("persistent_worker")
        if persistent:
            return self._run_persistent(
                spec,
                log_path,
                environment,
                cancel_event,
                progress,
                persistent,
                started,
            )
        with log_path.open("w", encoding="utf-8") as log:
            log.write("cwd=%s\n" % spec.cwd)
            log.write("argv=%r\n" % spec.argv)
            log.flush()
            process = subprocess.Popen(
                spec.argv,
                cwd=str(spec.cwd),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            assert process.stdout is not None
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            terminate_started: Optional[float] = None
            result_metrics: Dict[str, Any] = {}
            while True:
                if cancel_event.is_set() and process.poll() is None:
                    if terminate_started is None:
                        terminate_started = time.monotonic()
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    elif time.monotonic() - terminate_started >= 5.0:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                events = selector.select(timeout=0.25)
                for key, _ in events:
                    line = key.fileobj.readline()
                    if line:
                        log.write(line)
                        log.flush()
                        self._update_result_metrics(line, result_metrics)
                        if progress:
                            parsed = self._parse_progress(line)
                            if parsed:
                                progress(parsed[0], parsed[1], line.rstrip())
                            else:
                                progress("", -1.0, line.rstrip())
                if process.poll() is not None:
                    remainder = process.stdout.read()
                    if remainder:
                        log.write(remainder)
                        log.flush()
                        for line in remainder.splitlines():
                            self._update_result_metrics(line, result_metrics)
                    break
            selector.close()
            process.stdout.close()
            return ProcessResult(
                return_code=int(process.returncode or 0),
                wall_seconds=time.perf_counter() - started,
                log_path=log_path,
                metrics=result_metrics,
            )

    def _run_persistent(
        self,
        spec: CommandSpec,
        log_path: Path,
        environment: Dict[str, str],
        cancel_event: threading.Event,
        progress: Optional[ProgressCallback],
        persistent: Dict[str, Any],
        started: float,
    ) -> ProcessResult:
        key = str(persistent["key"])
        with self.persistent_lock:
            worker = self.persistent.get(key)
            if worker is None:
                worker = PersistentProcess(
                    persistent["argv"],
                    Path(str(persistent.get("cwd", spec.cwd))),
                    environment,
                )
                self.persistent[key] = worker
        with log_path.open("w", encoding="utf-8") as log:
            log.write("cwd=%s\n" % spec.cwd)
            log.write("argv=%r\n" % spec.argv)
            log.write("persistent_worker_key=%s\n" % key)
            log.flush()
            return_code, metrics = worker.run(
                dict(persistent["request"]), log, cancel_event, progress
            )
        if return_code != 0:
            with self.persistent_lock:
                if self.persistent.get(key) is worker:
                    self.persistent.pop(key, None)
            worker.terminate()
        return ProcessResult(
            return_code=return_code,
            wall_seconds=time.perf_counter() - started,
            log_path=log_path,
            metrics=metrics,
        )

    def close(self) -> None:
        with self.persistent_lock:
            workers = list(self.persistent.values())
            self.persistent.clear()
        for worker in workers:
            worker.terminate()

    @staticmethod
    def _update_result_metrics(line: str, metrics: Dict[str, Any]) -> None:
        prefix = "turbocider_result="
        stripped = line.strip()
        if not stripped.startswith(prefix):
            return
        try:
            parsed = json.loads(stripped[len(prefix):])
        except json.JSONDecodeError:
            return
        if isinstance(parsed, dict):
            metrics.update(parsed)

    @staticmethod
    def _parse_progress(line: str):
        for pattern in _PROGRESS_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            total = int(match.group(3))
            completed = int(match.group(2))
            if total > 0:
                return match.group(1), max(0.0, min(1.0, completed / total))
        return None
