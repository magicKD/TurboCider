"""Serialized heavy-generation job queue and durable metadata."""

from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

from turbocider.adapters.base import CommandSpec
from turbocider.errors import JobNotFoundError, TurboCiderError
from turbocider.models import GenerationRequest, JobRecord, JobState
from turbocider.runner import ProcessRunner


class JobManager:
    def __init__(self, state_directory: Path, max_workers: int = 1):
        self.state_directory = state_directory
        self.jobs_directory = state_directory / "jobs"
        self.logs_directory = state_directory / "logs"
        self.jobs_directory.mkdir(parents=True, exist_ok=True)
        self.logs_directory.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="turbocider")
        self.runner = ProcessRunner()
        self.records: Dict[str, JobRecord] = {}
        self.specs: Dict[str, CommandSpec] = {}
        self.cancel_events: Dict[str, threading.Event] = {}
        self.futures: Dict[str, Future] = {}
        self.lock = threading.RLock()
        self._restore_records()

    def _restore_records(self) -> None:
        for path in sorted(self.jobs_directory.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                record = JobRecord.from_dict(raw)
            except (
                KeyError,
                TypeError,
                ValueError,
                OSError,
                json.JSONDecodeError,
                TurboCiderError,
            ):
                continue
            if record.state in (JobState.QUEUED, JobState.RUNNING):
                record.state = JobState.FAILED
                record.phase = "interrupted"
                record.finished_at = time.time()
                record.error = "TurboCider daemon stopped before this job completed"
                self._persist(record)
            self.records[record.id] = record

    def submit(
        self,
        request: GenerationRequest,
        plan_id: str,
        spec: CommandSpec,
    ) -> JobRecord:
        job_id = uuid.uuid4().hex
        record = JobRecord(
            id=job_id,
            request=request,
            plan_id=plan_id,
            output_paths=[str(path) for path in spec.output_paths],
            command=list(spec.argv),
            log_path=str(self.logs_directory / (job_id + ".log")),
            estimated_total_seconds=(
                float(spec.metadata["expected_seconds"])
                if spec.metadata.get("expected_seconds") is not None
                else None
            ),
        )
        with self.lock:
            self.records[job_id] = record
            self.specs[job_id] = spec
            self.cancel_events[job_id] = threading.Event()
            self._persist(record)
            self.futures[job_id] = self.executor.submit(self._execute, job_id)
        return record

    def _execute(self, job_id: str) -> None:
        with self.lock:
            record = self.records[job_id]
            spec = self.specs[job_id]
            cancel = self.cancel_events[job_id]
            if cancel.is_set():
                record.state = JobState.CANCELLED
                record.phase = "cancelled"
                record.finished_at = time.time()
                self._persist(record)
                return
            record.state = JobState.RUNNING
            record.phase = "starting"
            record.started_at = time.time()
            self._persist(record)

        def progress(phase: str, fraction: float, line: str) -> None:
            now = time.monotonic()
            with self.lock:
                current = self.records[job_id]
                if phase:
                    current.phase = phase
                if (
                    phase.strip().casefold() == "model_load"
                    and fraction == 0.0
                    and spec.metadata.get("expected_seconds_cold") is not None
                ):
                    current.estimated_total_seconds = float(
                        spec.metadata["expected_seconds_cold"]
                    )
                if fraction >= 0:
                    overall = self._overall_progress(spec, phase, fraction)
                    current.progress = max(current.progress, overall)
                    if (
                        current.started_at is not None
                        and current.progress >= 0.02
                        and current.progress < 1.0
                    ):
                        elapsed = max(0.0, time.time() - current.started_at)
                        sample = elapsed / current.progress
                        previous = current.estimated_total_seconds
                        current.estimated_total_seconds = (
                            sample if previous is None else previous * 0.65 + sample * 0.35
                        )
                current.metrics["last_log_line"] = line[-1000:]
                last_persist = float(current.metrics.get("last_progress_persist_monotonic", 0.0))
                if now - last_persist >= 1.0:
                    current.metrics["last_progress_persist_monotonic"] = now
                    self._persist(current)

        try:
            result = self.runner.run(
                spec,
                Path(record.log_path or (self.logs_directory / (job_id + ".log"))),
                cancel,
                progress,
            )
            with self.lock:
                record = self.records[job_id]
                record.finished_at = time.time()
                record.metrics.update(
                    {
                        key: value
                        for key, value in spec.metadata.items()
                        if key != "persistent_worker"
                    }
                )
                result_metrics = dict(result.metrics)
                if "wall_seconds" in result_metrics:
                    record.metrics["engine_wall_seconds"] = result_metrics.pop(
                        "wall_seconds"
                    )
                record.metrics.update(result_metrics)
                record.metrics.pop("last_progress_persist_monotonic", None)
                record.metrics["wall_seconds"] = result.wall_seconds
                record.metrics["job_wall_seconds"] = result.wall_seconds
                record.metrics["exit_code"] = result.return_code
                missing_outputs = [str(path) for path in spec.output_paths if not path.exists()]
                if cancel.is_set():
                    record.state = JobState.CANCELLED
                    record.phase = "cancelled"
                elif result.return_code == 0 and not missing_outputs:
                    record.state = JobState.SUCCEEDED
                    record.phase = "complete"
                    record.progress = 1.0
                    if record.started_at is not None:
                        record.estimated_total_seconds = max(
                            0.0, record.finished_at - record.started_at
                        )
                else:
                    record.state = JobState.FAILED
                    record.phase = "failed"
                    if missing_outputs and result.return_code == 0:
                        record.error = "engine did not create required outputs: %s" % ", ".join(missing_outputs)
                    else:
                        record.error = "engine exited with code %d; see %s" % (
                            result.return_code,
                            result.log_path,
                        )
                self._persist(record)
        except Exception as error:
            with self.lock:
                record = self.records[job_id]
                record.state = JobState.CANCELLED if cancel.is_set() else JobState.FAILED
                record.phase = record.state.value
                record.error = str(error)
                record.finished_at = time.time()
                self._persist(record)

    def get(self, job_id: str) -> JobRecord:
        with self.lock:
            try:
                return self.records[job_id]
            except KeyError as error:
                raise JobNotFoundError("unknown job: %s" % job_id) from error

    def cancel(self, job_id: str) -> JobRecord:
        with self.lock:
            record = self.get(job_id)
            if record.state in (JobState.QUEUED, JobState.RUNNING):
                self.cancel_events[job_id].set()
                record.phase = "cancelling"
                self._persist(record)
            return record

    def wait(self, job_id: str, timeout: Optional[float] = None) -> JobRecord:
        with self.lock:
            future = self.futures.get(job_id)
        if future:
            future.result(timeout=timeout)
        return self.get(job_id)

    def all(self):
        with self.lock:
            return sorted(self.records.values(), key=lambda item: item.created_at, reverse=True)

    def close(self) -> None:
        with self.lock:
            for job_id, record in self.records.items():
                if record.state in (JobState.QUEUED, JobState.RUNNING):
                    self.cancel_events[job_id].set()
                    record.phase = "cancelling"
                    self._persist(record)
        self.executor.shutdown(wait=True, cancel_futures=True)
        with self.lock:
            for record in self.records.values():
                if record.state in (JobState.QUEUED, JobState.RUNNING):
                    record.state = JobState.CANCELLED
                    record.phase = "cancelled"
                    record.finished_at = time.time()
                    self._persist(record)
        self.runner.close()

    def _persist(self, record: JobRecord) -> None:
        destination = self.jobs_directory / (record.id + ".json")
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record.as_dict(), indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)

    @staticmethod
    def _overall_progress(spec: CommandSpec, phase: str, fraction: float) -> float:
        phases = spec.metadata.get("progress_phases", [])
        if not isinstance(phases, list) or not phases:
            return max(0.0, min(1.0, fraction))
        normalized = phase.strip().casefold()
        total = 0.0
        completed = 0.0
        matched_weight = None
        for item in phases:
            if not isinstance(item, dict):
                continue
            weight = max(0.0, float(item.get("weight", 0.0)))
            total += weight
            if matched_weight is None:
                names = [str(item.get("name", ""))]
                aliases = item.get("aliases", [])
                if isinstance(aliases, list):
                    names.extend(str(alias) for alias in aliases)
                if normalized in {name.strip().casefold() for name in names}:
                    matched_weight = weight
                    completed += weight * max(0.0, min(1.0, fraction))
                else:
                    completed += weight
        if matched_weight is None or total <= 0.0:
            return max(0.0, min(1.0, fraction))
        return max(0.0, min(1.0, completed / total))
