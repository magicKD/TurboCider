"""Standard-library client for TurboCider's local Unix-socket API.

No model libraries, uploads, service launch, or automatic submission retries.
Use capabilities() and plan() before constructing a workflow.
"""
from __future__ import annotations

import json
import math
import socket
import time
from typing import Any, Callable

TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled", "interrupted"})


class APIError(RuntimeError):
    """The service rejected a request without accepting it as a successful RPC."""


class TransportError(RuntimeError):
    """A connection or response failed; a submission may already be persisted."""

    def __init__(self, message: str, *, submission_may_have_succeeded: bool):
        super().__init__(message)
        self.submission_may_have_succeeded = submission_may_have_succeeded


class JobTimeout(TimeoutError):
    """Waiting timed out. The job is still owned by the service, not cancelled."""

    def __init__(self, job_id: str, last_job: dict[str, Any] | None):
        super().__init__(f"Timed out waiting for {job_id}; query status or cancel explicitly")
        self.job_id, self.last_job = job_id, last_job


class JobFailed(RuntimeError):
    def __init__(self, job: dict[str, Any]):
        super().__init__(f"Job {job.get('id')} {job.get('state')}: {job.get('error', '')}")
        self.job = job


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate response field: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> Any:
    raise ValueError(f"Non-finite JSON constant: {value}")


class Client:
    def __init__(self, socket_path: str, *, timeout: float = 30,
                 max_response_bytes: int = 32 * 1024 * 1024):
        if not isinstance(socket_path, str) or not socket_path or "\0" in socket_path:
            raise ValueError("socket_path must be a nonempty local socket path")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if type(max_response_bytes) is not int or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be a positive integer")
        self.socket_path = socket_path
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes

    def rpc(self, action: str, **fields: Any) -> Any:
        return self._exchange(action, fields, self.timeout)

    def _exchange(self, action: str, fields: dict[str, Any], timeout: float) -> Any:
        if not isinstance(action, str) or not action:
            raise ValueError("action must be a nonempty string")
        payload = json.dumps({"action": action, **fields}, ensure_ascii=False,
                             allow_nan=False, separators=(",", ":")).encode("utf-8")
        if len(payload) > 1048576:
            raise ValueError("RPC request exceeds 1 MiB")
        sent = False
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(timeout)
                connection.connect(self.socket_path)
                # A partial write followed by an error is also ambiguous.
                sent = True
                connection.sendall(payload + b"\n")
                with connection.makefile("rb") as stream:
                    line = stream.readline(self.max_response_bytes + 1)
            if len(line) > self.max_response_bytes:
                raise ValueError("RPC response exceeds configured limit; request a smaller jobs page")
            if not line.endswith(b"\n"):
                raise ValueError("RPC response ended before newline")
            response = json.loads(line, object_pairs_hook=_unique_object,
                                  parse_constant=_invalid_constant)
            if not isinstance(response, dict) or type(response.get("ok")) is not bool:
                raise ValueError("Invalid RPC response envelope")
            if response["ok"]:
                if "result" not in response:
                    raise ValueError("RPC response has no result")
                return response["result"]
            if not isinstance(response.get("error"), str):
                raise ValueError("RPC error response has no error message")
        except (OSError, ValueError) as error:
            ambiguous = action == "submit" and sent
            suffix = "; inspect jobs before resubmitting" if ambiguous else ""
            raise TransportError(str(error) + suffix,
                                 submission_may_have_succeeded=ambiguous) from error
        raise APIError(response["error"])

    def capabilities(self) -> dict[str, Any]:
        return self.rpc("capabilities")

    def models(self) -> dict[str, Any]:
        return self.rpc("models")

    def plan(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.rpc("plan", request=request)

    def submit(self, model_path: str, request: dict[str, Any]) -> str:
        result = self.rpc("submit", model_path=model_path, request=request)
        if not isinstance(result, dict) or not isinstance(result.get("id"), str) or not result["id"]:
            raise TransportError("Submission response has no job ID; inspect jobs before resubmitting",
                                 submission_may_have_succeeded=True)
        return result["id"]

    def status(self, job_id: str) -> dict[str, Any]:
        return self.rpc("status", id=job_id)

    def jobs(self, *, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        return self.rpc("jobs", offset=offset, limit=limit)

    def cancel(self, job_id: str) -> dict[str, Any]:
        return self.rpc("cancel", id=job_id)

    def wait(self, job_id: str, *, timeout: float = 1800, poll_interval: float = 1,
             on_progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be finite and positive")
        deadline = time.monotonic() + timeout
        last_job = None
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                last_job = self._exchange("status", {"id": job_id}, min(self.timeout, remaining))
            except TransportError as error:
                if time.monotonic() >= deadline:
                    raise JobTimeout(job_id, last_job) from error
                raise
            if not isinstance(last_job, dict) or last_job.get("id") != job_id:
                raise TransportError("Status returned an invalid job identity",
                                     submission_may_have_succeeded=False)
            if on_progress is not None:
                on_progress(last_job)
            if last_job.get("state") == "succeeded":
                return last_job
            if last_job.get("state") in TERMINAL_STATES:
                raise JobFailed(last_job)
            time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))
        raise JobTimeout(job_id, last_job)
