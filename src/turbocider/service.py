"""Local HTTP service exposing the same TurboCider runtime as the CLI."""

from __future__ import annotations

import json
import hmac
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from pathlib import Path
from urllib.parse import urlparse

from turbocider.errors import TurboCiderError
from turbocider.models import GenerationRequest, JobState
from turbocider.runtime import TurboCiderRuntime


_TERMINAL = {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}


class TurboCiderHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address,
        runtime: TurboCiderRuntime,
        token: Optional[str] = None,
        allowed_input_roots=None,
        allow_external_output: bool = False,
    ):
        super().__init__(address, TurboCiderHandler)
        self.runtime = runtime
        self.token = token
        if allowed_input_roots is None:
            roots = (runtime.output_directory.resolve(),)
        else:
            roots = tuple(
                Path(path).expanduser().resolve() for path in allowed_input_roots
            )
        self.allowed_input_roots = roots
        self.allow_external_output = allow_external_output


class TurboCiderHandler(BaseHTTPRequestHandler):
    server: TurboCiderHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:
        return

    def _authorized(self) -> bool:
        token = self.server.token
        if not token:
            return True
        supplied = self.headers.get("Authorization", "")
        return hmac.compare_digest(supplied, "Bearer " + token)

    @staticmethod
    def _within(path: Path, roots) -> bool:
        return any(path == root or root in path.parents for root in roots)

    def _validate_paths(self, request: GenerationRequest) -> None:
        output = request.output.path
        if output and not self.server.allow_external_output:
            destination = Path(output).expanduser().resolve()
            output_root = self.server.runtime.output_directory.resolve()
            if not self._within(destination, (output_root,)):
                raise ValueError(
                    "API output.path must be inside the TurboCider output directory"
                )
        for asset in request.inputs:
            for label, raw_path in (("path", asset.path), ("audio_path", asset.audio_path)):
                if not raw_path:
                    continue
                candidate = Path(raw_path).expanduser().resolve()
                if not candidate.is_file():
                    raise ValueError("input %s is not a regular file: %s" % (label, candidate))
                if not self._within(candidate, self.server.allowed_input_roots):
                    raise ValueError("input %s is outside allowed roots" % label)

    def _json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 4 * 1024 * 1024:
            raise ValueError("request body exceeds 4 MiB")
        payload = self.rfile.read(length) if length else b"{}"
        raw = json.loads(payload.decode("utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("JSON body must be an object")
        return raw

    def _send_json(self, status: int, value: Any) -> None:
        payload = (json.dumps(value, indent=2) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _dispatch(self, method: str) -> None:
        if not self._authorized():
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            if method == "GET" and path == "/health":
                self._send_json(HTTPStatus.OK, {"status": "ok"})
                return
            if method == "GET" and path == "/v1/models":
                self._send_json(
                    HTTPStatus.OK,
                    {"data": [item.public_dict() for item in self.server.runtime.registry.all()]},
                )
                return
            if method == "GET" and path == "/v1/system":
                self._send_json(HTTPStatus.OK, self.server.runtime.public_system())
                return
            if method == "GET" and path == "/v1/jobs":
                self._send_json(
                    HTTPStatus.OK,
                    {"data": [item.as_dict() for item in self.server.runtime.jobs.all()]},
                )
                return
            if method == "POST" and path == "/v1/jobs":
                request = GenerationRequest.from_dict(self._json_body())
                self._validate_paths(request)
                record = self.server.runtime.submit(request)
                self._send_json(HTTPStatus.ACCEPTED, record.as_dict())
                return
            if method == "POST" and path == "/v1/plans":
                request = GenerationRequest.from_dict(self._json_body())
                self._validate_paths(request)
                self._send_json(HTTPStatus.OK, {"data": self.server.runtime.plans(request)})
                return
            if path.startswith("/v1/jobs/"):
                suffix = path[len("/v1/jobs/") :]
                if suffix.endswith("/events") and method == "GET":
                    self._events(suffix[: -len("/events")])
                    return
                job_id = suffix
                if method == "GET":
                    self._send_json(HTTPStatus.OK, self.server.runtime.jobs.get(job_id).as_dict())
                    return
                if method == "DELETE":
                    self._send_json(HTTPStatus.OK, self.server.runtime.jobs.cancel(job_id).as_dict())
                    return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (TurboCiderError, ValueError, KeyError, json.JSONDecodeError) as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def _events(self, job_id: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        last = None
        while True:
            record = self.server.runtime.jobs.get(job_id)
            rendered = json.dumps(record.as_dict(), separators=(",", ":"))
            if rendered != last:
                self.wfile.write(("event: job\ndata: " + rendered + "\n\n").encode("utf-8"))
                self.wfile.flush()
                last = rendered
            if record.state in _TERMINAL:
                break
            time.sleep(0.25)
        self.close_connection = True

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")


def serve(
    runtime: TurboCiderRuntime,
    host: str = "127.0.0.1",
    port: int = 11435,
    token: Optional[str] = None,
    allowed_input_roots=(),
    allow_external_output: bool = False,
) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"} and not token:
        raise ValueError("a bearer token is required when binding outside loopback")
    roots = tuple(allowed_input_roots)
    if not roots:
        roots = (
            (Path.home(),)
            if host in {"127.0.0.1", "localhost", "::1"}
            else (runtime.output_directory,)
        )
    server = TurboCiderHTTPServer(
        (host, port),
        runtime,
        token=token,
        allowed_input_roots=roots,
        allow_external_output=allow_external_output,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
