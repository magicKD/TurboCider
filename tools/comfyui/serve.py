#!/usr/bin/env python3
"""Loopback HTTP gateway to a running TurboCider service; no model downloads.

The caller has the same trust as a local native API client. Model and adapter
selection use service-side registrations. Uploaded PNGs and outputs are owned
by this gateway; no arbitrary filesystem download endpoint is exposed.
"""
from __future__ import annotations
import argparse
import hmac
import json
import os
from pathlib import Path
import re
import struct
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bindings/python"))
from turbocider_local import Client, APIError, TransportError, _unique_object, _invalid_constant

ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
MAX_UPLOAD = 64 * 1024 * 1024


class Gateway:
    def __init__(self, client, root, token=""):
        self.client, self.root, self.token = client, Path(root).resolve(), token
        for name in ("inputs", "outputs"):
            (self.root / name).mkdir(parents=True, exist_ok=True, mode=0o700)

    def catalog(self):
        models = self.client.models()["models"]
        inventory = self.client.installations()["index"]
        executable = {m["id"]: m for m in models if m.get("executor")}
        return {"protocol": "turbocider.http", "version": 1,
                "models": models,
                "installations": [i for i in inventory["installations"] if i["modelID"] in executable],
                "loras": inventory.get("loras", []),
                "workflows": self.client.workflows(),
                "files_verified": False}

    def prepare(self, payload):
        if not isinstance(payload, dict) or set(payload) != {"installation_id", "request"}:
            raise ValueError("Expected installation_id and request")
        catalog = self.catalog()
        installation = next((i for i in catalog["installations"] if i["id"] == payload["installation_id"]), None)
        if installation is None:
            raise ValueError("Unknown installation; refresh catalog and select an existing TurboCider model")
        if not Path(installation["path"]).exists():
            raise ValueError("Registered model path no longer exists; repair the TurboCider registration")
        request = payload["request"]
        if not isinstance(request, dict):
            raise ValueError("request must be an object")
        request = json.loads(json.dumps(request, allow_nan=False))
        if request.get("schema_version", 1) != 1:
            raise ValueError("HTTP generation uses native schema 1; use the Unix API for schema 2")
        if any(key in request for key in ("output", "outputs", "model_path")):
            raise ValueError("Output paths are allocated by the gateway")
        if request.get("model", installation["modelID"]) != installation["modelID"]:
            raise ValueError("Request model does not match the selected installation")
        request["model"] = installation["modelID"]
        model = next(m for m in catalog["models"] if m["id"] == request["model"])
        extension = ".mp4" if model["output"] == "video" else ".png"
        request["output"] = str(self.root / "outputs" / (uuid.uuid4().hex + extension))
        inputs = request.get("inputs", [])
        if not isinstance(inputs, list):
            raise ValueError("inputs must be a list")
        for item in inputs:
            if not isinstance(item, dict):
                raise ValueError("input must be an object")
            if "asset_id" in item:
                asset_id = item.pop("asset_id")
                if not isinstance(asset_id, str) or not ID.fullmatch(asset_id) or "path" in item:
                    raise ValueError("Invalid asset reference")
                path = self.root / "inputs" / (asset_id + ".png")
                if not path.is_file():
                    raise ValueError("Unknown uploaded image")
                item["path"] = str(path)
        loras = request.get("loras", [])
        if not isinstance(loras, list):
            raise ValueError("loras must be a list")
        for item in loras:
            if not isinstance(item, dict):
                raise ValueError("LoRA must be an object")
            if "id" in item:
                adapter_id = item.pop("id")
                adapter = next((x for x in catalog["loras"] if x["id"] == adapter_id), None)
                if adapter is None or adapter["modelID"] != request["model"] or "path" in item:
                    raise ValueError("Unknown LoRA or LoRA belongs to a different model")
                if not Path(adapter["path"]).is_file():
                    raise ValueError("Registered LoRA file no longer exists")
                item["path"] = adapter["path"]
        return installation["path"], request

    def status(self, job_id):
        if not ID.fullmatch(job_id):
            raise ValueError("Invalid job ID")
        job = self.client.status(job_id)
        output = job.get("request", {}).get("output", "")
        path = Path(output).resolve()
        if path.parent != self.root / "outputs" or not re.fullmatch(r"[0-9a-f]{32}\.(png|mp4)", path.name):
            raise ValueError("Unknown gateway job")
        return job

    def submit(self, payload):
        model_path, request = self.prepare(payload)
        # The native service owns durable job history. Ownership derives from
        # its persisted, gateway-allocated output path, so even a lost HTTP
        # reply remains recoverable through /jobs after gateway restart.
        return {"id": self.client.submit(model_path, request)}

    def artifact(self, job_id):
        job = self.status(job_id)
        if job["state"] != "succeeded":
            raise ValueError("Artifact is available only after job success")
        path = Path(job["request"]["output"]).resolve()
        if job.get("result", {}).get("output") != str(path):
            raise ValueError("Result artifact does not match the submitted output")
        return path


class Handler(BaseHTTPRequestHandler):
    server_version = "TurboCiderHTTP/1"

    def log_message(self, fmt, *args):
        # Authorization and request bodies are never logged.
        super().log_message(fmt, *args)

    def send_data(self, status, data, mime="application/json"):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def body(self, binary=False):
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("Chunked requests are unsupported")
        expected = "image/png" if binary else "application/json"
        if self.headers.get_content_type() != expected:
            raise ValueError("Content-Type must be " + expected)
        length = int(self.headers.get("Content-Length", "-1"))
        if not 0 <= length <= (MAX_UPLOAD if binary else 1048576):
            raise ValueError("Missing Content-Length or request too large")
        data = self.rfile.read(length)
        if len(data) != length:
            raise ValueError("Incomplete body")
        return data if binary else json.loads(data, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)

    def handle_request(self):
        self.connection.settimeout(30)
        gateway = self.server.gateway
        try:
            if self.headers.get("Origin") is not None:
                self.send_data(403, {"error": "Browser-origin requests are disabled; use the ComfyUI backend nodes"})
                return
            host = self.headers.get("Host", "").split(":")[0].lower()
            if host not in ("localhost", "127.0.0.1"):
                self.send_data(403, {"error": "Only loopback Host headers are accepted"})
                return
            if gateway.token and not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + gateway.token):
                self.send_data(401, {"error": "Invalid API token"})
                return
            path = self.path
            if self.command == "GET" and path == "/v1/health":
                self.send_data(200, gateway.client.service_status())
            elif self.command == "GET" and path == "/v1/catalog":
                self.send_data(200, gateway.catalog())
            elif self.command == "GET" and path == "/v1/jobs":
                # Native history is retained even when a submit reply was lost.
                self.send_data(200, gateway.client.jobs(limit=100))
            elif self.command == "POST" and path == "/v1/plan":
                _, request = gateway.prepare(self.body())
                self.send_data(200, gateway.client.plan(request))
            elif self.command == "POST" and path == "/v1/jobs":
                self.send_data(202, gateway.submit(self.body()))
            elif self.command == "POST" and path == "/v1/assets":
                data = self.body(binary=True)
                if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
                    raise ValueError("Upload must be a PNG image")
                width, height = struct.unpack(">II", data[16:24])
                if not width or not height or width * height > 80000000:
                    raise ValueError("Image exceeds 80 million pixels or has zero dimension")
                asset_id = uuid.uuid4().hex
                with (gateway.root / "inputs" / (asset_id + ".png")).open("xb") as stream:
                    stream.write(data)
                self.send_data(201, {"id": asset_id})
            else:
                match = re.fullmatch(r"/v1/jobs/([A-Za-z0-9_-]{1,80})(/cancel|/artifact)?", path)
                if not match:
                    self.send_data(404, {"error": "Unknown endpoint"})
                    return
                job_id, suffix = match.groups()
                if self.command == "GET" and suffix is None:
                    self.send_data(200, gateway.status(job_id))
                elif self.command == "POST" and suffix == "/cancel":
                    if self.body() != {}:
                        raise ValueError("Cancel body must be {}")
                    gateway.status(job_id)
                    self.send_data(200, gateway.client.cancel(job_id))
                elif self.command == "GET" and suffix == "/artifact":
                    artifact = gateway.artifact(job_id)
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png" if artifact.suffix == ".png" else "video/mp4")
                    self.send_header("Content-Length", str(artifact.stat().st_size))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    with artifact.open("rb") as stream:
                        while chunk := stream.read(1024 * 1024):
                            self.wfile.write(chunk)
                else:
                    self.send_data(405, {"error": "Method not allowed"})
        except TransportError as error:
            self.send_data(503, {"error": str(error), "submission_may_have_succeeded": error.submission_may_have_succeeded})
        except (ValueError, APIError) as error:
            self.send_data(400, {"error": str(error)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except OSError as error:
            self.send_data(500, {"error": str(error)})

    do_GET = handle_request
    do_POST = handle_request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token-env", default="TURBOCIDER_API_TOKEN", help="Name of optional bearer-token environment variable")
    args = parser.parse_args()
    client = Client(args.socket)
    client.service_status()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    server.gateway = Gateway(client, args.data_dir, os.environ.get(args.token_env, ""))
    print(json.dumps({"ready": True, "url": f"http://127.0.0.1:{server.server_port}"}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
