"""Benchmark sequential LTX requests through the durable TurboCider service."""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
import uuid
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--residency", choices=["resident", "component_staged"],
                        default="resident")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--restart-between", action="store_true")
    parser.add_argument("--prompt", default=(
        "A cinematic red fox running through a snowy forest"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class Service:
    def __init__(self, executable, state, resident_candidate=False):
        self.executable = str(Path(executable).resolve())
        self.state = Path(state).resolve()
        self.state.mkdir(parents=True, exist_ok=True)
        self.socket_path = Path("/private/tmp") / (
            "turbocider-ltx-bench-" + uuid.uuid4().hex[:12] + ".sock")
        self.log = None
        self.process = None
        self.resident_candidate = resident_candidate

    def rpc(self, value):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(10)
            client.connect(str(self.socket_path))
            client.sendall(json.dumps(value).encode() + b"\n")
            payload = b""
            while b"\n" not in payload:
                chunk = client.recv(65536)
                if not chunk:
                    raise RuntimeError("service closed the RPC connection")
                payload += chunk
        response = json.loads(payload.split(b"\n", 1)[0])
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["result"]

    def start(self):
        self.log = (self.state / "benchmark-service.log").open("a")
        environment = os.environ.copy()
        if self.resident_candidate:
            environment["TURBOCIDER_LTX_RESIDENT_CANDIDATE"] = "1"
        self.process = subprocess.Popen(
            [self.executable, "serve", str(self.socket_path), str(self.state)],
            stdout=self.log, stderr=self.log, env=environment)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("TurboCider service exited during startup")
            try:
                self.rpc({"action": "doctor"})
                return
            except (FileNotFoundError, ConnectionError, OSError, RuntimeError):
                time.sleep(0.1)
        raise TimeoutError("TurboCider service did not become ready")

    def stop(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=30)
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None
        self.socket_path.unlink(missing_ok=True)
        Path(str(self.socket_path) + ".lock").unlink(missing_ok=True)


def wait_for_job(service, job_id):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        value = service.rpc({"action": "status", "id": job_id})
        if value["state"] in {
                "succeeded", "failed", "cancelled", "interrupted"}:
            return value
        time.sleep(0.2)
    raise TimeoutError(job_id)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be positive")
    model = Path(args.model).resolve()
    output = Path(args.output).resolve()
    report = Path(args.report).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    service = Service(args.executable, args.state,
                      resident_candidate=args.residency == "resident")
    rows = []
    try:
        service.start()
        for index in range(args.runs):
            destination = output / f"ltx-{args.residency}-{index}.mp4"
            request = {
                "schema_version": 1,
                "model": "ltx-2.5-distilled",
                "operation": "video.generate",
                "prompt": args.prompt,
                "width": 704,
                "height": 448,
                "frames": 97,
                "fps": 24,
                "steps": 11,
                "seed": args.seed,
                "execution": "gpu",
                "residency": args.residency,
                "audio": False,
                "output": str(destination),
            }
            started = time.perf_counter()
            submitted = service.rpc({
                "action": "submit",
                "model_path": str(model),
                "request": request,
            })
            job = wait_for_job(service, submitted["id"])
            wall = time.perf_counter() - started
            if job["state"] != "succeeded":
                raise RuntimeError(job.get("error", json.dumps(job)))
            rows.append({
                "run": index,
                "client_wall_seconds": wall,
                "job": job,
                "output_bytes": destination.stat().st_size,
                "output_sha256": file_sha256(destination),
            })
            report.write_text(json.dumps({
                "format": "turbocider-ltx-service-benchmark-v1",
                "residency": args.residency,
                "restart_between": args.restart_between,
                "runs": rows,
            }, indent=2))
            print(json.dumps({
                "run": index,
                "client_wall_seconds": wall,
                "result": job["result"],
            }), flush=True)
            if args.restart_between and index + 1 < args.runs:
                service.stop()
                service = Service(
                    args.executable, args.state,
                    resident_candidate=args.residency == "resident")
                service.start()
    finally:
        service.stop()


if __name__ == "__main__":
    main()
