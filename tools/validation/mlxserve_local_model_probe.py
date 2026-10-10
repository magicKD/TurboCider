#!/usr/bin/env python3
"""Owned loopback-only compatibility probe, not a framework latency ranking."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

from runtime_ane_common import benchmark_environment, sha256_file


def listener_pid(port):
    result = subprocess.run(["/usr/sbin/lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp"],
                            capture_output=True, text=True, timeout=5)
    return {int(line[1:]) for line in result.stdout.splitlines() if line.startswith("p")}


def stop_owned(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        except ProcessLookupError:
            process.wait(timeout=5)


def request(port, route, payload=None, timeout=5):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}{route}", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()


def generation_request(modality):
    route = "/v1/images/generations" if modality == "image" else "/v1/video/generations"
    payload = dict(prompt="A red fox sitting in a snowy forest.", seed=42)
    payload.update(dict(size="512x512", steps=4) if modality == "image" else
                   dict(width=704, height=448, frames=97, steps=11, fps=24))
    return route, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout-label", default="original local FLUX9B BF16 diffusers shards")
    parser.add_argument("--modality", choices=("image", "video"), default="image")
    args = parser.parse_args()
    binary, model = args.binary.resolve(strict=True), args.model.resolve(strict=True)
    if args.output.exists():
        parser.error("fresh probe output required")
    args.output.mkdir(parents=True)
    version = subprocess.check_output([str(binary), "--version"], text=True, stderr=subprocess.STDOUT)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [str(binary), "--serve", "--host", "127.0.0.1", "--port", str(port),
               "--model", str(model), "--log-file", "off"]
    report = dict(schema="tc-mlxserve-local-compatibility-v1", status="running", version=version,
                  binary_sha256=sha256_file(binary), model_layout=args.layout_label, modality=args.modality,
                  qualification_passed=False, latency_comparison_valid=False, requests=[],
                  scope="owned server over loopback; original unmodified model; no precision/layout conversion")
    start = time.perf_counter()
    with (args.output / "stdout.txt").open("x") as out, (args.output / "stderr.txt").open("x") as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, env=benchmark_environment(), start_new_session=True)
        try:
            ready = False
            while time.perf_counter() - start < 60 and process.poll() is None:
                # Refuse to POST to some other service if port allocation raced.
                if listener_pid(port) == {process.pid}:
                    try:
                        code, text = request(port, "/health")
                        if code == 200:
                            ready = True
                            report["ready_seconds"] = time.perf_counter() - start
                            break
                    except (OSError, urllib.error.URLError):
                        pass
                time.sleep(.2)
            if ready:
                if listener_pid(port) != {process.pid}:
                    raise RuntimeError("listener identity changed")
                route, payload = generation_request(args.modality)
                code, text = request(port, route, payload, timeout=120)
                report["requests"].append(dict(http_status=code, response_bytes=len(text.encode()),
                                                response=text if code != 200 else "image payload retained separately"))
                if code == 200:
                    (args.output / "image-response.json").write_text(text)
                report["status"] = "loaded_generation_succeeded" if code == 200 else "generation_rejected"
            else:
                report.update(status="startup_rejected" if process.poll() is not None else "startup_timeout",
                              startup_seconds=time.perf_counter() - start, exit_code=process.poll())
        finally:
            stop_owned(process)
    report["server_terminal"] = process.poll() is not None
    report["stdout_sha256"] = sha256_file(args.output / "stdout.txt")
    report["stderr_sha256"] = sha256_file(args.output / "stderr.txt")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "server_terminal", "latency_comparison_valid")}))


if __name__ == "__main__":
    main()
