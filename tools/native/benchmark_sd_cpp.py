#!/usr/bin/env python3
"""Compare resident TurboCider GPU inference with stable-diffusion.cpp Metal.

Both engines read the same local model component paths. Each engine receives one
cold request followed by repeated warm requests. GPU work is strictly serial.
stable-diffusion.cpp phase timings are parsed from its own INFO log; its external
wall additionally includes HTTP polling, PNG encoding/base64 transfer and file
write. TurboCider wall includes its native PNG export.
"""

from __future__ import annotations

import argparse
import base64
import ctypes as C
import hashlib
import json
import os
import re
import socket
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path


PHASE_PATTERNS = {
    "text_encode": re.compile(r"get_learned_condition completed, taking ([0-9.]+)s"),
    "denoise": re.compile(r"sampling completed, taking ([0-9.]+)s"),
    "vae_decode": re.compile(r"decode_first_stage completed, taking ([0-9.]+)s"),
    "internal_request_wall": re.compile(r"generate_image completed in ([0-9.]+)s"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(url: str, method: str = "GET", value: dict | None = None) -> dict:
    body = json.dumps(value).encode() if value is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def wait_server(process: subprocess.Popen, base: str) -> None:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"sd-server exited with {process.returncode}")
        try:
            http_json(base + "/sdcpp/v1/capabilities")
            return
        except Exception:
            time.sleep(0.1)
    raise TimeoutError("sd-server did not become ready")


def decode_job_image(status: dict, output: Path) -> None:
    images = status.get("result", {}).get("images", [])
    blob = images[0].get("b64_json") if images else None
    if not blob:
        raise RuntimeError(f"sd-server completed without image: {status}")
    output.write_bytes(base64.b64decode(blob))


def consume(lib, pointer: C.c_void_p) -> str | None:
    if not pointer.value:
        return None
    value = C.string_at(pointer).decode()
    lib.tc_string_free(pointer)
    return value


def configure_turbocider(path: Path):
    lib = C.CDLL(str(path))
    lib.tc_string_free.argtypes = [C.c_void_p]
    lib.tc_engine_create_model.argtypes = [
        C.c_char_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)
    ]
    lib.tc_engine_generate.argtypes = [
        C.c_void_p, C.c_char_p, C.c_void_p, C.c_void_p,
        C.POINTER(C.c_void_p), C.POINTER(C.c_void_p),
    ]
    lib.tc_engine_free.argtypes = [C.c_void_p]
    return lib


def run_turbocider(args, output: Path) -> dict:
    lib = configure_turbocider(args.library)
    engine, error = C.c_void_p(), C.c_void_p()
    started = time.perf_counter()
    status = lib.tc_engine_create_model(
        args.model_id.encode(), str(args.model_root).encode(), C.byref(engine), C.byref(error)
    )
    constructor = time.perf_counter() - started
    message = consume(lib, error)
    if status:
        raise RuntimeError(message)

    runs = []
    try:
        for index in range(args.runs):
            target = output / f"turbocider-{index}.png"
            request = {
                "schema_version": 2,
                "model": args.model_id,
                "operation": "image.generate",
                "inputs": [{"kind": "text", "role": "prompt", "text": args.prompt}],
                "outputs": [{
                    "kind": "image", "path": str(target), "width": args.width,
                    "height": args.height, "frames": 1, "audio": False,
                }],
                "sampling": {"steps": args.steps, "seed": args.seed},
                "execution": {"policy": "gpu", "residency": "resident"},
                "parameters": {"dynamic_text": True},
            }
            result, error = C.c_void_p(), C.c_void_p()
            started = time.perf_counter()
            status = lib.tc_engine_generate(
                engine, json.dumps(request).encode(), None, None,
                C.byref(result), C.byref(error),
            )
            wall = time.perf_counter() - started
            report, message = consume(lib, result), consume(lib, error)
            if status:
                raise RuntimeError(message)
            metrics = json.loads(report)
            phases = metrics["timings_seconds"]
            known = phases["text_encode"] + phases["denoise"] + phases["vae_decode"]
            runs.append({
                "run": index,
                "cold": index == 0,
                "external_e2e_seconds": wall,
                "internal_request_wall_seconds": phases["request_wall"],
                "text_encode_seconds": phases["text_encode"],
                "denoise_seconds": phases["denoise"],
                "vae_decode_seconds": phases["vae_decode"],
                "export_and_framework_seconds": max(0.0, wall - known),
                "prompt_cache_hit": metrics.get("prompt_cache_hit"),
                "mlx_active_bytes": metrics.get("memory", {}).get("mlx_active_bytes"),
                "mlx_peak_bytes": metrics.get("memory", {}).get("mlx_peak_bytes"),
                "gpu_graph": metrics.get("gpu_graph"),
                "output": str(target),
                "output_sha256": sha256(target),
                "raw_metrics": metrics,
            })
    finally:
        lib.tc_engine_free(engine)
    return {"constructor_seconds": constructor, "runs": runs}


def parse_sd_phases(lines: list[str]) -> dict:
    phases = {}
    joined = "\n".join(lines)
    for name, pattern in PHASE_PATTERNS.items():
        matches = pattern.findall(joined)
        if not matches:
            raise RuntimeError(f"missing stable-diffusion.cpp timing {name}")
        phases[name] = float(matches[-1])
    return phases


def run_sd_cpp(args, output: Path) -> dict:
    port = free_port()
    command = [
        str(args.sd_server),
        "--diffusion-model", str(args.diffusion_model),
        "--vae", str(args.vae),
        "--llm", str(args.llm),
        "--backend", "metal",
        "--params-backend", "metal",
        "--listen-ip", "127.0.0.1",
        "--listen-port", str(port),
        "--diffusion-fa",
        "--diffusion-conv-direct",
        "--vae-conv-direct",
        "--cfg-scale", "1.0",
    ]
    lines: list[str] = []
    log_path = output / "stable-diffusion-cpp.log"
    log_file = log_path.open("w")
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    def drain() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line.rstrip("\n"))
            log_file.write(line)
            log_file.flush()

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    base = f"http://127.0.0.1:{port}"
    started = time.perf_counter()
    wait_server(process, base)
    ready = time.perf_counter() - started
    payload = {
        "prompt": args.prompt,
        "negative_prompt": "",
        "width": args.width,
        "height": args.height,
        "batch_count": 1,
        "seed": args.seed,
        "sample_params": {
            "sample_steps": args.steps,
            "sample_method": "euler",
            "guidance": {"txt_cfg": 1.0, "img_cfg": 1.0},
        },
        "output_format": "png",
        "output_compression": 100,
    }
    runs = []
    try:
        for index in range(args.runs):
            target = output / f"stable-diffusion-cpp-{index}.png"
            line_start = len(lines)
            completed_before = sum("generate_image completed in" in line for line in lines)
            started = time.perf_counter()
            job = http_json(base + "/sdcpp/v1/img_gen", "POST", payload)
            job_id = job.get("id")
            if not job_id:
                raise RuntimeError(f"sd-server returned no job id: {job}")
            while True:
                status = http_json(base + f"/sdcpp/v1/jobs/{job_id}")
                state = status.get("status")
                if state == "completed":
                    decode_job_image(status, target)
                    break
                if state in {"failed", "cancelled"}:
                    raise RuntimeError(f"sd-server generation failed: {status}")
                time.sleep(0.02)
            wall = time.perf_counter() - started
            deadline = time.monotonic() + 10
            while (sum("generate_image completed in" in line for line in lines)
                   <= completed_before and time.monotonic() < deadline):
                time.sleep(0.01)
            phases = parse_sd_phases(lines[line_start:])
            known = phases["text_encode"] + phases["denoise"] + phases["vae_decode"]
            runs.append({
                "run": index,
                "cold": index == 0,
                "external_e2e_seconds": wall,
                "internal_request_wall_seconds": phases["internal_request_wall"],
                "text_encode_seconds": phases["text_encode"],
                "denoise_seconds": phases["denoise"],
                "vae_decode_seconds": phases["vae_decode"],
                "export_and_framework_seconds": max(0.0, wall - known),
                "output": str(target),
                "output_sha256": sha256(target),
            })
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        reader.join(timeout=5)
        log_file.close()
    return {"server_ready_seconds": ready, "command": command, "runs": runs}


def summarize(runs: list[dict]) -> dict:
    warm = runs[1:]
    keys = [
        "external_e2e_seconds", "internal_request_wall_seconds", "text_encode_seconds",
        "denoise_seconds", "vae_decode_seconds", "export_and_framework_seconds",
    ]
    return {
        "cold": {key: runs[0][key] for key in keys},
        "warm_samples": len(warm),
        "warm_median": {key: statistics.median(run[key] for run in warm) for key in keys},
        "warm_min": {key: min(run[key] for run in warm) for key in keys},
        "warm_max": {key: max(run[key] for run in warm) for key in keys},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=("z-image-turbo", "flux2-klein-4b", "flux2-klein-9b"), required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--diffusion-model", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--llm", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--sd-server", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--order", choices=("turbo-first", "sd-first"), default="turbo-first")
    parser.add_argument("--prompt", default="A cinematic red fox walking through fresh snow, soft morning light.")
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2 so a warm result exists")
    for name in ("model_root", "diffusion_model", "vae", "llm", "library", "sd_server"):
        path = getattr(args, name).resolve()
        if not path.exists():
            parser.error(f"--{name.replace('_', '-')} does not exist: {path}")
        setattr(args, name, path)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    engines = ("turbocider", "stable-diffusion.cpp")
    if args.order == "sd-first":
        engines = tuple(reversed(engines))
    results = {}
    for engine in engines:
        print(f"RUN {args.model_id} {args.width}x{args.height} {engine}", flush=True)
        if engine == "turbocider":
            results[engine] = run_turbocider(args, args.output)
        else:
            results[engine] = run_sd_cpp(args, args.output)
        results[engine]["summary"] = summarize(results[engine]["runs"])
        (args.output / "report.json").write_text(json.dumps({
            "schema_version": 1,
            "method": "one cold request plus repeated same-prompt warm requests per resident engine; engines run serially",
            "model_id": args.model_id,
            "model_root": str(args.model_root),
            "shared_components": {
                "diffusion_model": str(args.diffusion_model),
                "vae": str(args.vae),
                "llm": str(args.llm),
            },
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "seed": args.seed,
            "prompt": args.prompt,
            "run_order": engines,
            "binary_identity": {
                "turbocider_library": str(args.library),
                "turbocider_library_sha256": sha256(args.library),
                "sd_server": str(args.sd_server),
                "sd_server_sha256": sha256(args.sd_server),
            },
            "results": results,
        }, indent=2) + "\n")
        time.sleep(2)
    print(json.dumps({name: value["summary"] for name, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
