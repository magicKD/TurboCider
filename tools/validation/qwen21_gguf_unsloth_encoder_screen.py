#!/usr/bin/env python3
"""Read-only Unsloth placement/max flags; fresh prompts in one owned sd-server.

Two-step requests are only an encoder lifecycle comparison, not full denoising
or image-quality qualification. Model loading and first native condition are
kept separate from the subsequent fresh-prompt conditions.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

from qwen21_gguf_hybrid_screen import PROMPTS
from qwen21_gguf_unsloth_screen import measured_time
from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_sampled

ROOT = Path(__file__).resolve().parents[2]


def worker(output):
    launch = json.loads((output / "launch.json").read_text())
    address = "http://127.0.0.1:" + str(launch["port"])
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with (output / "server.stdout.txt").open("x") as stdout, (output / "server.stderr.txt").open("x") as stderr:
        server = subprocess.Popen(launch["command"], cwd=ROOT, stdout=stdout, stderr=stderr)
        started = time.perf_counter()
        try:
            deadline = started + 180
            while True:
                if server.poll() is not None:
                    raise RuntimeError("owned reference server exited before readiness")
                try:
                    with opener.open(address + "/v1/models", timeout=1) as response:
                        models = json.load(response)
                    if not models.get("data"):
                        raise ValueError("missing actual reference model readiness")
                    break
                except (urllib.error.URLError, TimeoutError):
                    if time.perf_counter() >= deadline:
                        raise TimeoutError("reference server readiness deadline exceeded")
                    time.sleep(.2)
            ready_seconds = time.perf_counter() - started
            rows = []
            for index, prompt in enumerate(PROMPTS):
                stdout.flush()
                stderr.flush()
                paths = [output / "server.stdout.txt", output / "server.stderr.txt"]
                offsets = [p.stat().st_size for p in paths]
                request = dict(prompt=prompt, width=512, height=512, steps=2, seed=29,
                               cfg_scale=1, batch_size=1, sampler_name="euler")
                (output / f"request-{index}.json").write_text(json.dumps(request, indent=2) + "\n")
                start = time.perf_counter()
                with opener.open(urllib.request.Request(address + "/sdapi/v1/txt2img",
                    data=json.dumps(request).encode(), headers={"Content-Type": "application/json"}), timeout=300) as response:
                    result = json.load(response)
                request_seconds = time.perf_counter() - start
                images = result.get("images")
                if not isinstance(images, list) or len(images) != 1:
                    raise ValueError("missing complete reference image response")
                image = base64.b64decode(images[0], validate=True)
                if not image.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("reference response is not an original PNG")
                png = output / f"image-{index}.png"
                png.write_bytes(image)
                info = result.get("info", {})
                if isinstance(info, str):
                    info = json.loads(info)
                if any(info.get(k) != v for k, v in request.items() if k in ("prompt", "width", "height", "steps", "seed", "cfg_scale")):
                    raise ValueError("actual reference geometry/prompt/seed/steps differs")
                log = ""
                for path, offset in zip(paths, offsets):
                    with path.open("rb") as stream:
                        stream.seek(offset)
                        log += stream.read().decode(errors="replace")
                (output / f"request-{index}.log").write_text(log)
                rows.append(dict(index=index, fresh_prompt=prompt, http_request_seconds=request_seconds,
                                 condition_seconds=measured_time(log, "get_learned_condition completed"),
                                 sampling_seconds=measured_time(log, "sampling completed"),
                                 png_sha256=sha256_file(png), info=info))
                (output / "conditions.json").write_text(json.dumps(dict(ready_seconds=ready_seconds, requests=rows), indent=2) + "\n")
                print(json.dumps(rows[-1]), flush=True)
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder-placement", choices=("default", "gpu"), default="default")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.output.resolve(strict=True))
        return
    if args.output.exists() or args.output.is_symlink() or any(v is None for v in (args.reference, args.binary, args.model)):
        parser.error("reference, server, model and fresh evidence directory required")
    reference, binary, model = (p.resolve(strict=True) for p in (args.reference, args.binary, args.model))
    sys.dont_write_bytecode = True
    backend = reference / "studio/backend"
    sys.path.insert(0, str(backend))
    from core.inference.sd_cpp_args import metal_text_encoder_flags, native_speed_flags
    placement_key = "UNSLOTH_DIFFUSION_SD_CPP_METAL_TE_GPU"
    previous = os.environ.get(placement_key)
    try:
        os.environ[placement_key] = "1" if args.encoder_placement == "gpu" else "0"
        flags = metal_text_encoder_flags() + native_speed_flags("max")
    finally:
        if previous is None:
            os.environ.pop(placement_key, None)
        else:
            os.environ[placement_key] = previous
    if ("--clip-on-cpu" in flags) is not (args.encoder_placement == "default"):
        raise ValueError("original Unsloth encoder placement builder differs")
    sources = [binary, backend / "core/inference/sd_cpp_args.py", backend / "core/inference/diffusion_memory.py",
               model / "diffusion_models/qwen-image-2.1-Q4_K_M.gguf",
               model / "text_encoders/Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
               model / "vae/qwen_image_2.1_vae_bf16.safetensors"]
    identities = {str(p): sha256_file(p) for p in sources}
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    command = [str(binary), "--diffusion-model", str(sources[3]), "--llm", str(sources[4]), "--vae", str(sources[5]),
               *flags, "--threads", "12", "--verbose", "--cfg-scale", "1", "--listen-ip", "127.0.0.1", "--listen-port", str(port)]
    args.output.mkdir(parents=True)
    output = args.output.resolve()
    (output / "launch.json").write_text(json.dumps(dict(command=command, port=port), indent=2) + "\n")
    summary = dict(schema="tc-qwen21-unsloth-encoder-session-screen-v1", status="running", qualification_passed=False,
                   source_identities=identities, command=command, steps=2,
                   encoder_placement=args.encoder_placement,
                   scope="original Unsloth Apple default/explicit GPU text placement and Metal DiT; one process, three fresh prompts; native condition spans, not40-step or numerical parity")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    try:
        with (output / "driver.stdout.jsonl").open("x") as stdout, (output / "driver.stderr.txt").open("x") as stderr:
            memory = run_sampled([sys.executable, str(Path(__file__).resolve()), "--worker", "--output", str(output)],
                                 repo=ROOT, output=output, stem="reference", env=benchmark_environment(),
                                 stdout=stdout, stderr=stderr, timeout=900, interval_ms=100, max_gap_ms=500,
                                 child_roles=(("reference_inference", str(binary)),))
    except Exception as error:
        summary.update(status="failed_diagnostic", error=str(error))
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        raise
    conditions = json.loads((output / "conditions.json").read_text())
    log = (output / "server.stdout.txt").read_text() + (output / "server.stderr.txt").read_text()
    placement_proof = r"Conditioner[^\n]*-> compute " + ("CPU, params cpu" if args.encoder_placement == "default" else "MTL0, params MTL0")
    if (len(conditions["requests"]) != 3 or not re.search(placement_proof, log) or
        any(sha256_file(p) != identities[str(p)] for p in sources)):
        raise ValueError("incomplete original reference source/placement/lifecycle evidence")
    summary.update(**conditions, memory=memory, status="complete_diagnostic",
                   warm_condition_median_seconds=statistics.median(r["condition_seconds"] for r in conditions["requests"][1:]))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
