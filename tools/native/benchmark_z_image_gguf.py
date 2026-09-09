#!/usr/bin/env python3
"""Interleaved benchmark for TurboCider's resident Z-Image GGUF backends.

The reference side always uses the pinned Unsloth stable-diffusion.cpp
``sd-server``.  The TurboCider side can use either the matching managed
sd.cpp bridge or the native MLX GGUF implementation.  Both sides stay resident
and requests are issued serially in an explicit order (``ABBA`` by default),
so backend choice and the public C ABI overhead are recorded explicitly.

The script has no Pillow or NumPy dependency. It records PNG SHA-256 values
and decodes ordinary 8-bit, non-interlaced PNGs for pixel parity metrics.
"""
from __future__ import annotations

import argparse
import base64
import ctypes as C
import hashlib
import json
import math
import os
import shutil
import socket
import statistics
import struct
import subprocess
import tempfile
import time
import urllib.request
import zlib
from pathlib import Path
from typing import Optional


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(url: str, method: str = "GET", value: Optional[dict] = None) -> dict:
    body = None
    headers = {}
    if value is not None:
        body = json.dumps(value).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def wait_server(process: subprocess.Popen, base: str) -> None:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"sd-server exited with {process.returncode}")
        try:
            if http_json(base + "/v1/models"):
                return
        except Exception:
            time.sleep(0.25)
    raise TimeoutError("sd-server did not become ready")


def direct_generate(base: str, payload: dict, output: Path) -> float:
    started = time.perf_counter()
    job = http_json(base + "/sdcpp/v1/img_gen", "POST", payload)
    job_id = job.get("id")
    if not job_id:
        raise RuntimeError(f"sd-server returned no job id: {job}")
    while True:
        status = http_json(base + f"/sdcpp/v1/jobs/{job_id}")
        if status.get("status") == "completed":
            images = status.get("result", {}).get("images", [])
            blob = images[0].get("b64_json") if images else None
            if not blob:
                raise RuntimeError(f"sd-server completed without image: {status}")
            output.write_bytes(base64.b64decode(blob))
            return time.perf_counter() - started
        if status.get("status") in {"failed", "cancelled"}:
            raise RuntimeError(f"sd-server generation failed: {status}")
        # Match TurboCider's resident sd.cpp bridge. A 400 ms cadence made the
        # completion-detection jitter much larger than the sub-percent bridge
        # overhead this benchmark is intended to gate.
        time.sleep(0.1)


def consume(lib, pointer):
    if not pointer.value:
        return None
    value = C.string_at(pointer).decode()
    lib.tc_string_free(pointer)
    return value


def turbo_generate(lib, engine, args, output: Path) -> tuple[float, dict]:
    value = {
        "model": "z-image-turbo-gguf",
        "model_variant": args.variant,
        "operation": "image.generate",
        "execution": "gpu",
        "residency": "resident",
        "prompt": args.prompt,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "seed": args.seed,
        "frames": 1,
        "audio": False,
        "output": str(output),
    }
    if args.lora:
        value["loras"] = [{
            "path": str(args.lora.resolve()),
            "strength": args.lora_strength,
            "role": "transformer",
        }]
    result, error = C.c_void_p(), C.c_void_p()
    started = time.perf_counter()
    status = lib.tc_engine_generate(
        engine, json.dumps(value).encode(), None, None, C.byref(result), C.byref(error)
    )
    report, message = consume(lib, result), consume(lib, error)
    if status:
        raise RuntimeError(message)
    return time.perf_counter() - started, json.loads(report)


def resolve_gguf(root: Path, variant: str) -> Path:
    values = sorted(
        path.resolve()
        for path in ([root] if root.is_file() else root.rglob("*.gguf"))
        if "lora" not in path.name.lower()
    )
    wanted = variant.lower()
    matches = [path for path in values if wanted in path.name.lower()]
    if len(matches) != 1:
        names = ", ".join(path.name for path in values) or "none"
        raise SystemExit(f"variant {variant!r} matched {len(matches)} GGUF files; found: {names}")
    return matches[0]


def component(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise SystemExit(f"missing required Z-Image component: {path}")
    return path.resolve()


def resolve_component_root(model_root: Path, configured: Optional[Path]) -> Path:
    root = configured.resolve() if configured else (
        model_root if model_root.is_dir() else model_root.parent
    )
    if not root.is_dir():
        raise SystemExit(f"Z-Image component root is missing: {root}")
    return root


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    distances = (abs(estimate - left), abs(estimate - above), abs(estimate - upper_left))
    return (left, above, upper_left)[distances.index(min(distances))]


def decode_png(path: Path) -> tuple[int, int, int, bytes]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    offset = 8
    width = height = color_type = bit_depth = interlace = -1
    compressed = bytearray()
    while offset < len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        value = data[offset + 8:offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", value)
        elif kind == b"IDAT":
            compressed.extend(value)
        elif kind == b"IEND":
            break
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if bit_depth != 8 or interlace != 0 or channels is None:
        raise ValueError(
            f"unsupported PNG encoding in {path}: depth={bit_depth}, color={color_type}, interlace={interlace}"
        )
    raw = zlib.decompress(compressed)
    stride = width * channels
    expected = height * (stride + 1)
    if len(raw) != expected:
        raise ValueError(f"PNG payload size mismatch in {path}: {len(raw)} != {expected}")
    pixels = bytearray(height * stride)
    previous = bytearray(stride)
    source = 0
    for row in range(height):
        filter_type = raw[source]
        source += 1
        encoded = raw[source:source + stride]
        source += stride
        decoded = bytearray(stride)
        for index, byte in enumerate(encoded):
            left = decoded[index - channels] if index >= channels else 0
            above = previous[index]
            upper_left = previous[index - channels] if index >= channels else 0
            if filter_type == 0:
                prediction = 0
            elif filter_type == 1:
                prediction = left
            elif filter_type == 2:
                prediction = above
            elif filter_type == 3:
                prediction = (left + above) // 2
            elif filter_type == 4:
                prediction = paeth(left, above, upper_left)
            else:
                raise ValueError(f"invalid PNG filter {filter_type} in {path}")
            decoded[index] = (byte + prediction) & 255
        begin = row * stride
        pixels[begin:begin + stride] = decoded
        previous = decoded
    return width, height, channels, bytes(pixels)


def pixel_metrics(reference: Path, candidate: Path) -> dict:
    a = decode_png(reference)
    b = decode_png(candidate)
    if a[:3] != b[:3]:
        raise ValueError(f"PNG shape mismatch: {reference} {a[:3]} != {candidate} {b[:3]}")
    left, right = a[3], b[3]
    count = len(left)
    absolute = squared = dot = left_sq = right_sq = 0.0
    left_sum = right_sum = 0.0
    for x, y in zip(left, right):
        delta = float(x) - float(y)
        absolute += abs(delta)
        squared += delta * delta
        dot += float(x) * float(y)
        left_sq += float(x) * float(x)
        right_sq += float(y) * float(y)
        left_sum += x
        right_sum += y
    left_mean = left_sum / count
    right_mean = right_sum / count
    covariance = left_variance = right_variance = 0.0
    for x, y in zip(left, right):
        dx = float(x) - left_mean
        dy = float(y) - right_mean
        covariance += dx * dy
        left_variance += dx * dx
        right_variance += dy * dy
    return {
        "width": a[0],
        "height": a[1],
        "channels": a[2],
        # Decoded 8-bit PNG samples are finite by construction. Keep these
        # contract fields explicit so benchmark summaries can distinguish
        # structural correctness from numerical similarity.
        "shape_equal": True,
        "finite": True,
        "pixel_exact": left == right,
        "mae_255": absolute / count,
        "rmse_255": math.sqrt(squared / count),
        "cosine": dot / math.sqrt(left_sq * right_sq) if left_sq and right_sq else 1.0,
        "correlation": covariance / math.sqrt(left_variance * right_variance)
        if left_variance and right_variance else 1.0,
    }


def configure_library(path: Path):
    lib = C.CDLL(str(path))
    lib.tc_string_free.argtypes = [C.c_void_p]
    lib.tc_string_free.restype = None
    lib.tc_engine_create_model.argtypes = [
        C.c_char_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)
    ]
    lib.tc_engine_create_model.restype = C.c_int
    lib.tc_engine_generate.argtypes = [
        C.c_void_p, C.c_char_p, C.c_void_p, C.c_void_p,
        C.POINTER(C.c_void_p), C.POINTER(C.c_void_p),
    ]
    lib.tc_engine_generate.restype = C.c_int
    lib.tc_engine_load.argtypes = [
        C.c_void_p, C.c_void_p, C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)
    ]
    lib.tc_engine_load.restype = C.c_int
    lib.tc_engine_free.argtypes = [C.c_void_p]
    lib.tc_engine_free.restype = None
    return lib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument(
        "--component-root", type=Path,
        help="root containing split_files/vae and split_files/text_encoders; "
             "defaults to --model-root",
    )
    parser.add_argument("--variant", default="Q4_K_M")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--server", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--turbocider-backend", choices=("sd-cpp", "native-mlx"), default="sd-cpp",
        help="TurboCider implementation to compare with the direct Unsloth sd-server",
    )
    parser.add_argument(
        "--native-component-root", type=Path,
        help="Z-Image tokenizer/Qwen3/VAE root used by --turbocider-backend=native-mlx",
    )
    parser.add_argument("--order", default="ABBA", help="A=direct Unsloth sd-server, B=TurboCider")
    parser.add_argument("--runs", type=int, default=1, help="number of repetitions of --order")
    parser.add_argument("--warmups", type=int, default=1, help="unmeasured requests per resident backend")
    parser.add_argument(
        "--max-overhead-ratio", type=float, default=1.0,
        help="required TurboCider/reference median ratio; defaults to the strict not-slower gate",
    )
    parser.add_argument(
        "--noise-tolerance-ratio", type=float, default=1.02,
        help="secondary diagnostic threshold; it never replaces the required performance gate",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="A small red fox sitting in fresh snow, soft morning light")
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--lora-strength", type=float, default=1.0)
    args = parser.parse_args()
    if args.runs < 1 or args.warmups < 0:
        raise SystemExit("runs must be positive and warmups must be nonnegative")
    order = args.order.upper()
    if not order or set(order) - {"A", "B"} or "A" not in order or "B" not in order:
        raise SystemExit("order must contain only A/B and must exercise both backends")

    root = args.model_root.resolve()
    gguf = resolve_gguf(root, args.variant)
    model_file_root = root if root.is_dir() else root.parent
    component_root = resolve_component_root(root, args.component_root)
    vae = component(component_root, "split_files/vae/ae.safetensors")
    llm = component(component_root, "split_files/text_encoders/qwen_3_4b.safetensors")
    server = args.server
    if server is None:
        configured = os.environ.get("TURBOCIDER_SD_CPP_BIN")
        server = Path(configured) if configured else model_file_root / "bin/sd-server"
    server = server.resolve()
    library = args.library.resolve()
    if not server.is_file() or not os.access(server, os.X_OK):
        raise SystemExit(f"sd-server is missing or not executable: {server}")
    if not library.is_file():
        raise SystemExit(f"TurboCider library is missing: {library}")
    if args.lora and not args.lora.is_file():
        raise SystemExit(f"LoRA is missing: {args.lora}")
    args.output.mkdir(parents=True, exist_ok=True)

    # Passing the exact GGUF file keeps tc_engine_load from warming a different
    # quant when the directory contains multiple variants.
    os.environ["TURBOCIDER_SD_CPP_BIN"] = str(server)
    os.environ["TURBOCIDER_Z_IMAGE_VAE"] = str(vae)
    os.environ["TURBOCIDER_Z_IMAGE_LLM"] = str(llm)
    if args.turbocider_backend == "native-mlx":
        os.environ["TURBOCIDER_Z_GGUF_NATIVE_GPU"] = "1"
        if args.native_component_root:
            native_root = args.native_component_root.resolve()
            if not native_root.is_dir():
                raise SystemExit(f"native component root is missing: {native_root}")
            os.environ["TURBOCIDER_Z_IMAGE_NATIVE_ROOT"] = str(native_root)
    else:
        os.environ.pop("TURBOCIDER_Z_GGUF_NATIVE_GPU", None)
    lib = configure_library(library)
    engine, error = C.c_void_p(), C.c_void_p()
    status = lib.tc_engine_create_model(
        b"z-image-turbo-gguf", str(gguf).encode(), C.byref(engine), C.byref(error)
    )
    message = consume(lib, error)
    if status:
        raise RuntimeError(message)

    reference_times: list[float] = []
    turbo_times: list[float] = []
    reference_outputs: list[Path] = []
    turbo_outputs: list[Path] = []
    reports: list[dict] = []
    sequence: list[dict] = []
    load_report = None
    process = None
    log = None
    try:
        with tempfile.TemporaryDirectory(prefix="tc-gguf-bench-") as temporary:
            scratch = Path(temporary)
            direct_lora = None
            if args.lora:
                direct_lora = scratch / ("benchmark_lora" + args.lora.suffix.lower())
                try:
                    direct_lora.symlink_to(args.lora.resolve())
                except OSError:
                    shutil.copy2(args.lora, direct_lora)
            payload = {
                "prompt": args.prompt,
                "negative_prompt": "",
                "width": args.width,
                "height": args.height,
                "batch_count": 1,
                "output_format": "png",
                "seed": args.seed,
                "sample_params": {"sample_steps": args.steps},
            }
            if direct_lora:
                payload["lora"] = [{
                    "path": direct_lora.name,
                    "multiplier": args.lora_strength,
                }]

            server_port = free_port()
            log_path = args.output / "reference-sd-server.log"
            log = log_path.open("wb")
            direct_load_started = time.perf_counter()
            process = subprocess.Popen(
                [str(server), "--diffusion-model", str(gguf), "--vae", str(vae), "--llm", str(llm),
                 "--listen-ip", "127.0.0.1", "--listen-port", str(server_port),
                 "--lora-model-dir", str(scratch), "--hires-upscalers-dir", str(scratch),
                 "--embd-dir", str(scratch), "--diffusion-fa", "--diffusion-conv-direct",
                 "--clip-on-cpu", "--cfg-scale", "1.0"],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            base = f"http://127.0.0.1:{server_port}"
            wait_server(process, base)
            reference_load = time.perf_counter() - direct_load_started

            loaded, error = C.c_void_p(), C.c_void_p()
            turbo_load_started = time.perf_counter()
            status = lib.tc_engine_load(engine, None, None, C.byref(loaded), C.byref(error))
            turbo_load = time.perf_counter() - turbo_load_started
            load_value, message = consume(lib, loaded), consume(lib, error)
            if status:
                raise RuntimeError(message)
            load_report = json.loads(load_value)

            for index in range(args.warmups):
                direct_generate(base, payload, args.output / f"warmup-reference-{index}.png")
                turbo_generate(lib, engine, args, args.output / f"warmup-turbocider-{index}.png")

            counters = {"A": 0, "B": 0}
            for symbol in order * args.runs:
                backend = "reference" if symbol == "A" else "turbocider"
                index = counters[symbol]
                counters[symbol] += 1
                output = args.output / f"{backend}-{index:03d}.png"
                if symbol == "A":
                    elapsed = direct_generate(base, payload, output)
                    reference_times.append(elapsed)
                    reference_outputs.append(output)
                    report = None
                else:
                    elapsed, report = turbo_generate(lib, engine, args, output)
                    turbo_times.append(elapsed)
                    turbo_outputs.append(output)
                    reports.append(report)
                sequence.append({
                    "backend": backend,
                    "index": index,
                    "seconds": elapsed,
                    "output": output.name,
                    "sha256": sha256(output),
                })
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if log is not None:
            log.close()
        lib.tc_engine_free(engine)

    parity = []
    for reference, candidate in zip(reference_outputs, turbo_outputs):
        metrics = pixel_metrics(reference, candidate)
        metrics.update({
            "reference": reference.name,
            "turbocider": candidate.name,
            "reference_sha256": sha256(reference),
            "turbocider_sha256": sha256(candidate),
        })
        parity.append(metrics)
    reference_median = statistics.median(reference_times)
    turbo_median = statistics.median(turbo_times)
    ratio = turbo_median / reference_median
    result = {
        "schema_version": 2,
        "model": str(gguf),
        "model_sha256": sha256(gguf),
        "variant": args.variant,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "seed": args.seed,
        "order": order,
        "order_repetitions": args.runs,
        "warmups_per_backend": args.warmups,
        "lora": None if not args.lora else {
            "path": str(args.lora.resolve()),
            "sha256": sha256(args.lora),
            "strength": args.lora_strength,
        },
        "runtime": {
            "sd_server": str(server),
            "sd_server_sha256": sha256(server),
            "flags": ["--diffusion-fa", "--diffusion-conv-direct", "--clip-on-cpu", "--cfg-scale", "1.0"],
            "requested_turbocider_backend": args.turbocider_backend,
            "observed_turbocider_backend": (
                reports[0].get("runtime_backend") if reports else "unknown"
            ),
        },
        # Readiness includes process launch, argument validation and the
        # backend's model-open path. It is not proof that the operating system
        # has faulted every lazily mapped weight page into memory.
        "reference_server_ready_seconds": reference_load,
        "turbocider_server_ready_seconds": turbo_load,
        "turbocider_load_report": load_report,
        "sequence": sequence,
        "reference_seconds": reference_times,
        "turbocider_seconds": turbo_times,
        "reference_median_seconds": reference_median,
        "turbocider_median_seconds": turbo_median,
        "turbocider_over_reference": ratio,
        "performance_gate": {
            "requirement": "TurboCider warm median must not exceed the matched reference median",
            "maximum_ratio": args.max_overhead_ratio,
            "passed": ratio <= args.max_overhead_ratio,
            "measurement_noise_tolerance_ratio": args.noise_tolerance_ratio,
            "within_measurement_noise": ratio <= args.noise_tolerance_ratio,
        },
        "parity": parity,
        "parity_gate": {
            "all_pixel_exact": bool(parity) and all(item["pixel_exact"] for item in parity),
            "minimum_correlation": min(item["correlation"] for item in parity),
            "maximum_mae_255": max(item["mae_255"] for item in parity),
        },
        "reports": reports,
        "note": (
            "Both backends stayed resident simultaneously and were invoked serially in the recorded order. "
            + (
                "Both sides use separate processes with the same pinned Unsloth sd-server, files and Metal flags."
                if args.turbocider_backend == "sd-cpp"
                else "The reference uses the pinned Unsloth sd-server while TurboCider uses its native MLX GGUF runtime."
            )
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["performance_gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
