#!/usr/bin/env python3
"""ABBA benchmark for FastVideo and TurboCider FastH3 MLX INT6.

Both runtimes use the same local ModelScope model tree, INT6/g64 checkpoint,
prompt, seed, geometry, four-step schedule, full H3 video VAE, and audio VAE.
The script records raw stdout/stderr, structured runtime results, ffprobe media
metadata, encoded-media comparison metrics, and median performance ratios.
It never downloads model assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree"
VSA_MODEL_ID = "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"
MODEL_ENDPOINT = "https://modelscope.cn"
PROFILE = "minimax-h3-fasth3-mlx-int6"
VSA_PROFILE = "minimax-h3-fasth3-mlx-int6-vsa"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_last_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in reversed(list(re.finditer(r"\{", text))):
        try:
            value, end = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and not text[end:].strip():
            return value
    raise ValueError("process output does not end with a JSON object")


def abba_schedule(blocks: int) -> list[str]:
    if blocks < 1:
        raise ValueError("ABBA blocks must be positive")
    return [runtime for _ in range(blocks)
            for runtime in ("fastvideo", "turbocider", "turbocider", "fastvideo")]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def absolute_without_resolving_symlinks(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return Path(os.path.abspath(expanded))


def validate_args(args: argparse.Namespace) -> None:
    for path, label in [
        (args.turbocider_bin, "TurboCider binary"),
        (args.python, "Python interpreter"),
        (args.fastvideo_runner, "FastVideo runner"),
        (args.model_root, "ModelScope model root"),
        (args.checkpoint, "INT6 checkpoint"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} is missing: {path}")
    expected_model = VSA_MODEL_ID if args.profile == "vsa" else MODEL_ID
    expected_profile = VSA_PROFILE if args.profile == "vsa" else PROFILE
    provenance = read_json(args.model_root / "modelscope_download.json")
    if provenance.get("model_id") != expected_model or provenance.get("endpoint") != MODEL_ENDPOINT:
        raise ValueError("model root is not the approved ModelScope FastH3 repository")
    manifest = read_json(args.checkpoint / "mlx_h3_dit.json")
    quant = manifest.get("quantization", {})
    source = manifest.get("source", {})
    if quant != {"bits": 6, "group_size": 64, "mode": "affine"}:
        raise ValueError("benchmark requires the affine INT6/g64 checkpoint")
    if source.get("repository") != expected_model:
        raise ValueError("INT6 checkpoint is not bound to the approved ModelScope repository")
    vsa = manifest.get("vsa", {})
    if args.profile == "vsa":
        if not vsa.get("capable") or int(vsa.get("num_gate_matrices", 0)) != 50:
            raise ValueError("VSA benchmark requires a 50-gate VSA-capable checkpoint")
    elif vsa.get("capable"):
        raise ValueError("dense benchmark cannot use a VSA-capable checkpoint")
    if expected_profile == VSA_PROFILE and not args.vsa:
        raise ValueError("VSA benchmark requires --vsa")
    if args.width <= 0 or args.height <= 0 or args.width % 32 or args.height % 32:
        raise ValueError("H3 width and height must be positive multiples of 32")
    if args.frames < 124 or args.frames > 362 or (args.frames - 5) % 17:
        raise ValueError("paired FastVideo benchmark frames must be 17n+5 in the 5-15 second range")
    if args.fps != 24 or args.steps != 4:
        raise ValueError("FastH3 acceptance is fixed to 24 fps and four steps")
    if not args.prompts or not args.seeds:
        raise ValueError("at least one prompt and seed are required")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required")


def fastvideo_command(args: argparse.Namespace, prompt: str, seed: int,
                      output: Path, cache: Path) -> list[str]:
    command = [
        str(args.python), str(args.fastvideo_runner),
        "--model-root", str(args.model_root),
        "--mlx-checkpoint", str(args.checkpoint),
        "--prompt", prompt,
        "--height", str(args.height), "--width", str(args.width),
        "--num-frames", str(args.frames), "--seed", str(seed),
        "--steps", str(args.steps), "--vae-dtype", "fp32",
        "--video-decode-backend", "h3-vae",
        "--prompt-cache-dir", str(cache),
        "--output-path", str(output),
    ]
    if args.profile == "vsa":
        command.extend([
            "--vsa", "--vsa-sparsity", str(args.vsa_sparsity),
            "--vsa-tile-size", str(args.vsa_tile_size),
            "--vsa-prefix-mode", args.vsa_prefix_mode,
            "--vsa-dense-first-n-steps", str(args.vsa_dense_first_n_steps),
            "--vsa-impl", args.vsa_impl,
        ])
        if args.vsa_dense_layers:
            command.extend(["--vsa-dense-layers", ",".join(map(str, args.vsa_dense_layers))])
    return command


def turbocider_request(args: argparse.Namespace, prompt: str, seed: int,
                       output: Path) -> dict[str, Any]:
    request = {
        "schema_version": 1,
        "model": VSA_PROFILE if args.profile == "vsa" else PROFILE,
        "operation": "video.generate",
        "prompt": prompt,
        "output": str(output),
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "fps": args.fps,
        "steps": args.steps,
        "seed": seed,
        "execution": "gpu",
        "residency": "component_staged",
        "audio": True,
    }
    if args.profile == "vsa":
        request.update({
            "vsa": True,
            "vsa_sparsity": args.vsa_sparsity,
            "vsa_tile_size": args.vsa_tile_size,
            "vsa_prefix_mode": args.vsa_prefix_mode,
            "vsa_dense_first_n_steps": args.vsa_dense_first_n_steps,
            "vsa_dense_layers": list(args.vsa_dense_layers),
            "vsa_impl": args.vsa_impl,
        })
    return request


def ffprobe(path: Path) -> dict[str, Any]:
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size,format_name:stream=index,codec_name,codec_type,width,height,"
        "avg_frame_rate,nb_frames,sample_rate,channels,duration",
        "-of", "json", str(path),
    ], text=True)
    return json.loads(raw)


def valid_media(probe: dict[str, Any], args: argparse.Namespace) -> bool:
    streams = probe.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    return bool(
        video and audio and video.get("codec_name") == "h264" and
        int(video.get("width", 0)) == args.width and
        int(video.get("height", 0)) == args.height and
        video.get("avg_frame_rate") == "24/1" and
        int(video.get("nb_frames", 0)) == args.frames and
        audio.get("codec_name") == "aac" and
        int(audio.get("sample_rate", 0)) == 32000 and
        int(audio.get("channels", 0)) == 2
    )


def process_tree_rss_bytes(ps_output: str, root_pid: int) -> int:
    processes: dict[int, tuple[int, int]] = {}
    for line in ps_output.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            pid, parent, rss_kib = (int(field) for field in fields)
        except ValueError:
            continue
        processes[pid] = (parent, rss_kib)
    tree = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent, _) in processes.items():
            if parent in tree and pid not in tree:
                tree.add(pid)
                changed = True
    return sum(processes[pid][1] for pid in tree if pid in processes) * 1024


def sample_process_tree_rss_bytes(root_pid: int) -> int:
    try:
        output = subprocess.check_output(
            ["ps", "-axo", "pid=,ppid=,rss="], text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return 0
    return process_tree_rss_bytes(output, root_pid)


def system_swap_used_bytes() -> int | None:
    try:
        output = subprocess.check_output(
            ["sysctl", "-n", "vm.swapusage"], text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"used\s*=\s*([0-9.]+)([KMGTP])", output)
    if not match:
        return None
    scale = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30,
             "T": 1 << 40, "P": 1 << 50}[match.group(2)]
    return int(float(match.group(1)) * scale)


def run_process(command: list[str], cwd: Path, stdout: Path, stderr: Path,
                env: dict[str, str], sample_interval_s: float = 0.5
                ) -> tuple[int, float, dict[str, Any]]:
    started = time.perf_counter()
    before_swap = system_swap_used_bytes()
    peak_rss = 0
    samples = 0
    with stdout.open("w") as out, stderr.open("w") as err:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=out, stderr=err)
        while process.poll() is None:
            peak_rss = max(peak_rss, sample_process_tree_rss_bytes(process.pid))
            samples += 1
            time.sleep(sample_interval_s)
        returncode = process.wait()
    after_swap = system_swap_used_bytes()
    return returncode, time.perf_counter() - started, {
        "process_tree_peak_rss_bytes": peak_rss,
        "rss_sample_interval_s": sample_interval_s,
        "rss_samples": samples,
        "system_swap_used_before_bytes": before_swap,
        "system_swap_used_after_bytes": after_swap,
        "system_swap_growth_bytes": (
            max(0, after_swap - before_swap)
            if before_swap is not None and after_swap is not None else None
        ),
        "scope": "sampled root process plus descendants; swap is system-wide",
    }


def runtime_total(runtime: str, result: dict[str, Any]) -> float:
    if runtime == "fastvideo":
        return float(result["timings_s"]["generate_s"])
    return float(result["timings_seconds"]["request_wall"])


def runtime_stages(runtime: str, result: dict[str, Any]) -> dict[str, float]:
    if runtime == "fastvideo":
        raw = result["timings_s"]
        return {
            "condition": float(raw["condition_s"]),
            "denoise": float(raw["denoise_s"]),
            "video_decode": float(raw["video_decode_s"]),
            "audio_decode": float(raw["audio_decode_s"]),
            "mux": float(raw["mux_s"]),
        }
    raw = result["timings_seconds"]
    return {
        "condition": float(raw["text_encode"]),
        "denoise": float(raw["denoise"]),
        "video_decode": float(raw["video_decode"]),
        "audio_decode": float(raw["audio_decode"]),
        "mux": float(raw["mux"]),
    }


def run_one(args: argparse.Namespace, runtime: str, prompt: str, seed: int,
            mode: str, ordinal: int, measured: bool) -> dict[str, Any]:
    stem = f"{mode}-{ordinal:03d}-{runtime}"
    media = args.output_dir / "media" / f"{stem}.mp4"
    stdout = args.output_dir / "logs" / f"{stem}.stdout"
    stderr = args.output_dir / "logs" / f"{stem}.stderr"
    cache = args.output_dir / "cache" / runtime / (
        "warm" if mode == "warm" else f"cold-{ordinal:03d}")
    cache.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    if runtime == "fastvideo":
        command = fastvideo_command(args, prompt, seed, media, cache)
    else:
        request_path = args.output_dir / "requests" / f"{stem}.json"
        request_path.write_text(json.dumps(turbocider_request(args, prompt, seed, media), indent=2) + "\n")
        env["TURBOCIDER_H3_PROMPT_CACHE_DIR"] = str(cache)
        command = [str(args.turbocider_bin), "generate", str(args.model_root), str(request_path)]
    code, wall, process_memory = run_process(command, ROOT, stdout, stderr, env)
    record: dict[str, Any] = {
        "runtime": runtime,
        "mode": mode,
        "ordinal": ordinal,
        "measured": measured,
        "prompt": prompt,
        "seed": seed,
        "command": command,
        "returncode": code,
        "external_wall_s": wall,
        "process_memory": process_memory,
        "stdout": str(stdout),
        "stderr": str(stderr),
        "media": str(media),
    }
    if code:
        record["error_tail"] = stderr.read_text(errors="replace")[-4000:]
        return record
    result = extract_last_json(stdout.read_text(errors="replace"))
    probe = ffprobe(media)
    record.update({
        "result": result,
        "internal_total_s": runtime_total(runtime, result),
        "stages_s": runtime_stages(runtime, result),
        "media_probe": probe,
        "media_valid": valid_media(probe, args),
        "media_sha256": sha256_file(media),
    })
    return record


def filter_metric(reference: Path, candidate: Path, name: str) -> float:
    expression = (
        f"[0:v]settb=AVTB,setpts=N/(24*TB)[a];"
        f"[1:v]settb=AVTB,setpts=N/(24*TB)[b];[a][b]{name}"
    )
    process = subprocess.run([
        "ffmpeg", "-v", "info", "-i", str(reference), "-i", str(candidate),
        "-lavfi", expression, "-f", "null", "-",
    ], text=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    if name == "psnr":
        match = re.search(r"average:([0-9.]+)", process.stderr)
    else:
        match = re.search(r"All:([0-9.]+)", process.stderr)
    if not match:
        raise ValueError(f"could not parse {name} output")
    return float(match.group(1))


def decoded_audio(path: Path) -> np.ndarray:
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0",
        "-f", "f32le", "-acodec", "pcm_f32le", "-ar", "32000", "-ac", "2", "-",
    ])
    return np.frombuffer(raw, dtype="<f4").reshape(-1, 2).astype(np.float64)


def raw_video_metrics(reference: Path, candidate: Path, width: int,
                      height: int, frames: int) -> dict[str, float]:
    def command(path: Path) -> list[str]:
        return [
            "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
            "-frames:v", str(frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]

    left = subprocess.Popen(command(reference), stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    right = subprocess.Popen(command(candidate), stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL)
    assert left.stdout is not None and right.stdout is not None
    frame_bytes = width * height * 3
    sum_difference_squared = 0.0
    sum_reference_squared = 0.0
    sum_candidate_squared = 0.0
    dot = 0.0
    maximum = 0
    try:
        for index in range(frames):
            raw_left = left.stdout.read(frame_bytes)
            raw_right = right.stdout.read(frame_bytes)
            if len(raw_left) != frame_bytes or len(raw_right) != frame_bytes:
                raise ValueError(
                    f"decoded video ended before frame {index + 1}/{frames}")
            reference_frame = np.frombuffer(raw_left, dtype=np.uint8)
            candidate_frame = np.frombuffer(raw_right, dtype=np.uint8)
            difference = (reference_frame.astype(np.int16) -
                          candidate_frame.astype(np.int16))
            maximum = max(maximum, int(np.max(np.abs(difference))))
            difference64 = difference.astype(np.float64)
            reference64 = reference_frame.astype(np.float64)
            candidate64 = candidate_frame.astype(np.float64)
            sum_difference_squared += float(np.dot(difference64, difference64))
            sum_reference_squared += float(np.dot(reference64, reference64))
            sum_candidate_squared += float(np.dot(candidate64, candidate64))
            dot += float(np.dot(reference64, candidate64))
    finally:
        left.stdout.close()
        right.stdout.close()
    if left.wait() or right.wait():
        raise RuntimeError("ffmpeg raw video decode failed")
    samples = frame_bytes * frames
    mean_squared = sum_difference_squared / samples
    denominator = math.sqrt(sum_reference_squared * sum_candidate_squared)
    return {
        "raw_video_max_abs": float(maximum),
        "raw_video_rmse": math.sqrt(mean_squared),
        "raw_video_psnr_db": (
            20.0 * math.log10(255.0 / math.sqrt(mean_squared))
            if mean_squared else float("inf")
        ),
        "raw_video_cosine": dot / denominator if denominator else 1.0,
    }


def media_comparison(reference: Path, candidate: Path, fps: int) -> dict[str, Any]:
    # Keep encoded PSNR/SSIM for compatibility with earlier reports, but also
    # compare the same raw RGB frame stream that the product quality gate uses.
    reference_probe = ffprobe(reference)
    video_stream = next(stream for stream in reference_probe["streams"]
                        if stream.get("codec_type") == "video")
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    frames = int(video_stream["nb_frames"])
    video_metrics = raw_video_metrics(reference, candidate, width, height, frames)
    reference_audio = decoded_audio(reference)
    candidate_audio = decoded_audio(candidate)
    reference_samples = len(reference_audio)
    candidate_samples = len(candidate_audio)
    count = min(reference_samples, candidate_samples)
    left, right = reference_audio[:count], candidate_audio[:count]
    difference = left - right
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return {
        "encoded_video_psnr_db": filter_metric(reference, candidate, "psnr"),
        "encoded_video_ssim": filter_metric(reference, candidate, "ssim"),
        **video_metrics,
        "audio_reference_samples": int(reference_samples),
        "audio_candidate_samples": int(candidate_samples),
        "audio_duration_difference_s": abs(reference_samples - candidate_samples) / 32000,
        "audio_cosine": float(np.sum(left * right) / denominator) if denominator else 1.0,
        "audio_rmse": float(np.sqrt(np.mean(difference * difference))),
        "within_one_frame": abs(reference_samples - candidate_samples) / 32000 <= 1 / fps,
    }


def summarize(report: dict[str, Any], args: argparse.Namespace) -> None:
    comparisons: dict[str, Any] = {}
    measured = [run for run in report["runs"] if run.get("measured") and not run.get("returncode")]
    for mode in args.modes:
        by_runtime = {
            runtime: [run for run in measured if run["mode"] == mode and run["runtime"] == runtime]
            for runtime in ("fastvideo", "turbocider")
        }
        if not all(by_runtime.values()):
            continue
        medians = {
            runtime: statistics.median(run["internal_total_s"] for run in runs)
            for runtime, runs in by_runtime.items()
        }
        stage_medians = {
            runtime: {
                stage: statistics.median(run["stages_s"][stage] for run in runs)
                for stage in ("condition", "denoise", "video_decode", "audio_decode", "mux")
            }
            for runtime, runs in by_runtime.items()
        }
        rss_medians = {
            runtime: statistics.median(
                int(run.get("process_memory", {}).get(
                    "process_tree_peak_rss_bytes", 0)) for run in runs
            )
            for runtime, runs in by_runtime.items()
        }
        swap_growth_max = {
            runtime: max(
                int(run.get("process_memory", {}).get(
                    "system_swap_growth_bytes") or 0) for run in runs
            )
            for runtime, runs in by_runtime.items()
        }
        ratio = medians["turbocider"] / medians["fastvideo"]
        comparisons[mode] = {
            "runs_per_runtime": {runtime: len(runs) for runtime, runs in by_runtime.items()},
            "median_total_s": medians,
            "turbocider_over_fastvideo": ratio,
            "passes_max_overhead": ratio <= args.max_overhead_ratio,
            "stage_median_s": stage_medians,
            "process_tree_peak_rss_median_bytes": rss_medians,
            "system_swap_growth_max_bytes": swap_growth_max,
        }
    report["comparison"] = comparisons
    report["passed"] = bool(comparisons) and all(
        value["passes_max_overhead"] for value in comparisons.values()) and all(
            run.get("media_valid", False) for run in measured)


def hardware() -> dict[str, Any]:
    def sysctl(name: str) -> str:
        try:
            return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "model": sysctl("hw.model"),
        "memory_bytes": int(sysctl("hw.memsize")) if sysctl("hw.memsize").isdigit() else None,
        "cpu": sysctl("machdep.cpu.brand_string"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turbocider-bin", type=Path, default=ROOT / "build/native/turbocider")
    parser.add_argument("--python", type=Path, default=ROOT / "Python/bin/python3")
    parser.add_argument("--fastvideo-runner", type=Path,
                        default=ROOT / "tools/h3/run_fastvideo_mlx_fasth3.py")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", choices=("dense", "vsa"), default="dense")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", dest="prompts", action="append")
    parser.add_argument("--seed", dest="seeds", action="append", type=int)
    parser.add_argument("--mode", dest="modes", action="append", choices=("cold", "warm"))
    parser.add_argument("--blocks", type=int, default=1,
                        help="ABBA blocks; one block runs each runtime twice")
    parser.add_argument("--warmups", type=int, default=1,
                        help="excluded warm-cache runs per runtime before measured runs")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=124)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--vsa-sparsity", type=float, default=0.9)
    parser.add_argument("--vsa-tile-size", type=int, choices=(64, 256), default=64)
    parser.add_argument("--vsa-prefix-mode", choices=("exempt", "compete"), default="exempt")
    parser.add_argument("--vsa-dense-first-n-steps", type=int, default=0)
    parser.add_argument("--vsa-dense-layers", type=int, action="append", default=[])
    parser.add_argument("--vsa-impl", choices=("auto", "reference", "simd"), default="reference")
    parser.add_argument("--vsa", action="store_true",
                        help="enable VSA for --profile vsa")
    parser.add_argument("--max-overhead-ratio", type=float, default=1.15)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-media", action="store_true")
    args = parser.parse_args()
    args.prompts = args.prompts or ["A red fox runs through fresh snow."]
    args.seeds = args.seeds or [2026]
    args.modes = args.modes or ["warm"]
    for name in ("turbocider_bin", "python", "fastvideo_runner", "model_root",
                 "checkpoint", "output_dir"):
        setattr(args, name, absolute_without_resolving_symlinks(getattr(args, name)))
    return args


def main() -> None:
    args = parse_args()
    validate_args(args)
    for child in ("media", "logs", "requests", "cache"):
        (args.output_dir / child).mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.checkpoint / "mlx_h3_dit.json")
    model_id = VSA_MODEL_ID if args.profile == "vsa" else MODEL_ID
    report: dict[str, Any] = {
        "schema": "turbocider-fasth3-mlx-abba-v2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hardware": hardware(),
        "fastvideo_git": subprocess.check_output(
            ["git", "-C", str(ROOT.parent / "references/FastVideo"), "rev-parse", "HEAD"], text=True).strip(),
        "model": {
            "repository": model_id,
            "profile": VSA_PROFILE if args.profile == "vsa" else PROFILE,
            "endpoint": MODEL_ENDPOINT,
            "revision": manifest.get("source", {}).get("revision"),
            "checkpoint_sha256": manifest.get("source", {}).get("checkpoint_sha256"),
            "quantization": manifest.get("quantization"),
        },
        "geometry": {"width": args.width, "height": args.height, "frames": args.frames,
                     "fps": args.fps, "steps": args.steps},
        "prompts": args.prompts,
        "seeds": args.seeds,
        "vsa": {
            "enabled": args.profile == "vsa",
            "sparsity": args.vsa_sparsity,
            "tile_size": args.vsa_tile_size,
            "prefix_mode": args.vsa_prefix_mode,
            "dense_first_n_steps": args.vsa_dense_first_n_steps,
            "dense_layers": args.vsa_dense_layers,
            "impl": args.vsa_impl,
        },
        "runs": [],
    }
    report_path = args.output_dir / "report.json"
    ordinal = 0
    for prompt in args.prompts:
        for seed in args.seeds:
            if "warm" in args.modes:
                for _ in range(args.warmups):
                    for runtime in ("fastvideo", "turbocider"):
                        ordinal += 1
                        run = run_one(args, runtime, prompt, seed, "warm", ordinal, False)
                        report["runs"].append(run)
                        report_path.write_text(json.dumps(report, indent=2) + "\n")
                        if run["returncode"]:
                            raise RuntimeError(f"{runtime} warmup failed: {run.get('error_tail')}")
            for mode in args.modes:
                for runtime in abba_schedule(args.blocks):
                    ordinal += 1
                    run = run_one(args, runtime, prompt, seed, mode, ordinal, True)
                    report["runs"].append(run)
                    summarize(report, args)
                    report_path.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps({
                        "runtime": runtime, "mode": mode, "ordinal": ordinal,
                        "seconds": run.get("internal_total_s"),
                        "returncode": run["returncode"],
                    }), flush=True)
                    if run["returncode"]:
                        raise RuntimeError(f"{runtime} benchmark failed: {run.get('error_tail')}")

    first = {}
    for run in report["runs"]:
        if run.get("measured") and not run.get("returncode"):
            first.setdefault(run["runtime"], Path(run["media"]))
    if set(first) == {"fastvideo", "turbocider"}:
        report["encoded_media_comparison"] = media_comparison(
            first["fastvideo"], first["turbocider"], args.fps)
    summarize(report, args)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if not args.keep_media:
        for run in report["runs"]:
            Path(run["media"]).unlink(missing_ok=True)
    print(json.dumps({"report": str(report_path), "passed": report["passed"],
                      "comparison": report.get("comparison", {})}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
