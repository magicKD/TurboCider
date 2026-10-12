#!/usr/bin/env python3
"""Publish full stock ComfyUI LTX AV and reuse existing native bests."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

import numpy as np

from flux9_bf16_framework_screen import write_json
from flux9_framework_evidence import portable
from ltx_comfy_reference_screen import FILES, MODEL, validate_events, workflow
from runtime_ane_common import sha256_file
from z_comfy_framework_evidence import event_spans

ROOT = Path(__file__).resolve().parents[2]


def native_baselines(model_hashes):
    config = json.loads((ROOT / "docs/benchmark/best-local-configs-20261011.json").read_text())
    best = next(c for c in config["media_configurations"] if c["selected_route"] == "gpu-text256-batch")
    evidence_path = ROOT / best["evidence"]
    evidence = json.loads(evidence_path.read_text())
    if evidence["status"] != "complete_diagnostic":
        raise ValueError("complete existing LTX baseline required")
    for k, digest in model_hashes.items():
        if evidence["source_sha256"].get(k) != digest:
            raise ValueError("same original shared LTX model sources required")
    routes = {r["name"]: r for r in evidence["routes"]}
    quality = routes["gpu-quality-batch"]
    fastest = routes["gpu-text256-batch"]
    for r in (quality, fastest):
        if statistics.median(r["warm_seconds"]) != r["warm_median"]:
            raise ValueError("baseline must derive from original actual full-media samples")
    if not math.isclose(fastest["warm_median"], best["warm_median_seconds"], abs_tol=1e-6, rel_tol=0):
        raise ValueError("preserved fastest native configuration differs")
    return dict(evidence=best["evidence"], evidence_sha256=sha256_file(evidence_path),
        quality_seconds=quality["warm_median"], fastest_opt_in_seconds=fastest["warm_median"],
        quality_samples=quality["warm_seconds"], fastest_samples=fastest["warm_seconds"],
        fastest_quality=best["media_quality"],
        scope="existing original full97frame/11step AV batch conditioning-cache-hit measurements; not retested/same-window paired; opt-in Sol+text256 changes math and quality")


def audit(screen):
    data = json.loads((screen / "summary.json").read_text())
    if (data["status"] != "complete_diagnostic" or data["reference_only"] is not True or
            data["report"]["status"] != "complete" or data["report"]["server_terminal"] is not True or
            len(data["report"]["runs"]) != 2 or data["memory"]["verified"]["complete"] is not True):
        raise ValueError("owned cold/warm full AV and complete memory required")
    memory = data["memory"]
    if any(sha256_file(screen / memory[k]) != memory[k + "_sha256"] for k in ("evidence", "report")):
        raise ValueError("independent memory SHA chain differs")
    aliases = {str((MODEL / p).resolve(strict=True)): "models/LTX-2.5/" + p for p in FILES.values()}
    model_hashes = {aliases[k]: v for k, v in data["model_sha256"].items()}
    baseline = native_baselines(model_hashes)
    runs, checks = [], []
    for index, original in enumerate(data["report"]["runs"]):
        if (original["index"], original["seed"]) != (index, 42 + index):
            raise ValueError("actual cold/repeated-prompt-new-seed sequence differs")
        graph = json.loads((screen / f"request-{index}.json").read_text())
        history = json.loads((screen / f"history-{index}.json").read_text())
        events = json.loads((screen / f"events-{index}.json").read_text())
        if graph != workflow(index) or history["prompt"][2] != graph or history["status"]["status_str"] != "success":
            raise ValueError("actual successful LTX graph differs")
        actual = validate_events(events, index)
        if any(original[k] != v for k, v in actual.items()) or any(e["data"].get("prompt_id") != original["prompt_id"] for e in events):
            raise ValueError("actual LTX request/step/cache chain differs")
        path = screen / f"{index}.mp4"
        if sha256_file(path) != data["media"][index]["sha256"]:
            raise ValueError("original full MP4 differs")
        probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)], text=True))
        video = next(s for s in probe["streams"] if s["codec_type"] == "video")
        audio = next(s for s in probe["streams"] if s["codec_type"] == "audio")
        if (video["width"], video["height"], int(video["nb_frames"]), video["avg_frame_rate"], video["codec_name"]) != (704, 448, 97, "24/1", "h264"):
            raise ValueError("same full97frame H264 video required")
        # Decode the complete video and audio. Check all audio samples rather
        # than only trusting metadata; this work is outside timed requests.
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], check=True)
        pcm = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "f32le", "-"])
        samples = np.frombuffer(pcm, dtype="<f4")
        if not len(samples) or not np.isfinite(samples).all():
            raise ValueError("full decoded finite audio required")
        spans = event_spans(events)
        runs.append(dict(original, stage_host_envelopes=spans,
            request_evidence_sha256={k: sha256_file(screen / f"{k}-{index}.json") for k in ("request", "history", "events")}))
        checks.append(dict(index=index, sha256=sha256_file(path), video_frames=97, width=704, height=448,
            fps="24/1", video_codec="h264", audio_codec=audio["codec_name"], sample_rate=audio["sample_rate"],
            audio_channels=audio["channels"], audio_samples=len(samples), audio_finite=True, complete_decode=True))
    warm = runs[1]["request_wall_including_export"]
    stats = data["report"]["system_stats"]
    stats["system"].pop("argv", None)
    normalized = {aliases.get(k, portable(k)): v for k, v in data["source_sha256"].items()}
    return dict(schema="tc-ltx-comfy-reference-validation-v1", date=data["date"], status="complete_diagnostic",
        qualification_passed=False, goal_complete=False, reference_only=True, same_window_native_pair=False,
        hardware="M4 Max64GiB/macOS26.6.2", evidence=str((screen / "summary.json").relative_to(ROOT)),
        evidence_sha256=sha256_file(screen / "summary.json"), model_sha256=model_hashes,
        source_snapshot_sha256=hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest(),
        comfy_commit=data["comfy_commit"], system_stats=stats, scope=data["scope"], memory=memory,
        cold_seconds=runs[0]["request_wall_including_export"], warm_seconds=warm, warm_sample_count=1,
        warm_scope="conditioning cache hit, seed43 forces both sampler stages/upscale/AV decode/export; one observation, not a warm median",
        stage_scope="WebSocket host node envelopes include load/eviction; not GPU-exclusive or added fences",
        runs=runs, output_checks=checks, existing_native_baseline=baseline,
        comparisons=dict(comfy_over_native_quality=warm / baseline["quality_seconds"],
            comfy_over_native_fastest=warm / baseline["fastest_opt_in_seconds"],
            quality_reduction_percent=100 * (1 - baseline["quality_seconds"] / warm),
            fastest_reduction_percent=100 * (1 - baseline["fastest_opt_in_seconds"] / warm)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh evidence output required")
    result = audit(args.screen.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(dict(status=result["status"], warm=result["warm_seconds"], comparisons=result["comparisons"])))


if __name__ == "__main__":
    main()
