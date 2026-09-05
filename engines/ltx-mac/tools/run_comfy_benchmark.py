#!/usr/bin/env python3
"""Submit a matched LTX-2.5 workflow to a running ComfyUI server.

The local workflow is treated as a template and is never modified in place.
Defaults match the native ltx-mac 480p-class workload: a 704x480 request,
97 frames at 24 FPS, distilled 8+3 schedules, deterministic Stage 2, and the
conv video VAE.  Re-run with a different seed to measure a warm generation
while allowing ComfyUI to reuse prompt conditioning and resident models.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


STAGE1_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
STAGE2_SIGMAS = "0.909375, 0.725, 0.421875, 0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=97)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--video-vae",
        default="ltx-2.5-video-vae-conv-bf16.safetensors",
    )
    parser.add_argument("--output-prefix", default="video/ltx25_matched")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--keep-audio", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require_node(prompt: dict[str, Any], node_id: str) -> dict[str, Any]:
    try:
        node = prompt[node_id]
    except KeyError as exc:
        raise ValueError(f"workflow is missing expected node {node_id}") from exc
    if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
        raise ValueError(f"workflow node {node_id} is malformed")
    return node


def prepare_prompt(document: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    source = document.get("prompt", document)
    if not isinstance(source, dict):
        raise ValueError("workflow JSON has no prompt object")
    prompt = copy.deepcopy(source)

    if args.width <= 0 or args.height <= 0 or args.frames <= 1 or args.fps <= 0:
        raise ValueError("width/height/fps must be positive and frames must exceed one")
    duration = (args.frames - 1) / args.fps
    if abs(round(duration * args.fps) + 1 - args.frames) > 0:
        raise ValueError("frames must equal duration * fps + 1")

    require_node(prompt, "372")["inputs"]["value"] = args.width
    require_node(prompt, "360")["inputs"]["value"] = args.height
    require_node(prompt, "361")["inputs"]["value"] = args.fps
    require_node(prompt, "362")["inputs"]["value"] = duration
    prompt_value = require_node(prompt, "376")["inputs"].get("value")
    if args.prompt is not None:
        prompt_value = args.prompt
    if not isinstance(prompt_value, str):
        raise ValueError("workflow prompt node 376 has no string value")

    # The saved workflow contains an optional prompt-enhancement branch.  Its
    # separate Gemma checkpoint is not needed for the matched inference run,
    # but ComfyUI validates it even when the switch is false.  Collapse the
    # branch to the literal prompt so only the production LTX text encoder is
    # part of the graph.
    prompt_source = require_node(prompt, "382")
    prompt_source["class_type"] = "PrimitiveStringMultiline"
    prompt_source["inputs"] = {"value": prompt_value}
    for unused_node in ("376", "380", "383", "393"):
        prompt.pop(unused_node, None)

    require_node(prompt, "339")["inputs"]["noise_seed"] = args.seed
    require_node(prompt, "338")["inputs"]["noise_seed"] = args.seed + 2
    require_node(prompt, "404")["inputs"]["sigmas"] = STAGE1_SIGMAS
    require_node(prompt, "395")["inputs"]["sigmas"] = STAGE2_SIGMAS
    require_node(prompt, "352")["inputs"]["sampler_name"] = "euler_ancestral"
    require_node(prompt, "341")["inputs"]["sampler_name"] = "euler"

    require_node(prompt, "385")["inputs"]["vae_name"] = args.video_vae
    decode = require_node(prompt, "374")
    decode["class_type"] = "VAEDecode"
    decode["inputs"] = {"samples": ["369", 0], "vae": ["385", 0]}

    create_video = require_node(prompt, "370")
    if not args.keep_audio:
        create_video["inputs"].pop("audio", None)
    require_node(prompt, "save")["inputs"]["filename_prefix"] = args.output_prefix
    return prompt


def request_json(url: str, payload: dict[str, Any] | None = None) -> Any:
    data = None
    headers: dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30.0) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI HTTP {exc.code}: {body}") from exc


def output_files(history: dict[str, Any]) -> list[str]:
    files: list[str] = []
    outputs = history.get("outputs", {})
    if not isinstance(outputs, dict):
        return files
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        for value in node_output.values():
            if not isinstance(value, list):
                continue
            for item in value:
                if isinstance(item, dict) and isinstance(item.get("filename"), str):
                    files.append(item["filename"])
    return files


def main() -> None:
    args = parse_args()
    document = json.loads(args.workflow.read_text(encoding="utf-8"))
    prompt = prepare_prompt(document, args)
    if args.dry_run:
        print(json.dumps(prompt, indent=2, sort_keys=True))
        return

    client_id = f"ltx-mac-bench-{uuid.uuid4()}"
    started = time.monotonic()
    response = request_json(
        f"{args.server.rstrip('/')}/prompt",
        {"prompt": prompt, "client_id": client_id},
    )
    prompt_id = response.get("prompt_id")
    if not isinstance(prompt_id, str):
        raise RuntimeError(f"ComfyUI did not return a prompt id: {response!r}")
    print(f"prompt_id={prompt_id}", flush=True)

    deadline = started + args.timeout
    history: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        result = request_json(f"{args.server.rstrip('/')}/history/{prompt_id}")
        candidate = result.get(prompt_id) if isinstance(result, dict) else None
        if isinstance(candidate, dict):
            history = candidate
            break
        time.sleep(1.0)
    if history is None:
        raise TimeoutError(f"ComfyUI prompt {prompt_id} exceeded {args.timeout:.1f}s")

    elapsed = time.monotonic() - started
    status = history.get("status", {})
    completed = status.get("completed") if isinstance(status, dict) else None
    result = {
        "prompt_id": prompt_id,
        "seed": args.seed,
        "requested_width": args.width,
        "requested_height": args.height,
        "frames": args.frames,
        "fps": args.fps,
        "elapsed_seconds": elapsed,
        "completed": completed,
        "output_files": output_files(history),
        "status": status,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if completed is not True:
        raise RuntimeError(f"ComfyUI prompt did not complete successfully: {status!r}")


if __name__ == "__main__":
    main()
