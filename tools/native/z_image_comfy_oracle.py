#!/usr/bin/env python3
"""Run the official embedded ComfyUI Z-Image workflow as a local oracle."""

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image


def request_json(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def embedded_prompt(path: Path) -> dict:
    with Image.open(path) as image:
        prompt = image.info.get("prompt")
    if not prompt:
        raise SystemExit(f"workflow PNG has no embedded prompt: {path}")
    return json.loads(prompt)


def node_item(prompt: dict, class_type: str, title_contains: str = "") -> tuple[str, dict]:
    matches = [
        (key, value)
        for key, value in prompt.items()
        if value.get("class_type") == class_type
        and title_contains.lower() in value.get("_meta", {}).get("title", "").lower()
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"expected one {class_type!r} node containing {title_contains!r}, got {len(matches)}"
        )
    return matches[0]


def download(server: str, locator: dict, output: Path) -> None:
    query = urllib.parse.urlencode(
        {
            "filename": locator["filename"],
            "subfolder": locator.get("subfolder", ""),
            "type": locator.get("type", "output"),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{server}/view?{query}", timeout=30) as response:
        output.write_bytes(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8190")
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--prefix", default="TurboCider-z-image-oracle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latent-output", type=Path)
    parser.add_argument(
        "--lora-name",
        help="ComfyUI lora filename to insert after the diffusion-model loader",
    )
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    prompt = embedded_prompt(args.workflow)
    sampler_id, sampler = node_item(prompt, "KSampler")
    _, positive = node_item(prompt, "CLIPTextEncode", "positive")
    _, saver = node_item(prompt, "SaveImage")
    if args.lora_name:
        loader_id, loader = node_item(prompt, "UNETLoader")
        lora_id = str(max(map(int, prompt)) + 1)
        prompt[lora_id] = {
            "inputs": {
                "model": [loader_id, 0],
                "lora_name": args.lora_name,
                "strength_model": args.lora_strength,
            },
            "class_type": "LoraLoaderModelOnly",
            "_meta": {"title": "TurboCider LoRA oracle"},
        }
        sampler["inputs"]["model"] = [lora_id, 0]
    sampler["inputs"]["seed"] = args.seed
    positive["inputs"]["text"] = args.prompt
    saver["inputs"]["filename_prefix"] = args.prefix
    if args.latent_output:
        latent_id = str(max(map(int, prompt)) + 1)
        prompt[latent_id] = {
            "inputs": {
                "samples": [sampler_id, 0],
                "filename_prefix": args.prefix + "-latent",
            },
            "class_type": "SaveLatent",
            "_meta": {"title": "Save Latent"},
        }

    submitted = request_json(f"{args.server}/prompt", {"prompt": prompt})
    prompt_id = submitted["prompt_id"]
    deadline = time.monotonic() + args.timeout
    history = {}
    while time.monotonic() < deadline:
        response = request_json(f"{args.server}/history/{prompt_id}")
        if prompt_id in response:
            history = response[prompt_id]
            break
        time.sleep(1)
    if not history:
        raise SystemExit(f"timed out waiting for ComfyUI prompt {prompt_id}")
    status = history.get("status", {})
    if status.get("status_str") != "success":
        raise SystemExit(json.dumps(status, indent=2))

    images = []
    latents = []
    for output in history.get("outputs", {}).values():
        images.extend(output.get("images", []))
        latents.extend(output.get("latents", []))
    if len(images) != 1:
        raise SystemExit(f"expected one output image, got {len(images)}")
    image = images[0]
    download(args.server, image, args.output)
    if args.latent_output:
        if len(latents) != 1:
            raise SystemExit(f"expected one output latent, got {len(latents)}")
        download(args.server, latents[0], args.latent_output)
    print(
        json.dumps(
            {
                "prompt_id": prompt_id,
                "output": str(args.output),
                "latent_output": str(args.latent_output) if args.latent_output else None,
                "lora_name": args.lora_name,
                "lora_strength": args.lora_strength if args.lora_name else None,
                "comfy_output": image,
                "status": status,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
