#!/usr/bin/env python3
"""Run an owned stock-ComfyUI HunyuanVideo 1.5 480p I2V reference.

This is the full step-distilled FP16 checkpoint, not the separately released
CFG-distilled model. The official guide recommends CFG 1, shift 7 and 8 or
12 steps. Timings include the request and MP4 retrieval, but not startup or
checkpoint hashing. No existing user ComfyUI server is contacted.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import uuid

from ltx098_comfy_reference import sha256
from mlxserve_local_model_probe import listener_pid
from wan22_comfy_reference import PROMPT
from z_comfy_framework_screen import launch_command


COMPONENTS = {
    "transformer": ("diffusion_models", "hunyuanvideo1.5_480p_i2v_step_distilled_fp16.safetensors",
                    "ebe7c60253c8369e02a0e9a8adb49f11f3003dff01954c250b7650330551edc3"),
    "qwen_text": ("text_encoders", "qwen_2.5_vl_7b.safetensors",
                  "cfafd739459bc86257397259f612a9aee88e5b98e85b5c0d0d1717e898b3463a"),
    "glyph_text": ("text_encoders", "byt5_small_glyphxl_fp16.safetensors",
                   "516910bb4c9b225370290e40585d1b0e6c8cd3583690f7eec2f7fb593990fb48"),
    "vision": ("clip_vision", "sigclip_vision_patch14_384.safetensors",
               "1fee501deabac72f0ed17610307d7131e3e9d1e838d0363aa3c2b97a6e03fb33"),
    "vae": ("vae", "hunyuanvideo15_vae_fp16.safetensors",
            "e7c3091949c27e2d55ae6d5df917b99dadfebbf308e5a50d0ade0d16c90297ae"),
}


def workflow(prompt, seed, width, height, frames, fps, steps, save_latent=False, tiled_vae=False):
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}
    graph = {
        "1": node("UNETLoader", unet_name=COMPONENTS["transformer"][1], weight_dtype="default"),
        "2": node("DualCLIPLoader", clip_name1=COMPONENTS["qwen_text"][1],
                  clip_name2=COMPONENTS["glyph_text"][1], type="hunyuan_video_15", device="default"),
        "3": node("CLIPTextEncode", clip=["2", 0], text=prompt),
        "4": node("CLIPTextEncode", clip=["2", 0], text=""),
        "5": node("VAELoader", vae_name=COMPONENTS["vae"][1]),
        "6": node("LoadImage", image="input.png"),
        "7": node("CLIPVisionLoader", clip_name=COMPONENTS["vision"][1]),
        "8": node("CLIPVisionEncode", clip_vision=["7", 0], image=["6", 0], crop="center"),
        "9": node("HunyuanVideo15ImageToVideo", positive=["3", 0], negative=["4", 0], vae=["5", 0],
                  start_image=["6", 0], clip_vision_output=["8", 0], width=width, height=height,
                  length=frames, batch_size=1),
        "10": node("ModelSamplingSD3", model=["1", 0], shift=7.0),
        "11": node("KSampler", model=["10", 0], positive=["9", 0], negative=["9", 1],
                   latent_image=["9", 2], seed=seed, steps=steps, cfg=1.0, sampler_name="euler",
                   scheduler="simple", denoise=1.0),
        "12": node("VAEDecode", samples=["11", 0], vae=["5", 0]),
        "13": node("CreateVideo", images=["12", 0], fps=float(fps), bit_depth=8),
        "14": node("SaveVideo", video=["13", 0], filename_prefix="hunyuan15-reference",
                   **{"format": "mp4", "format.codec": "h264", "format.codec.encoding": "auto"}),
    }
    if save_latent:
        graph["15"] = node("SaveLatent", samples=["11", 0], filename_prefix="diagnostic/hunyuan15-final")
        graph["12"]["inputs"]["samples"] = ["15", 0]
    if tiled_vae:
        graph["12"] = node("VAEDecodeTiled", samples=["15", 0] if save_latent else ["11", 0], vae=["5", 0],
                           tile_size=256, overlap=64, temporal_size=4096, temporal_overlap=8)
    return graph


async def run_request(server, port, graph, output, steps, required_nodes=("9", "11", "12", "14")):
    import aiohttp
    address = f"http://127.0.0.1:{port}"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600), trust_env=False) as session:
        started = time.perf_counter()
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"owned ComfyUI exited before readiness: {server.returncode}")
            if listener_pid(port) == {server.pid}:
                try:
                    async with session.get(address + "/system_stats") as response:
                        response.raise_for_status()
                        system = await response.json()
                    break
                except (aiohttp.ClientConnectionError, TimeoutError):
                    pass
            if time.perf_counter() - started > 180:
                raise TimeoutError("owned ComfyUI did not become ready")
            await asyncio.sleep(.2)
        if not any(device.get("type") == "mps" for device in system.get("devices", [])):
            raise RuntimeError("reference ComfyUI did not report MPS")
        async with session.get(address + "/object_info") as response:
            response.raise_for_status()
            objects = await response.json()
        missing = {value["class_type"] for value in graph.values()} - objects.keys()
        if missing:
            raise RuntimeError(f"required ComfyUI nodes missing: {sorted(missing)}")
        client = uuid.uuid4().hex
        events = []
        async with session.ws_connect(address + "/ws?clientId=" + client) as ws:
            request_start = time.perf_counter()
            async with session.post(address + "/prompt", json={"prompt": graph, "client_id": client}) as response:
                payload = await response.json()
                if response.status != 200:
                    raise RuntimeError(json.dumps(payload))
            prompt_id = payload["prompt_id"]
            async with asyncio.timeout(3600):
                async for message in ws:
                    if message.type != aiohttp.WSMsgType.TEXT:
                        continue
                    event = json.loads(message.data)
                    if event.get("data", {}).get("prompt_id") != prompt_id:
                        continue
                    events.append({"elapsed": time.perf_counter() - request_start, **event})
                    if event["type"] in ("execution_error", "execution_interrupted"):
                        (output / "events.json").write_text(json.dumps(events, indent=2) + "\n")
                        raise RuntimeError(json.dumps(event))
                    if event["type"] == "execution_success":
                        break
                else:
                    (output / "events.json").write_text(json.dumps(events, indent=2) + "\n")
                    raise RuntimeError("ComfyUI websocket closed before success")
        async with session.get(address + "/history/" + prompt_id) as response:
            response.raise_for_status()
            history = (await response.json())[prompt_id]
        if history["status"]["status_str"] != "success":
            raise RuntimeError("reference history is not successful")
        files = [item for group in history["outputs"].values() for entries in group.values()
                 if isinstance(entries, list) for item in entries
                 if isinstance(item, dict) and str(item.get("filename", "")).endswith(".mp4")]
        if len(files) != 1:
            raise RuntimeError("reference must produce exactly one MP4")
        locator = {key: files[0][key] for key in ("filename", "subfolder", "type")}
        async with session.get(address + "/view", params=locator) as response:
            response.raise_for_status()
            (output / "reference.mp4").write_bytes(await response.read())
        wall = time.perf_counter() - request_start
        progress = [event["data"]["value"] for event in events
                    if event["type"] == "progress" and event["data"].get("node") == "11"]
        if steps is not None and (not progress or progress[-1] != steps or any(value < 1 or value > steps for value in progress)):
            raise RuntimeError(f"reference final sampler progress is missing: {progress}")
        executed = [event["data"].get("node") for event in events if event["type"] == "executing"]
        if any(node not in executed for node in required_nodes):
            raise RuntimeError("conditioning, sampler, decoder and exporter must execute")
        (output / "events.json").write_text(json.dumps(events, indent=2) + "\n")
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames,r_frame_rate,duration", "-of", "json",
            str(output / "reference.mp4")], text=True))["streams"][0]
        return {"request_wall_seconds": wall, "progress_events": progress, "video": info,
                "video_sha256": sha256(output / "reference.mp4"), "system_stats": system}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("python", "comfy", "extra", "model", "input", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frames", type=int, default=97)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=8, choices=(8, 12))
    parser.add_argument("--attention", choices=("pytorch", "subquad"), default="pytorch",
                        help="stock ComfyUI attention implementation")
    parser.add_argument("--save-latent", action="store_true", help="persist final latent before decoding")
    parser.add_argument("--tiled-vae", action="store_true", help="spatial tiled VAE with temporal_size 4096")
    args = parser.parse_args()
    import yaml
    if args.output.exists():
        parser.error("output directory must be new")
    if not args.prompt or not 0 <= args.seed < 2**64 or args.width % 32 or args.height % 32 or \
            args.frames < 5 or args.frames % 4 != 1 or args.fps <= 0:
        parser.error("invalid prompt, seed, dimensions, frames or fps")
    model, comfy, extra, source = (path.resolve(strict=True) for path in (args.model, args.comfy, args.extra, args.input))
    identities = {}
    for name, (directory, filename, expected) in COMPONENTS.items():
        actual = sha256((model / directory / filename).resolve(strict=True))
        if actual != expected:
            raise ValueError(f"{name} checksum mismatch: {actual}")
        identities[name] = actual
    output = args.output.absolute()
    output.mkdir(parents=True)
    instance = output / "instance"
    (instance / "input").mkdir(parents=True)
    (instance / "user").mkdir()
    shutil.copyfile(source, instance / "input/input.png")
    (output / "models.yaml").write_text(yaml.safe_dump({"hunyuan15": {"base_path": str(model),
        **{directory: directory for directory in ("diffusion_models", "text_encoders", "clip_vision", "vae")}}}))
    graph = workflow(args.prompt, args.seed, args.width, args.height, args.frames, args.fps, args.steps,
                     args.save_latent, args.tiled_vae)
    (output / "workflow.json").write_text(json.dumps(graph, indent=2) + "\n")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = launch_command(comfy, args.python.absolute(), output, "comfy-sdpa-gpu", port)
    if args.attention == "subquad":
        command.remove("--use-pytorch-cross-attention")
        command.append("--use-quad-cross-attention")
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(extra), PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    environment.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    report = {"schema": "hunyuan15-comfy-reference-v1", "status": "running",
              "scope": "isolated ComfyUI MPS, stock workflow, GPU-only; request wall includes MP4 export; startup excluded",
              "source_model": "Comfy-Org/Hunyuan-Video-1.5 480p I2V step-distilled FP16",
              "comfy_commit": subprocess.check_output(["git", "-C", str(comfy), "rev-parse", "HEAD"], text=True).strip(),
              "component_sha256": identities, "input_sha256": sha256(source), "prompt": args.prompt,
              "seed": args.seed, "width": args.width, "height": args.height, "frames": args.frames,
              "fps": args.fps, "steps": args.steps, "cfg": 1.0, "flow_shift": 7.0,
              "sampler": "euler", "scheduler": "simple", "attention": args.attention,
              "save_final_latent": args.save_latent, "tiled_vae": args.tiled_vae,
              "vae_tile_size": 256 if args.tiled_vae else None,
              "vae_temporal_size": 4096 if args.tiled_vae else None}
    try:
        with (output / "server.stdout.txt").open("x") as stdout, (output / "server.stderr.txt").open("x") as stderr:
            server = subprocess.Popen(command, cwd=comfy, env=environment, stdout=stdout, stderr=stderr)
            try:
                result = asyncio.run(run_request(server, port, graph, output, args.steps))
                video = result["video"]
                if (video["width"], video["height"], int(video["nb_frames"]), video["r_frame_rate"]) != \
                        (args.width, args.height, args.frames, f"{args.fps}/1"):
                    raise ValueError("reference video geometry does not match request")
                report["result"] = result
                report["status"] = "complete"
            finally:
                if server.poll() is None:
                    server.terminate()
                    try:
                        server.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=10)
                report["server_terminal"] = server.poll() is not None
                report["server_returncode"] = server.returncode
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "seconds": report["result"]["request_wall_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
