#!/usr/bin/env python3
"""Run an isolated, stock-ComfyUI GPU reference for LTX-Video 0.9.8 I2V.

This is a reference measurement, not evidence that TurboCider implements the
model or is faster. It never connects to an existing user's ComfyUI instance.
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import uuid

from z_comfy_framework_screen import launch_command
from mlxserve_local_model_probe import listener_pid

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "ltxv-2b-0.9.8-distilled.safetensors"
TEXT_NAME = "t5xxl_fp16.safetensors"
MODEL_SHA256 = "76aa8c4786af752fa6f951947129d5290c3c6c0b2fadcadea6b5e114ae2cad8f"
TEXT_SHA256 = "6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635"
SIGMAS = "1.0, 0.9937, 0.9875, 0.9812, 0.975, 0.9094, 0.725, 0.4219, 0.0"
PROMPT = ("A lighthouse stands on dark coastal rocks at sunset. Ocean waves roll "
          "and break gently around the rocks, the light in the lantern glows, "
          "and clouds drift slowly across the warm evening sky. The camera remains steady.")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def workflow(index, decoder="tiled", diagnostic=False, trace_euler=False,
             prompt=PROMPT, seed=42, strength=1.0, trace_decode_noise=False, trace_first_ffn=False,
             trace_ffn_blocks=""):
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}
    if decoder not in ("tiled", "full"):
        raise ValueError("unknown ComfyUI VAE decoder")
    vae_decode = (node("VAEDecodeTiled", samples=["12", 0], vae=["1", 2], tile_size=512,
                       overlap=64, temporal_size=64, temporal_overlap=8)
                  if decoder == "tiled" else node("VAEDecode", samples=["12", 0], vae=["1", 2]))
    graph = {
        "1": node("CheckpointLoaderSimple", ckpt_name=MODEL_NAME),
        "2": node("CLIPLoader", clip_name=TEXT_NAME, type="ltxv", device="default"),
        "3": node("CLIPTextEncode", clip=["2", 0], text=prompt),
        "4": node("ConditioningZeroOut", conditioning=["3", 0]),
        "5": node("LTXVConditioning", positive=["3", 0], negative=["4", 0], frame_rate=24.0),
        "6": node("LoadImage", image="input.png"),
        "7": node("LTXVImgToVideo", positive=["5", 0], negative=["5", 1], vae=["1", 2],
                  image=["6", 0], width=512, height=512, length=121, batch_size=1, strength=strength),
        "8": node("RandomNoise", noise_seed=seed + index),
        "9": node("BasicGuider", model=["1", 0], conditioning=["7", 0]),
        "10": node("KSamplerSelect", sampler_name="euler"),
        "11": node("ManualSigmas", sigmas=SIGMAS),
        "12": node("SamplerCustomAdvanced", noise=["8", 0], guider=["9", 0], sampler=["10", 0],
                   sigmas=["11", 0], latent_image=["7", 2]),
        "13": vae_decode,
        "14": node("CreateVideo", images=["13", 0], fps=24.0, bit_depth=8),
        "15": node("SaveVideo", video=["14", 0], filename_prefix=f"ltx098-reference-{index}",
                   **{"format": "mp4", "format.codec": "h264", "format.codec.encoding": "auto"}),
    }
    if diagnostic:
        graph.update({
            "16": node("SaveLatent", samples=["7", 2], filename_prefix=f"diagnostic/{index}-condition"),
            "17": node("SaveLatent", samples=["12", 0], filename_prefix=f"diagnostic/{index}-final"),
            "18": node("SaveConditioning", conditioning=["3", 0], filename_prefix=f"diagnostic/{index}-text"),
            "19": node("SaveImage", images=["13", 0], filename_prefix=f"diagnostic/{index}-frame"),
        })
    if trace_euler:
        graph["10"] = node("LTX098EulerTrace", trace_name=f"trace-{index}")
        if trace_decode_noise:
            graph["10"]["inputs"]["capture_decode_noise"] = True
        if trace_first_ffn:
            graph["10"]["inputs"]["capture_first_ffn"] = True
        if trace_ffn_blocks:
            graph["10"]["inputs"]["capture_ffn_blocks"] = trace_ffn_blocks
    return graph


async def run_requests(server, output, port, count, decoder, diagnostic, trace_euler, prompt, seed, strength, trace_decode_noise, trace_first_ffn, trace_ffn_blocks):
    import aiohttp
    address = f"http://127.0.0.1:{port}"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600), trust_env=False) as session:
        started = time.perf_counter()
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"owned ComfyUI exited before readiness: {server.returncode}")
            listeners = listener_pid(port)
            if listeners and listeners != {server.pid}:
                raise RuntimeError("port is owned by a different process")
            try:
                if listeners != {server.pid}:
                    raise aiohttp.ClientConnectionError("owned listener not ready")
                async with session.get(address + "/system_stats") as response:
                    response.raise_for_status()
                    system = await response.json()
                break
            except (aiohttp.ClientConnectionError, TimeoutError):
                if time.perf_counter() - started > 180:
                    raise TimeoutError("owned ComfyUI did not become ready")
                await asyncio.sleep(0.2)
        if not any(device.get("type") == "mps" for device in system.get("devices", [])):
            raise RuntimeError("ComfyUI did not report an MPS GPU")
        report = {"status": "running", "system_stats": system,
                  "ready_seconds": time.perf_counter() - started, "runs": []}
        write_json(output / "report.json", report)
        async with session.get(address + "/object_info") as response:
            response.raise_for_status()
            objects = await response.json()
        missing = sorted({node["class_type"] for node in workflow(0, decoder, diagnostic, trace_euler, prompt, seed, strength, trace_decode_noise, trace_first_ffn, trace_ffn_blocks).values()} - objects.keys())
        if missing:
            raise RuntimeError(f"required stock ComfyUI nodes are unavailable: {missing}")
        client = uuid.uuid4().hex
        async with session.ws_connect(address + "/ws?clientId=" + client) as ws:
            for index in range(count):
                if server.poll() is not None or listener_pid(port) != {server.pid}:
                    raise RuntimeError("owned ComfyUI listener changed before request")
                graph = workflow(index, decoder, diagnostic, trace_euler, prompt, seed, strength, trace_decode_noise, trace_first_ffn, trace_ffn_blocks)
                write_json(output / f"request-{index}.json", graph)
                start = time.perf_counter()
                async with session.post(address + "/prompt", json={"prompt": graph, "client_id": client}) as response:
                    payload = await response.json()
                    if response.status != 200:
                        raise RuntimeError(json.dumps(payload))
                prompt_id, events = payload["prompt_id"], []
                async with asyncio.timeout(3600):
                    async for message in ws:
                        if message.type != aiohttp.WSMsgType.TEXT:
                            continue
                        event = json.loads(message.data)
                        if event.get("data", {}).get("prompt_id") != prompt_id:
                            continue
                        events.append({"seconds": time.perf_counter() - start, **event})
                        if event["type"] in ("execution_error", "execution_interrupted"):
                            write_json(output / f"events-{index}.json", events)
                            raise RuntimeError(json.dumps(event))
                        if event["type"] == "execution_success":
                            break
                    else:
                        raise RuntimeError("ComfyUI websocket closed before success")
                for _ in range(100):
                    async with session.get(address + "/history/" + prompt_id) as response:
                        history = (await response.json()).get(prompt_id)
                    if history:
                        break
                    await asyncio.sleep(0.01)
                if not history or history["status"]["status_str"] != "success":
                    raise RuntimeError("successful ComfyUI history is missing")
                locators = [item for value in history["outputs"].values() for items in value.values()
                            if isinstance(items, list) for item in items
                            if isinstance(item, dict) and str(item.get("filename", "")).endswith(".mp4")]
                if len(locators) != 1:
                    raise RuntimeError("expected exactly one returned MP4")
                locator = {key: locators[0][key] for key in ("filename", "subfolder", "type")}
                async with session.get(address + "/view", params=locator) as response:
                    response.raise_for_status()
                    media = await response.read()
                media_path = output / f"{index}.mp4"
                media_path.write_bytes(media)
                wall = time.perf_counter() - start
                write_json(output / f"history-{index}.json", history)
                write_json(output / f"events-{index}.json", events)
                executed = [event["data"]["node"] for event in events
                            if event["type"] == "executing" and event["data"].get("node")]
                cached = [node for event in events if event["type"] == "execution_cached"
                          for node in event["data"].get("nodes", [])]
                progress = [event["data"] for event in events if event["type"] == "progress"
                            and event["data"].get("node") == "12"]
                if [item["value"] for item in progress] != list(range(1, 9)) or any(item["max"] != 8 for item in progress):
                    raise RuntimeError("reference did not report all eight sampler steps")
                if any(node not in executed or node in cached for node in ("12", "13", "15")):
                    raise RuntimeError("reference sampler, decoder and output must all execute")
                if trace_euler:
                    trace = output / "instance/output/diagnostic" / f"trace-{index}"
                    if sorted(path.name for path in trace.glob("step_*.safetensors")) != [f"step_{i}.safetensors" for i in range(8)]:
                        raise RuntimeError("diagnostic sampler did not save exactly eight steps")
                    if trace_decode_noise and not (trace / "decode_noise.safetensors").is_file():
                        raise RuntimeError("diagnostic decoder noise is missing")
                    if trace_first_ffn and not (trace / "first_ffn_input.safetensors").is_file():
                        raise RuntimeError("diagnostic first FFN input is missing")
                    for block in (int(value) for value in trace_ffn_blocks.split(",") if value):
                        if not (trace / f"ffn_{block}_input.safetensors").is_file():
                            raise RuntimeError(f"diagnostic FFN input {block} is missing")
                media_info = json.loads(subprocess.check_output(
                    ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(media_path)], text=True))
                video = next(stream for stream in media_info["streams"] if stream["codec_type"] == "video")
                if (video["width"], video["height"], int(video["nb_frames"]), video["r_frame_rate"]) != (512, 512, 121, "24/1"):
                    raise RuntimeError("reference MP4 dimensions, frames or FPS do not match")
                report["runs"].append({"index": index, "seed": seed + index,
                    "prompt_id": prompt_id, "request_wall_including_export": wall,
                    "executed_nodes": executed, "cached_nodes": cached,
                    "sampler_steps": len(progress), "media_sha256": sha256(media_path),
                    "video_stream": video})
                write_json(output / "report.json", report)
    report["status"] = "complete"
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new, non-existing output directory")
    parser.add_argument("--input", type=Path, required=True, help="source image (copied into isolated instance)")
    parser.add_argument("--model", type=Path, default=ROOT / "models/LTX-Video-0.9.8")
    parser.add_argument("--comfy", type=Path, default=ROOT / ".deps/comfy-v0.39.0-20261012")
    parser.add_argument("--extra", type=Path, default=ROOT / ".deps/comfy-v0.39-runtime-extra-20261012")
    parser.add_argument("--python", type=Path, required=True, help="ComfyUI Python environment with torch and aiohttp")
    parser.add_argument("--count", type=int, choices=(1, 2), default=2)
    parser.add_argument("--decoder", choices=("tiled", "full"), default="tiled")
    parser.add_argument("--diagnostic", action="store_true", help="also save latents, conditioning and PNG frames; timing is not a baseline")
    parser.add_argument("--trace-euler", action="store_true", help="copy an isolated diagnostic sampler node; timing is not a baseline")
    parser.add_argument("--trace-decode-noise", action="store_true", help="observe VAE noise in the isolated trace; requires --trace-euler and full decode")
    parser.add_argument("--trace-first-ffn", action="store_true", help="capture the first real FFN activation in the isolated trace; requires --trace-euler")
    parser.add_argument("--trace-ffn-blocks", default="", help="comma-separated first-step FFN block indices to capture; requires --trace-euler")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    if args.trace_decode_noise and (not args.trace_euler or args.decoder != "full"):
        parser.error("decoder noise tracing requires --trace-euler --decoder full")
    if args.trace_first_ffn and not args.trace_euler:
        parser.error("FFN tracing requires --trace-euler")
    if args.trace_ffn_blocks:
        try:
            blocks = [int(value) for value in args.trace_ffn_blocks.split(",")]
        except ValueError:
            parser.error("FFN trace blocks must be comma-separated integers")
        if not args.trace_euler or args.trace_first_ffn or not blocks or len(set(blocks)) != len(blocks) or any(value < 0 or value >= 28 for value in blocks):
            parser.error("FFN trace blocks require --trace-euler, no --trace-first-ffn, and unique indices in [0,28)")
    if not args.prompt or args.seed < 0 or args.seed + args.count - 1 > 0xffffffffffffffff or not math.isfinite(args.strength) or not 0 <= args.strength <= 1:
        parser.error("nonempty prompt, uint64 seed sequence and strength in [0,1] are required")
    output, model, comfy, extra = args.output.resolve(), args.model.resolve(strict=True), args.comfy.resolve(strict=True), args.extra.resolve(strict=True)
    source_image = args.input.resolve(strict=True)
    python = args.python.absolute()
    if output.exists():
        parser.error("output must not already exist")
    # Fail before creating any output or server if the driver environment is
    # missing its own HTTP/YAML dependencies.
    import aiohttp  # noqa: F401
    import yaml
    for path, expected in ((model / MODEL_NAME, MODEL_SHA256), (model / "text_encoders" / TEXT_NAME, TEXT_SHA256)):
        if sha256(path) != expected:
            raise ValueError(f"weight hash mismatch: {path.name}")
    output.mkdir(parents=True)
    instance = output / "instance"
    (instance / "input").mkdir(parents=True)
    (instance / "user").mkdir()
    shutil.copyfile(source_image, instance / "input/input.png")
    (output / "models.yaml").write_text(yaml.safe_dump({"ltx098": {"base_path": str(model),
        "checkpoints": ".", "text_encoders": "text_encoders"}}))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = launch_command(comfy, python, output, "comfy-sdpa-gpu", port)
    if args.trace_euler:
        custom = instance / "custom_nodes" / "ltx098_euler_trace"
        custom.mkdir(parents=True)
        shutil.copyfile(Path(__file__).with_name("ltx098_comfy_euler_trace.py"), custom / "__init__.py")
        # The base directory is owned by this run and contains only this node.
        command.remove("--disable-all-custom-nodes")
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(extra), PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    environment.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    metadata = {"schema": "turbocider-ltx098-comfy-reference-v1", "status": "running",
        "comfy_commit": subprocess.check_output(["git", "-C", str(comfy), "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint_sha256": MODEL_SHA256, "text_encoder_sha256": TEXT_SHA256,
        "input_sha256": sha256(source_image), "command": command, "count": args.count,
        "scope": f"512x512,121 frames,24fps,8 Euler steps,CFG1,first-frame strength {args.strength}; {args.decoder} BF16 VAE; MPS GPU; request wall includes MP4 export",
        "prompt": args.prompt, "seed": args.seed, "first_frame_strength": args.strength,
        "diagnostic_extra_outputs": args.diagnostic, "diagnostic_euler_trace": args.trace_euler,
        "diagnostic_decode_noise": args.trace_decode_noise, "diagnostic_first_ffn": args.trace_first_ffn,
        "diagnostic_ffn_blocks": args.trace_ffn_blocks}
    write_json(output / "summary.json", metadata)
    with (output / "server.stdout.txt").open("x") as stdout, (output / "server.stderr.txt").open("x") as stderr:
        server = subprocess.Popen(command, cwd=comfy, env=environment, stdout=stdout, stderr=stderr)
        try:
            metadata["report"] = asyncio.run(run_requests(server, output, port, args.count, args.decoder, args.diagnostic,
                                                           args.trace_euler, args.prompt, args.seed, args.strength,
                                                           args.trace_decode_noise, args.trace_first_ffn, args.trace_ffn_blocks))
            metadata["status"] = "complete_reference_only"
        except BaseException as error:
            metadata.update(status="failed", error=str(error))
            raise
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=10)
            metadata["server_terminal"] = server.poll() is not None
            write_json(output / "summary.json", metadata)
    print(json.dumps({"status": metadata["status"], "output": str(output),
                      "seconds": [run["request_wall_including_export"] for run in metadata["report"]["runs"]]}))


if __name__ == "__main__":
    main()
