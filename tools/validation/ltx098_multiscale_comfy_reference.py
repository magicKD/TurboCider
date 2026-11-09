#!/usr/bin/env python3
"""Owned ComfyUI multiscale 0.9.8 I2V reference, strength=1 only.

At strength 1 the stock Comfy inpaint mask and original first-frame freeze
have the same conditioning behavior. Diagnostic custom nodes advance the CPU
RNG to match native's second-pass noise and apply official post-upscale AdaIN.
The checkpoint, transformer, VAE and learned upsampler use stock Comfy nodes.
The two video encoders and decoder noise streams are not matched.
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

from ltx098_comfy_reference import PROMPT, MODEL_NAME, TEXT_NAME, sha256
from z_comfy_framework_screen import launch_command
from mlxserve_local_model_probe import listener_pid


def workflow(prompt, seed, frames, keep_upscaled=False, trace_steps=False):
    def n(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}
    graph = {
        "1": n("CheckpointLoaderSimple", ckpt_name=MODEL_NAME),
        "2": n("CLIPLoader", clip_name=TEXT_NAME, type="ltxv", device="default"),
        "3": n("CLIPTextEncode", clip=["2", 0], text=prompt),
        "4": n("ConditioningZeroOut", conditioning=["3", 0]),
        "5": n("LTXVConditioning", positive=["3", 0], negative=["4", 0], frame_rate=24.0),
        "6": n("LoadImage", image="input.png"),
        "7": n("LTXVImgToVideo", positive=["5", 0], negative=["5", 1], vae=["1", 2], image=["6", 0],
               width=320, height=320, length=frames, batch_size=1, strength=1.0),
        "8": n("RandomNoise", noise_seed=seed),
        "9": n("BasicGuider", model=["1", 0], conditioning=["7", 0]),
        "10": n("KSamplerSelect", sampler_name="euler"),
        "11": n("ManualSigmas", sigmas="1, .9937, .9875, .9812, .975, .9094, .725, 0"),
        "12": n("SamplerCustomAdvanced", noise=["8", 0], guider=["9", 0], sampler=["10", 0], sigmas=["11", 0], latent_image=["7", 2]),
        "13": n("LatentUpscaleModelLoader", model_name="ltxv-spatial-upscaler-0.9.8.safetensors"),
        "14": n("LTXVLatentUpsampler", samples=["12", 0], upscale_model=["13", 0], vae=["1", 2]),
        "15": n("LTXVImgToVideoInplace", vae=["1", 2], image=["6", 0], latent=["24", 0], strength=1.0, bypass=False),
        "16": n("LTX098SecondNoise", seed=seed, first_height=10, first_width=10),
        "17": n("ManualSigmas", sigmas=".9094, .725, .4219, 0"),
        "18": n("SamplerCustomAdvanced", noise=["16", 0], guider=["9", 0], sampler=["10", 0], sigmas=["17", 0], latent_image=["15", 0]),
        "19": n("VAEDecode", samples=["18", 0], vae=["1", 2]),
        "20": n("ImageScale", image=["19", 0], upscale_method="bilinear", width=512, height=512, crop="disabled"),
        "21": n("CreateVideo", images=["20", 0], fps=24.0, bit_depth=8),
        "22": n("SaveVideo", video=["21", 0], filename_prefix="ltx098-multiscale",
                **{"format": "mp4", "format.codec": "h264", "format.codec.encoding": "auto"}),
        "23": n("SaveImage", images=["20", 0], filename_prefix="diagnostic/frame"),
        "24": n("LTX098AdaIN", upsampled=["14", 0], reference=["12", 0]),
        "25": n("SaveLatent", samples=["7", 2], filename_prefix="diagnostic/first_condition"),
        "26": n("SaveLatent", samples=["12", 0], filename_prefix="diagnostic/first_pass"),
        "27": n("SaveLatent", samples=["14", 0], filename_prefix="diagnostic/upsampled"),
        "28": n("SaveLatent", samples=["24", 0], filename_prefix="diagnostic/adain"),
        "29": n("SaveLatent", samples=["18", 0], filename_prefix="diagnostic/final"),
    }
    if keep_upscaled:
        graph["21"]["inputs"]["images"] = ["19", 0]
        graph["23"]["inputs"]["images"] = ["19", 0]
        del graph["20"]
    if trace_steps:
        graph["10"] = n("LTX098TraceSampler", stage="first")
        graph["30"] = n("LTX098TraceSampler", stage="second")
        graph["18"]["inputs"]["sampler"] = ["30", 0]
    return graph


async def run(server, port, graph, output):
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
        if not any(item.get("type") == "mps" for item in system.get("devices", [])):
            raise RuntimeError("ComfyUI did not report MPS")
        async with session.get(address + "/object_info") as response:
            objects = await response.json()
        missing = {item["class_type"] for item in graph.values()} - objects.keys()
        if missing:
            raise RuntimeError(f"missing ComfyUI nodes: {sorted(missing)}")
        client = uuid.uuid4().hex
        events = []
        async with session.ws_connect(address + "/ws?clientId=" + client) as ws:
            start = time.perf_counter()
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
                    events.append(event)
                    if event["type"] in ("execution_error", "execution_interrupted"):
                        raise RuntimeError(json.dumps(event))
                    if event["type"] == "execution_success":
                        break
                else:
                    raise RuntimeError("ComfyUI websocket closed before success")
        async with session.get(address + "/history/" + prompt_id) as response:
            history = (await response.json())[prompt_id]
        if history["status"]["status_str"] != "success":
            raise RuntimeError("ComfyUI reference history is not successful")
        locators = [item for group in history["outputs"].values() for value in group.values()
                    if isinstance(value, list) for item in value
                    if isinstance(item, dict) and str(item.get("filename", "")).endswith(".mp4")]
        if len(locators) != 1:
            raise RuntimeError("expected one ComfyUI video")
        locator = {key: locators[0][key] for key in ("filename", "subfolder", "type")}
        async with session.get(address + "/view", params=locator) as response:
            response.raise_for_status()
            (output / "reference.mp4").write_bytes(await response.read())
        progress = {node: [event["data"]["value"] for event in events
                           if event["type"] == "progress" and event["data"].get("node") == node]
                    for node in ("12", "18")}
        if progress != {"12": list(range(1, 8)), "18": list(range(1, 4))}:
            raise RuntimeError(f"unexpected sampler progress: {progress}")
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        (output / "events.json").write_text(json.dumps(events, indent=2) + "\n")
        return {"seconds_including_diagnostic_pngs": time.perf_counter() - start, "progress": progress,
                "video_sha256": sha256(output / "reference.mp4")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--extra", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--keep-upscaled", action="store_true", help="retain 640x640 instead of final resizing")
    parser.add_argument("--trace-steps", action="store_true", help="save every Euler boundary; diagnostic only")
    args = parser.parse_args()
    import yaml
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory must be new")
    model, comfy, extra, source = (p.resolve(strict=True) for p in (args.model, args.comfy, args.extra, args.input))
    if args.frames < 9 or args.frames % 8 != 1:
        parser.error("frames must be 8n+1 and >=9")
    output.mkdir(parents=True)
    instance = output / "instance"
    (instance / "input").mkdir(parents=True)
    (instance / "user").mkdir()
    shutil.copyfile(source, instance / "input/input.png")
    custom = instance / "custom_nodes" / "ltx098_second_noise"
    custom.mkdir(parents=True)
    shutil.copyfile(Path(__file__).with_name("ltx098_comfy_sequential_noise.py"), custom / "__init__.py")
    (output / "models.yaml").write_text(yaml.safe_dump({"ltx098": {"base_path": str(model), "checkpoints": ".",
                                             "text_encoders": "text_encoders", "latent_upscale_models": "."}}))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = launch_command(comfy, args.python.absolute(), output, "comfy-sdpa-gpu", port)
    command.remove("--disable-all-custom-nodes")
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(extra), PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    environment.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    graph = workflow(args.prompt, args.seed, args.frames, args.keep_upscaled, args.trace_steps)
    (output / "workflow.json").write_text(json.dumps(graph, indent=2) + "\n")
    report = {"schema": "ltx098-multiscale-comfy-v1", "status": "running", "seed": args.seed,
              "scope": f"ComfyUI MPS, {'320->640' if args.keep_upscaled else '320->640->512'}, 7+AdaIN+3, strength 1, CFG 1; diagnostic PNG export included",
              "comfy_commit": subprocess.check_output(["git", "-C", str(comfy), "rev-parse", "HEAD"], text=True).strip(),
              "input_sha256": sha256(source), "upscaler_sha256": sha256(model / "ltxv-spatial-upscaler-0.9.8.safetensors")}
    try:
        with (output / "server.stdout.txt").open("x") as stdout, (output / "server.stderr.txt").open("x") as stderr:
            server = subprocess.Popen(command, cwd=comfy, env=environment, stdout=stdout, stderr=stderr)
            try:
                report["result"] = asyncio.run(run(server, port, graph, output))
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
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
