#!/usr/bin/env python3
"""Run an owned stock-ComfyUI MPS reference for FastWan2.2 TI2V-5B.

The model, first-frame conditioner, text encoder and VAE are stock ComfyUI.
A small local custom sampler implements the FastVideo DMD re-noising schedule.
The script never connects to an existing user server. Timings include request
execution and MP4 export, but exclude server startup and checkpoint hashing.
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
from wan22_schedule import warped_sigmas
from z_comfy_framework_screen import launch_command


MODEL = "fastwan2.2_ti2v_5b_fullattn.comfy.safetensors"
TEXT = "umt5_xxl_fp16.safetensors"
VAE = "wan2.2_vae.safetensors"
PROMPT = ("A red fox walks slowly toward the camera through a snowy field. "
          "Its fur moves gently in the wind and its paws step naturally in the snow. "
          "Keep the camera steady and the fox's face and anatomy consistent.")


def workflow(prompt, seed, width, height, frames, fps, diagnostic=False, capture_block=False,
             capture_vae=False, capture_transformer=False, capture_text=False):
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}
    sigmas = warped_sigmas()
    graph = {
        "1": node("UNETLoader", unet_name=MODEL, weight_dtype="default"),
        "2": node("CLIPLoader", clip_name=TEXT, type="wan", device="default"),
        "3": node("CLIPTextEncode", clip=["2", 0], text=prompt),
        "4": node("VAELoader", vae_name=VAE),
        "5": node("LoadImage", image="input.png"),
        "6": node("Wan22ImageToVideoLatent", vae=["4", 0], start_image=["5", 0],
                  width=width, height=height, length=frames, batch_size=1),
        "7": node("RandomNoise", noise_seed=seed),
        "8": node("BasicGuider", model=["1", 0], conditioning=["3", 0]),
        "9": node("FastWan22DMDSampler"),
        "10": node("ManualSigmas", sigmas=", ".join(map(str, [*sigmas, 0.0]))),
        "11": node("SamplerCustomAdvanced", noise=["7", 0], guider=["8", 0], sampler=["9", 0],
                   sigmas=["10", 0], latent_image=["6", 0]),
        "12": node("VAEDecode", samples=["11", 0], vae=["4", 0]),
        "13": node("CreateVideo", images=["12", 0], fps=float(fps), bit_depth=8),
        "14": node("SaveVideo", video=["13", 0], filename_prefix="fastwan22-reference",
                   **{"format": "mp4", "format.codec": "h264", "format.codec.encoding": "auto"}),
    }
    if diagnostic:
        graph["15"] = node("SaveLatent", samples=["6", 0], filename_prefix="diagnostic/condition")
        graph["16"] = node("SaveLatent", samples=["11", 0], filename_prefix="diagnostic/final")
        graph["17"] = node("SaveImage", images=["12", 0], filename_prefix="diagnostic/frame")
        graph["18"] = node("SaveConditioning", conditioning=["3", 0], filename_prefix="diagnostic/text")
    if capture_block:
        graph["9"]["inputs"]["capture_first_block"] = True
    if capture_transformer:
        graph["9"]["inputs"]["capture_transformer"] = True
    if capture_text:
        graph["3"]["class_type"] = "FastWan22CaptureTextEncode"
    if capture_vae:
        graph["19"] = node("FastWan22CaptureVAE", vae=["4", 0])
        graph["6"]["inputs"]["vae"] = ["19", 0]
    return graph


async def run_request(server, port, graph, output):
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
                    raise RuntimeError("ComfyUI websocket closed before success")
        async with session.get(address + "/history/" + prompt_id) as response:
            response.raise_for_status()
            history = (await response.json())[prompt_id]
        if history["status"]["status_str"] != "success":
            raise RuntimeError("ComfyUI history is not successful")
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
        evidence_path = output / "instance/output/diagnostic/fastwan22-dmd-steps.json"
        evidence = json.loads(evidence_path.read_text())
        if [step["index"] for step in evidence["steps"]] != [0, 1, 2] or evidence["seed"] != graph["7"]["inputs"]["noise_seed"]:
            raise RuntimeError(f"reference did not execute three sampler steps: {evidence}")
        if not progress or progress[-1] != 3 or any(value not in (1, 2, 3) for value in progress):
            raise RuntimeError(f"reference final sampler progress is missing: {progress}")
        executed = [event["data"].get("node") for event in events if event["type"] == "executing"]
        if any(node not in executed for node in ("11", "12", "14")):
            raise RuntimeError("reference sampler, decoder and exporter must all execute")
        (output / "events.json").write_text(json.dumps(events, indent=2) + "\n")
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,nb_frames,r_frame_rate,duration", "-of", "json",
            str(output / "reference.mp4")], text=True))["streams"][0]
        return {"request_wall_seconds": wall, "progress_events": progress, "sampler_evidence": evidence, "video": info,
                "video_sha256": sha256(output / "reference.mp4"), "system_stats": system}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--extra", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--prompt-job", type=Path,
                        help="read the prompt from the first entry in a TurboCider jobs.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frames", type=int, default=97)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--capture-block", action="store_true", help="capture one real first-block forward for parity")
    parser.add_argument("--capture-vae", action="store_true", help="capture the real first-frame VAE encoding for parity")
    parser.add_argument("--capture-transformer", action="store_true", help="capture the first full transformer call for parity")
    parser.add_argument("--capture-text", action="store_true", help="capture UMT5 IDs and output for parity")
    args = parser.parse_args()
    import yaml
    if args.prompt_job:
        jobs = json.loads(args.prompt_job.read_text())
        if not isinstance(jobs, list) or not jobs or not isinstance(jobs[0].get("request"), dict):
            parser.error("prompt-job must contain a TurboCider job request")
        args.prompt = jobs[0]["request"].get("prompt", "")
    if args.output.exists():
        parser.error("output directory must be new")
    if not args.prompt or args.seed < 0 or args.seed >= 2**64 or args.width % 32 or args.height % 32 or args.frames < 5 or args.frames % 4 != 1 or args.fps <= 0:
        parser.error("invalid prompt, seed, dimensions, frames or fps")
    model, comfy, extra, source = (path.resolve(strict=True) for path in (args.model, args.comfy, args.extra, args.input))
    components = {"transformer": model / "diffusion_models" / MODEL,
                  "text_encoder": model / "text_encoders" / TEXT, "vae": model / "vae" / VAE}
    identities = {name: sha256(path.resolve(strict=True)) for name, path in components.items()}
    output = args.output.absolute()
    output.mkdir(parents=True)
    instance = output / "instance"
    (instance / "input").mkdir(parents=True)
    (instance / "user").mkdir()
    shutil.copyfile(source, instance / "input/input.png")
    custom = instance / "custom_nodes" / "fastwan22_dmd"
    custom.mkdir(parents=True)
    shutil.copyfile(Path(__file__).with_name("wan22_comfy_dmd_node.py"), custom / "__init__.py")
    (output / "models.yaml").write_text(yaml.safe_dump({"wan22": {"base_path": str(model),
        "diffusion_models": "diffusion_models", "text_encoders": "text_encoders", "vae": "vae"}}))
    graph = workflow(args.prompt, args.seed, args.width, args.height, args.frames, args.fps,
                     args.diagnostic, args.capture_block, args.capture_vae, args.capture_transformer, args.capture_text)
    (output / "workflow.json").write_text(json.dumps(graph, indent=2) + "\n")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = launch_command(comfy, args.python.absolute(), output, "comfy-sdpa-gpu", port)
    command.remove("--disable-all-custom-nodes")
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(extra), PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    environment.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    report = {"schema": "fastwan22-comfy-reference-v1", "status": "running",
              "scope": "isolated ComfyUI MPS, full VAE, GPU-only; request wall includes MP4 export; server startup excluded",
              "diagnostic_extra_outputs": args.diagnostic, "capture_first_block": args.capture_block,
              "capture_first_frame_vae": args.capture_vae,
              "capture_first_transformer": args.capture_transformer,
              "capture_text": args.capture_text,
              "source_model": "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
              "comfy_commit": subprocess.check_output(["git", "-C", str(comfy), "rev-parse", "HEAD"], text=True).strip(),
              "component_sha256": identities, "input_sha256": sha256(source), "prompt": args.prompt,
              "seed": args.seed, "width": args.width, "height": args.height, "frames": args.frames,
              "fps": args.fps, "sigmas": warped_sigmas(), "scheduler": "DMD-renoise-shift5-warped"}
    try:
        with (output / "server.stdout.txt").open("x") as stdout, (output / "server.stderr.txt").open("x") as stderr:
            server = subprocess.Popen(command, cwd=comfy, env=environment, stdout=stdout, stderr=stderr)
            try:
                result = asyncio.run(run_request(server, port, graph, output))
                video = result["video"]
                if (video["width"], video["height"], int(video["nb_frames"]), video["r_frame_rate"]) != (args.width, args.height, args.frames, f"{args.fps}/1"):
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
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(output),
                      "seconds": report["result"]["request_wall_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
