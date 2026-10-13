#!/usr/bin/env python3
"""Owned stock ComfyUI LTX2.5 full AV comparison; never reruns native."""
import argparse
import asyncio
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import uuid

from flux9_bf16_framework_screen import generation, check_generations, registered_hashes, write_json
from mlxserve_local_model_probe import listener_pid
from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_sampled
from z_comfy_framework_screen import launch_command

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/LTX-2.5"
FILES = dict(dit="diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
    encoder="text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
    video_vae="vae/ltx-2.5-video-vae-conv-bf16.safetensors", audio_vae="vae/ltx-2.5-audio-vae-bf16.safetensors",
    upscaler="latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")


def workflow(index):
    def node(kind, **inputs):
        return dict(class_type=kind, inputs=inputs)
    seed = 42 + index
    return {
        "1": node("UNETLoader", unet_name=Path(FILES["dit"]).name, weight_dtype="default"),
        "2": node("CLIPLoader", clip_name=Path(FILES["encoder"]).name, type="ltxv", device="default"),
        "3": node("VAELoader", vae_name=Path(FILES["video_vae"]).name),
        "4": node("VAELoader", vae_name=Path(FILES["audio_vae"]).name),
        "5": node("CLIPTextEncode", clip=["2", 0], text="A cinematic red fox running through a snowy forest"),
        "6": node("ConditioningZeroOut", conditioning=["5", 0]),
        "7": node("LTXVConditioning", positive=["5", 0], negative=["6", 0], frame_rate=24.),
        "8": node("SamplerCustomAdvanced", noise=["11", 0], guider=["12", 0], sampler=["13", 0], sigmas=["14", 0], latent_image=["17", 0]),
        "9": node("EmptyLTXVLatentVideo", width=352, height=224, length=97, batch_size=1),
        "10": node("LTXVEmptyLatentAudio", frames_number=97, frame_rate=24., batch_size=1, audio_vae=["4", 0]),
        "11": node("RandomNoise", noise_seed=seed),
        "12": node("LTXVDualCFGGuider", model=["1", 0], positive=["7", 0], negative=["7", 1], video_cfg=1., audio_cfg=1.),
        "13": node("KSamplerSelect", sampler_name="euler_ancestral"),
        "14": node("ManualSigmas", sigmas="1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"),
        "15": node("LatentUpscaleModelLoader", model_name=Path(FILES["upscaler"]).name),
        "16": node("LTXVSeparateAVLatent", av_latent=["8", 0]),
        "17": node("LTXVConcatAVLatent", video_latent=["9", 0], audio_latent=["10", 0]),
        "18": node("SamplerCustomAdvanced", noise=["19", 0], guider=["12", 0], sampler=["13", 0], sigmas=["20", 0], latent_image=["22", 0]),
        "19": node("RandomNoise", noise_seed=seed),
        "20": node("ManualSigmas", sigmas="0.909375, 0.725, 0.421875, 0.0"),
        "21": node("LTXVLatentUpsampler", samples=["16", 0], upscale_model=["15", 0], vae=["3", 0]),
        "22": node("LTXVConcatAVLatent", video_latent=["21", 0], audio_latent=["16", 1]),
        "23": node("LTXVSeparateAVLatent", av_latent=["18", 0]),
        "24": node("VAEDecodeTiled", samples=["23", 0], vae=["3", 0], tile_size=768, overlap=64, temporal_size=4096, temporal_overlap=32),
        "25": node("LTXVAudioVAEDecode", samples=["23", 1], audio_vae=["4", 0]),
        "26": node("CreateVideo", images=["24", 0], audio=["25", 0], fps=24., bit_depth=8),
        # DynamicCombo API inputs are flattened dotted keys, not nested dicts.
        "27": node("SaveVideo", video=["26", 0], filename_prefix=f"request-{index}",
                   **{"format": "mp4", "format.codec": "h264", "format.codec.encoding": "auto"}),
    }


def validate_events(events, index):
    for node_id, count in (("8", 8), ("18", 3)):
        progress = [e["data"] for e in events if e["type"] == "progress" and e["data"].get("node") == node_id]
        if [p["value"] for p in progress] != list(range(1, count + 1)) or any(p["max"] != count for p in progress):
            raise ValueError("actual complete8+3 LTX sampler steps required")
    executed = [e["data"]["node"] for e in events if e["type"] == "executing" and e["data"].get("node")]
    cached = [n for e in events if e["type"] == "execution_cached" for n in e["data"]["nodes"]]
    for required in ("8", "18", "21", "24", "25", "27"):
        if required not in executed or required in cached:
            raise ValueError("actual LTX sampling/upscale/AV decode/export required")
    if index == 0 and "5" not in executed:
        raise ValueError("cold LTX encoder must execute")
    return dict(executed_nodes=executed, cached_nodes=cached, actual_stage_steps=[8, 3])


async def requests(output, server, launch):
    import aiohttp
    address = f"http://127.0.0.1:{launch['port']}"
    report = dict(status="running", runs=[])
    started = time.perf_counter()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800), trust_env=False) as session:
        while True:
            if server.poll() is not None:
                raise RuntimeError("owned ComfyUI exited before readiness")
            if listener_pid(launch["port"]) == {server.pid}:
                try:
                    async with session.get(address + "/system_stats") as response:
                        response.raise_for_status()
                        stats = await response.json()
                    break
                except aiohttp.ClientConnectionError:
                    pass
            if time.perf_counter() - started > 180:
                raise TimeoutError("owned readiness deadline exceeded")
            await asyncio.sleep(.2)
        report.update(ready_seconds=time.perf_counter() - started, system_stats=stats)
        if not any(d["type"] == "mps" for d in stats["devices"]):
            raise ValueError("actual MPS device required")
        async with session.get(address + "/object_info") as response:
            response.raise_for_status()
            objects = await response.json()
        graph = workflow(0)
        if any(n["class_type"] not in objects for n in graph.values()):
            raise ValueError("stock LTX2.5 workflow nodes unavailable")
        write_json(output / "object-info.json", {n["class_type"]: objects[n["class_type"]] for n in graph.values()})
        client = uuid.uuid4().hex
        async with session.ws_connect(address + "/ws?clientId=" + client) as ws:
            for index in range(launch["count"]):
                if listener_pid(launch["port"]) != {server.pid}:
                    raise RuntimeError("owned listener identity changed")
                graph = workflow(index)
                write_json(output / f"request-{index}.json", graph)
                start = time.perf_counter()
                async with session.post(address + "/prompt", json=dict(prompt=graph, client_id=client)) as response:
                    payload = await response.json()
                    if response.status != 200:
                        raise RuntimeError(json.dumps(payload))
                prompt_id, events = payload["prompt_id"], []
                async with asyncio.timeout(1800):
                    async for message in ws:
                        if message.type != aiohttp.WSMsgType.TEXT:
                            continue
                        event = json.loads(message.data)
                        if event.get("data", {}).get("prompt_id") != prompt_id:
                            continue
                        events.append(dict(seconds=time.perf_counter() - start, **event))
                        if event["type"] in ("execution_error", "execution_interrupted"):
                            write_json(output / f"events-{index}.json", events)
                            raise RuntimeError(json.dumps(event))
                        if event["type"] == "execution_success":
                            break
                    else:
                        raise RuntimeError("websocket closed before successful video")
                for attempt in range(100):
                    async with session.get(address + "/history/" + prompt_id) as response:
                        history = (await response.json()).get(prompt_id)
                    if history:
                        break
                    await asyncio.sleep(.005)
                if not history or history["status"]["status_str"] != "success":
                    raise ValueError("successful complete history required")
                locators = [v for value in history["outputs"].values() for key, items in value.items()
                    if isinstance(items, list) for v in items if isinstance(v, dict) and str(v.get("filename", "")).endswith(".mp4")]
                if len(locators) != 1:
                    write_json(output / f"history-{index}.json", history)
                    raise ValueError("one complete MP4 response required")
                locator = {k: locators[0][k] for k in ("filename", "subfolder", "type")}
                async with session.get(address + "/view", params=locator) as response:
                    response.raise_for_status()
                    media = await response.read()
                (output / f"{index}.mp4").write_bytes(media)
                wall = time.perf_counter() - start
                write_json(output / f"history-{index}.json", history)
                write_json(output / f"events-{index}.json", events)
                report["runs"].append(dict(index=index, seed=42 + index, prompt_id=prompt_id,
                    request_wall_including_export=wall, **validate_events(events, index)))
                write_json(output / "report.json", report)
    report["status"] = "complete"
    return report


def worker(output):
    launch = json.loads((output / "launch.json").read_text())
    report = dict(status="failed", runs=[])
    with (output / "server.stdout.txt").open("x") as out, (output / "server.stderr.txt").open("x") as err:
        server = subprocess.Popen(launch["command"], cwd=launch["comfy"], stdout=out, stderr=err)
        try:
            report = asyncio.run(requests(output, server, launch))
        except BaseException as error:
            if (output / "report.json").is_file():
                report = json.loads((output / "report.json").read_text())
            report.update(status="failed", error=str(error))
            raise
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=5)
            report["server_terminal"] = server.poll() is not None
            write_json(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    for n in ("comfy", "python", "extra"):
        parser.add_argument("--" + n, type=Path)
    parser.add_argument("--count", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        worker(args.output)
        return
    if args.output.exists() or any(getattr(args, n) is None for n in ("comfy", "python", "extra")):
        parser.error("fresh output and explicit existing isolated ComfyUI/Python/packages required")
    args.comfy, args.extra = args.comfy.resolve(strict=True), args.extra.resolve(strict=True)
    args.python = args.python.absolute()
    sources = [(MODEL / p).resolve(strict=True) for p in FILES.values()]
    tracked = subprocess.check_output(["git", "-C", str(args.comfy), "ls-files", "*.py", "comfy/text_encoders/gemma_tokenizer/*"], text=True).splitlines()
    code = [args.comfy / p for p in tracked if (args.comfy / p).is_file()]
    extras = [p for p in args.extra.rglob("*") if p.is_file() and p.stat().st_size > 0 and p.suffix not in (".pyc", ".lock")]
    files = sorted(set([*sources, *code, *extras, Path(__file__).resolve(), ROOT / "tools/validation/z_comfy_framework_screen.py"]))
    identities = {str(p): generation(p) for p in files}
    hashes = registered_hashes(files, ROOT / "outputs/benchmark-source-registrations/comfy-ltx.json")
    if shutil.disk_usage(ROOT).free < 2 << 30:
        raise ValueError("2GiB reserve required; no original assets deleted")
    args.output.mkdir(parents=True)
    (args.output / "instance/user").mkdir(parents=True)
    import yaml
    (args.output / "models.yaml").write_text(yaml.safe_dump(dict(turbocider_benchmark=dict(base_path=str(MODEL),
        diffusion_models="diffusion_models", text_encoders="text_encoders", vae="vae", latent_upscale_models="latent_upscale_models"))))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = launch_command(args.comfy, args.python, args.output, "comfy-default", port)
    command += ["--use-pytorch-cross-attention", "--bf16-vae"]
    write_json(args.output / "launch.json", dict(command=command, comfy=str(args.comfy), port=port, count=args.count))
    data = dict(schema="tc-ltx-comfy-reference-screen-v1", status="running", date="2026-10-12", qualification_passed=False,
        reference_only=True, source_sha256=hashes, model_sha256={str(p): hashes[str(p)] for p in sources},
        comfy_commit=subprocess.check_output(["git", "-C", str(args.comfy), "rev-parse", "HEAD"], text=True).strip(),
        scope="existing local LTX2.5 distilled ConvRot/Gemma4/conv-video+audio VAE;704x448x97/24fps,8+3Euler ancestral,CFG1,no LoRA/no PE;SDPA/BF16 VAE,stock text placement; cold/new-seed repeat, full MP4 wall; native not retested")
    target = args.output / "summary.json"
    write_json(target, data)
    env = benchmark_environment()
    env.update(PYTHONPATH=str(args.extra), PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    try:
        print(json.dumps(dict(starting="comfy-ltx-sdpa", count=args.count)), flush=True)
        with (args.output / "stdout.txt").open("x") as out, (args.output / "stderr.txt").open("x") as err:
            data["memory"] = run_sampled([str(args.python), "-B", str(Path(__file__).resolve()), "--worker", "--output", str(args.output)],
                repo=ROOT, output=args.output, stem="run", env=env, stdout=out, stderr=err, timeout=4000,
                interval_ms=100, max_gap_ms=500, async_evidence=True,
                child_roles=(("comfy_server", str(args.python.resolve(strict=True))), ("listener_probe", "/usr/sbin/lsof")))
        check_generations(identities)
        report = json.loads((args.output / "report.json").read_text())
        if report["status"] != "complete" or report["server_terminal"] is not True or len(report["runs"]) != args.count:
            raise ValueError("owned full video requests must complete and exit")
        media = []
        for i in range(args.count):
            path = args.output / f"{i}.mp4"
            probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)], text=True))
            video = next(s for s in probe["streams"] if s["codec_type"] == "video")
            audio = next(s for s in probe["streams"] if s["codec_type"] == "audio")
            if (video["width"], video["height"], int(video["nb_frames"])) != (704, 448, 97):
                raise ValueError("complete matching97frame AV MP4 required")
            media.append(dict(sha256=sha256_file(path), video=video, audio=audio))
        data.update(status="complete_diagnostic" if args.count == 2 else "complete_precheck_only", report=report, media=media)
    except BaseException as error:
        data.update(status="failed", error=str(error))
        raise
    finally:
        write_json(target, data)
    print(json.dumps(dict(status=data["status"], seconds=[r["request_wall_including_export"] for r in data["report"]["runs"]])))


if __name__ == "__main__":
    main()
