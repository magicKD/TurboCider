#!/usr/bin/env python3
"""Owned ComfyUI MPS / native Z comparison. Never contact a user's server."""
import argparse
import asyncio
import importlib.metadata
import json
import math
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
import uuid

from flux9_bf16_framework_screen import generation, check_generations, registered_hashes, write_json
from image_framework_native import generate_requests
from mlxserve_local_model_probe import listener_pid
from runtime_ane_memory import run_sampled
from z_base_formats_screen import COMMON, CHECKPOINTS, FROZEN, PROMPTS
from z_mflux_framework_screen import requests, route_environment

ROOT = Path(__file__).resolve().parents[2]
ROUTES = ("native-gpu", "native-frozen", "comfy-default", "comfy-sdpa-gpu")
ORDER = (*ROUTES, *reversed(ROUTES))


def workflow(prompt, index):
    # Official Z template topology, Euler instead of res_multistep to match
    # the native task. Change repeat seed, not prompt, to force real sampling.
    def node(kind, **inputs):
        return dict(class_type=kind, inputs=inputs)
    return {
        "1": node("UNETLoader", unet_name="z_image_turbo_bf16.safetensors", weight_dtype="default"),
        "2": node("CLIPLoader", clip_name="qwen_3_4b.safetensors", type="lumina2", device="default"),
        "3": node("VAELoader", vae_name="ae.safetensors"),
        "4": node("CLIPTextEncode", clip=["2", 0], text=prompt),
        "5": node("ConditioningZeroOut", conditioning=["4", 0]),
        "6": node("EmptySD3LatentImage", width=512, height=512, batch_size=1),
        "7": node("ModelSamplingAuraFlow", model=["1", 0], shift=3.0),
        "8": node("KSampler", model=["7", 0], positive=["4", 0], negative=["5", 0],
                  latent_image=["6", 0], seed=43 if index == 3 else 42, steps=8,
                  cfg=1.0, sampler_name="euler", scheduler="simple", denoise=1.0),
        "9": node("VAEDecode", samples=["8", 0], vae=["3", 0]),
        "10": node("SaveImage", images=["9", 0], filename_prefix=f"request-{index}"),
    }


def launch_command(comfy, python, output, route, port):
    if route not in ROUTES[2:]:
        raise ValueError("declared ComfyUI MPS route required")
    command = [str(python), "-B", "-u", str(comfy / "main.py"), "--listen", "127.0.0.1", "--port", str(port),
        "--base-directory", str(output / "instance"), "--extra-model-paths-config", str(output / "models.yaml"),
        "--user-directory", str(output / "instance/user"), "--database-url", "sqlite:///:memory:",
        "--output-directory", str(output / "instance/output"), "--temp-directory", str(output / "instance/temp"),
        "--disable-all-custom-nodes", "--disable-api-nodes", "--disable-auto-launch", "--preview-method", "none"]
    if route == "comfy-sdpa-gpu":
        command += ["--use-pytorch-cross-attention", "--gpu-only", "--bf16-unet", "--bf16-text-enc", "--bf16-vae"]
    return command


def validate_comfy_row(row, index):
    wall = row.get("request_wall_including_export")
    if type(wall) not in (float, int) or not math.isfinite(wall) or wall <= 0:
        raise ValueError("finite positive complete PNG wall required")
    if row.get("index") != index or row.get("seed") != (43 if index == 3 else 42):
        raise ValueError("cold/two-fresh/repeated-prompt-new-seed sequence differs")
    if row.get("actual_steps") != list(range(1, 9)):
        raise ValueError("all eight sampler steps required; cached KSampler is not inference")
    executed = row.get("executed_nodes", [])
    cached = row.get("cached_nodes", [])
    if any(node not in executed or node in cached for node in ("8", "9", "10")):
        raise ValueError("KSampler/VAE/SaveImage must really execute")
    if index != 3 and ("4" not in executed or "4" in cached):
        raise ValueError("fresh prompt must really execute text encoder")
    if row.get("history_status") != "success":
        raise ValueError("successful complete history required")


def validate_trial(trial, count=4):
    route, report = trial["route"], trial["report"]
    if route not in ROUTES or len(report["runs"]) != count or trial["memory"]["verified"]["complete"] is not True:
        raise ValueError("known complete trial and independent memory required")
    if route.startswith("comfy-"):
        if report.get("server_terminal") is not True or report.get("status") != "complete":
            raise ValueError("owned server must complete and exit")
        for index, row in enumerate(report["runs"]):
            validate_comfy_row(row, index)
    else:
        for index, row in enumerate(report["runs"]):
            m = row["metrics"]
            if (m["model"], m["width"], m["height"], m["steps"], m["seed"], m["prompt_cache_hit"]) != (
                    "z-image-turbo", 512, 512, 8, 43 if index == 3 else 42, index == 3):
                raise ValueError("actual native task/cache differs")
            wall = row["request_wall_including_export"]
            if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
                raise ValueError("finite positive native wall required")
            h = m.get("hybrid")
            if route == "native-gpu" and h:
                raise ValueError("pure GPU cannot be hybrid")
            if route == "native-frozen" and (not h or h.get("runtime_calls_session_total") != 256 * (index + 1) or
                    h.get("checkpoint_sha256_verified") is not True or h.get("runtime_failed") is not False or
                    h.get("runtime_failures_session_total") != 0 or h.get("warmup_calls_session_total") != 0 or
                    h.get("ane_mlp_range") != [0, 5120] or h.get("compute_units") != "cpuAndNeuralEngine"):
                raise ValueError("actual complete source-bound frozen FFN work required")


async def comfy_requests(output, server):
    import aiohttp
    config = json.loads((output / "launch.json").read_text())
    address = f"http://127.0.0.1:{config['port']}"
    report = dict(status="running", runs=[], cache_policy="stock RAM pressure; repeat prompt uses seed43 to force sampling",
                  packages={p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "Pillow", "aiohttp")})
    started = time.perf_counter()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600), trust_env=False) as session:
        while True:
            if server.poll() is not None:
                raise RuntimeError("owned ComfyUI exited before readiness")
            if listener_pid(config["port"]) == {server.pid}:
                try:
                    async with session.get(address + "/system_stats") as response:
                        response.raise_for_status()
                        stats = await response.json()
                    break
                except (aiohttp.ClientConnectionError, TimeoutError):
                    pass
            if time.perf_counter() - started > 180:
                raise TimeoutError("owned ComfyUI readiness deadline exceeded")
            await asyncio.sleep(.2)
        report.update(ready_seconds=time.perf_counter() - started, system_stats=stats)
        if not any(d.get("type") == "mps" for d in stats.get("devices", [])):
            raise ValueError("actual MPS server device required")
        async with session.get(address + "/object_info") as response:
            response.raise_for_status()
            objects = await response.json()
        graph = workflow(PROMPTS[0], 0)
        if any(n["class_type"] not in objects for n in graph.values()):
            raise ValueError("stock Z workflow nodes unavailable")
        write_json(output / "object-info.json", {n["class_type"]: objects[n["class_type"]] for n in graph.values()})
        client = uuid.uuid4().hex
        async with session.ws_connect(address + "/ws?clientId=" + client) as ws:
            for index, prompt in enumerate(PROMPTS[:config["count"]]):
                if listener_pid(config["port"]) != {server.pid}:
                    raise RuntimeError("owned listener identity changed")
                graph = workflow(prompt, index)
                write_json(output / f"request-{index}.json", graph)
                start = time.perf_counter()
                async with session.post(address + "/prompt", json=dict(prompt=graph, client_id=client)) as response:
                    response.raise_for_status()
                    submitted = await response.json()
                prompt_id = submitted["prompt_id"]
                events = []
                async with asyncio.timeout(600):
                    async for message in ws:
                        if message.type != aiohttp.WSMsgType.TEXT:
                            continue
                        event = json.loads(message.data)
                        if event.get("data", {}).get("prompt_id") != prompt_id:
                            continue
                        events.append(dict(seconds=time.perf_counter() - start, **event))
                        if event["type"] in ("execution_error", "execution_interrupted"):
                            raise RuntimeError(json.dumps(event))
                        if event["type"] == "execution_success":
                            break
                    else:
                        raise RuntimeError("websocket closed before successful execution")
                # Success is enqueued just before history is committed. This
                # bounded yield waits for history, not for an arbitrary second.
                for attempt in range(100):
                    async with session.get(address + "/history/" + prompt_id) as response:
                        response.raise_for_status()
                        history = (await response.json()).get(prompt_id)
                    if history:
                        break
                    await asyncio.sleep(.005)
                if not history:
                    raise RuntimeError("successful history missing")
                images = [im for value in history["outputs"].values() for im in value.get("images", [])]
                if len(images) != 1:
                    raise ValueError("exactly one complete PNG required")
                async with session.get(address + "/view", params=images[0]) as response:
                    response.raise_for_status()
                    png = await response.read()
                if not png.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("PNG payload required")
                (output / f"{index}.png").write_bytes(png)
                wall = time.perf_counter() - start
                executed = [e["data"]["node"] for e in events if e["type"] == "executing" and e["data"].get("node")]
                cached = [n for e in events if e["type"] == "execution_cached" for n in e["data"]["nodes"]]
                steps = [e["data"]["value"] for e in events if e["type"] == "progress" and e["data"].get("node") == "8"]
                row = dict(index=index, seed=graph["8"]["inputs"]["seed"], prompt=prompt, prompt_id=prompt_id,
                           request_wall_including_export=wall, actual_steps=steps, executed_nodes=executed,
                           cached_nodes=cached, history_status=history["status"]["status_str"])
                write_json(output / f"events-{index}.json", events)
                write_json(output / f"history-{index}.json", history)
                validate_comfy_row(row, index)
                report["runs"].append(row)
                write_json(output / "report.json", report)
    report["status"] = "complete"
    return report


def comfy_worker(output):
    launch = json.loads((output / "launch.json").read_text())
    report = dict(status="failed", runs=[])
    with (output / "server.stdout.txt").open("x") as out, (output / "server.stderr.txt").open("x") as err:
        server = subprocess.Popen(launch["command"], cwd=launch["comfy"], stdout=out, stderr=err)
        try:
            report = asyncio.run(comfy_requests(output, server))
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
    parser.add_argument("--worker", choices=ROUTES)
    parser.add_argument("--comfy-extra", type=Path, help="isolated pinned package overlay; never install into user's environment")
    for name in ("comfy", "comfy-python", "native-python", "library"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--count", type=int, choices=(1, 4), default=4)
    parser.add_argument("--order", default=",".join(ORDER))
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        if args.worker.startswith("comfy-"):
            comfy_worker(args.output)
        else:
            native_requests = requests(args.output, args.worker, 8, args.count)
            if args.count == 4:
                native_requests[3]["seed"] = 43
            report = generate_requests(args.library, "z-image-turbo", COMMON, native_requests, args.output, args.worker)
            write_json(args.output / "report.json", report)
        return
    if args.output.exists() or any(getattr(args, n) is None for n in ("comfy", "comfy_python", "native_python", "library")):
        parser.error("fresh output and explicit existing installations/library required")
    args.comfy, args.library = args.comfy.resolve(strict=True), args.library.resolve(strict=True)
    args.comfy_python, args.native_python = args.comfy_python.absolute(), args.native_python.absolute()
    order = args.order.split(",")
    if any(r not in ROUTES for r in order) or args.count == 4 and order != list(ORDER):
        parser.error("declared precheck or complete four-route palindrome required")
    build_path = args.library.parent / "runtime-build/runtime-build-manifest.json"
    build = json.loads(build_path.read_text())
    from runtime_ane_common import sha256_file
    if any(sha256_file(ROOT / r["path"]) != r["sha256"] for r in build["inputs"]["sources"]):
        raise ValueError("native library inputs differ from current source")
    source_files = [CHECKPOINTS["bf16"], COMMON / "split_files/text_encoders/qwen_3_4b.safetensors",
                    COMMON / "split_files/vae/ae.safetensors"]
    tracked = subprocess.check_output(["git", "-C", str(args.comfy), "ls-files", "*.py", "comfy/text_encoders/qwen25_tokenizer/*"], text=True).splitlines()
    code_files = [args.comfy / p for p in tracked if (args.comfy / p).is_file()]
    ane_files = [p for p in FROZEN.parent.rglob("*") if p.is_file() and p.suffix != ".lock"]
    own_files = [ROOT / "tools/validation" / name for name in ("z_base_formats_screen.py", "z_mflux_framework_screen.py",
        "image_framework_native.py", "runtime_ane_memory.py", "runtime_ane_common.py", "flux9_bf16_framework_screen.py")]
    extra_files = [p for p in args.comfy_extra.resolve(strict=True).rglob("*") if p.is_file() and p.stat().st_size > 0
                   and p.suffix not in (".pyc", ".lock")] if args.comfy_extra else []
    files = sorted(set([*source_files, *code_files, *ane_files, *own_files, *extra_files, args.library, build_path, Path(__file__).resolve()]))
    identities = {str(p): generation(p) for p in files}
    hashes = registered_hashes(files, ROOT / "outputs/benchmark-source-registrations/unsloth-image.json")
    args.output.mkdir(parents=True)
    data = dict(schema="tc-z-comfy-four-route-screen-v1", date="2026-10-12", status="running", steps=8, count=args.count,
        order=order, trials=[], qualification_passed=False, library_sha256=hashes[str(args.library)],
        native_inputs_matched=len(build["inputs"]["sources"]), source_sha256=hashes,
        model_sha256={str(p): hashes[str(p)] for p in source_files},
        comfy_commit=subprocess.check_output(["git", "-C", str(args.comfy), "rev-parse", "HEAD"], text=True).strip(),
        comfy_local_changes=subprocess.check_output(["git", "-C", str(args.comfy), "status", "--short"], text=True).splitlines(),
        timing_scope="full512 PNG: native C API/export or owned ComfyUI HTTP submit/WebSocket success/history/PNG download/export; readiness/cold/repeat separate",
        representation_scope="same original BF16 encoder/DiT/VAE; Euler/simple shift3/cfg1; ComfyUI stock vs explicit SDPA+gpu-only/BF16; frozen W8A8 approximate; independent RNG/tokenization/compute/cache lifecycles",
        source_policy="initial source-generation SHA reused by metadata; reference/user service/models unchanged; all custom/API nodes disabled; repeat prompt seed43 forces real sampler/VAE")
    target = args.output / "summary.json"
    write_json(target, data)
    try:
        for index, route in enumerate(order):
            check_generations(identities)
            output = args.output / f"{index}-{route}"
            output.mkdir()
            env = route_environment(route if route.startswith("native-") else "mflux-bf16")
            env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
            if route.startswith("comfy-") and args.comfy_extra:
                env["PYTHONPATH"] = str(args.comfy_extra.resolve(strict=True))
            # Fail visibly for unsupported MPS ops, rather than a hidden
            # automatic torch CPU fallback. Explicit CPU text is still stock.
            env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
            python = args.comfy_python if route.startswith("comfy-") else args.native_python
            command = [str(python), "-B", str(Path(__file__).resolve()), "--worker", route, "--output", str(output),
                       "--library", str(args.library), "--count", str(args.count)]
            roles = ()
            if route.startswith("comfy-"):
                (output / "instance/user").mkdir(parents=True)
                import yaml
                (output / "models.yaml").write_text(yaml.safe_dump(dict(turbocider_benchmark=dict(
                    base_path=str(COMMON / "split_files"), diffusion_models="diffusion_models", text_encoders="text_encoders", vae="vae"))))
                with socket.socket() as reservation:
                    reservation.bind(("127.0.0.1", 0))
                    port = reservation.getsockname()[1]
                write_json(output / "launch.json", dict(command=launch_command(args.comfy, python, output, route, port),
                    port=port, comfy=str(args.comfy), count=args.count))
                roles = (("comfy_server", str(python.resolve(strict=True))), ("listener_probe", "/usr/sbin/lsof"))
            print(json.dumps(dict(starting=index, route=route)), flush=True)
            with (output / "stdout.txt").open("x") as out, (output / "stderr.txt").open("x") as err:
                memory = run_sampled(command, repo=ROOT, output=output, stem="run", env=env, stdout=out, stderr=err,
                    timeout=1800, interval_ms=100, max_gap_ms=500, async_evidence=True, child_roles=roles)
            check_generations(identities)
            report = json.loads((output / "report.json").read_text())
            trial = dict(index=index, route=route, report=report, memory=memory, png_sha256=[])
            validate_trial(trial, args.count)
            from PIL import Image
            for request_index in range(args.count):
                path = output / f"{request_index}.png"
                with Image.open(path) as im:
                    im.load()
                    if im.size != (512, 512):
                        raise ValueError("full size image required")
                trial["png_sha256"].append(sha256_file(path))
            data["trials"].append(trial)
            write_json(target, data)
        if args.count == 4:
            data["medians"] = {}
            for route in ROUTES:
                trials = [t for t in data["trials"] if t["route"] == route]
                samples = [r["request_wall_including_export"] for t in trials for r in t["report"]["runs"][1:3]]
                data["medians"][route] = dict(samples=samples, fresh_warm_seconds=statistics.median(samples),
                    cold_seconds=[t["report"]["runs"][0]["request_wall_including_export"] for t in trials],
                    repeat_seconds=[t["report"]["runs"][3]["request_wall_including_export"] for t in trials],
                    constructor_seconds=[t["report"].get("ready_seconds", t["report"].get("constructor_seconds")) for t in trials])
        data["status"] = "complete_diagnostic" if args.count == 4 else "complete_precheck_only"
    except BaseException as error:
        data.update(status="failed", error=str(error))
        raise
    finally:
        write_json(target, data)
    print(json.dumps(dict(status=data["status"], medians=data.get("medians"))))


if __name__ == "__main__":
    main()
