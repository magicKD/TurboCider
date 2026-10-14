#!/usr/bin/env python3
"""Owned complete SeedVR2 request observer; no user server is contacted.

Full warm requests re-run media import, resize, VAE encode, the one sampler
step, VAE decode and export. Fresh byte-identical input filenames invalidate
ComfyUI's encoded-latent cache while allowing model loaders to stay warm.
This tool records observations; a separate paired cohort is required before
claiming that the requested image/video speed gate is satisfied.
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
import statistics
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_request(args, source, output, index):
    hybrid = args.route == "native-hybrid"
    execution = dict(policy="gpu_ane" if hybrid else "gpu", residency=getattr(args,"residency","resident"))
    options = getattr(args,"vae_options",None)
    if options is not None: execution["seedvr2_vae"] = options
    if hybrid:
        execution.update(ane_manifest=str(args.ane_manifest), allow_approximation=True)
    return dict(schema_version=2, model="seedvr2-3b", operation=f"{args.kind}.upscale",
        inputs=[dict(kind=args.kind, role="source", path=str(source))],
        outputs=[dict(kind=args.kind, path=str(output), width=args.width, height=args.height,
                      frames=args.frames, fps=args.fps, audio=False)],
        sampling=dict(seed=args.seed + index, steps=1), execution=execution)


def check_native_execution(args, result):
    hybrid = args.route == "native-hybrid"
    execution = "gpu_ane" if hybrid else "gpu"
    recipe = "seedvr2-original-fp16-coreml-prefix-v1" if hybrid else "seedvr2-stable-fp16-fp32-v1"
    causal=getattr(args,"experimental_causal_manifest",None)
    causal_calls=(1+((args.frames-1+3)//4)*4-1)*math.ceil(args.height/64)*math.ceil(args.width/64) if args.kind=="video" and causal else 0
    if causal_calls:execution="gpu_vae_cpu_ne_experimental"
    options = getattr(args,"vae_options",None)
    if options is not None and options.get("encoder_spatial","full") == "file_rows":
        recipe += "+vae-encoder-head-globalstats-file-rows-v1"
    if options is not None and options.get("spatial","full") == "file_rows":
        recipe += "+vae-globalstats-file-rows-v1"
    if causal_calls:recipe+="+experimental-vae-causal-coreml-original-fp16-v1"
    if causal:
        expected_compute="cpuOnly" if getattr(args,"experimental_causal_compute","ne")=="cpu" else "cpuAndNeuralEngine"
        if (result.get("vae_causal_conv_experimental_enabled") is not True or
            result.get("vae_causal_conv_calls")!=causal_calls or result.get("vae_causal_conv_backing_hits")!=causal_calls or
            result.get("vae_causal_conv_frames")!=(1+((args.frames-1+3)//4)*4-1 if args.kind=="video" else 0) or
            result.get("vae_causal_conv_compute_units")!=expected_compute or
            result.get("vae_causal_conv_manifest_sha256")!=sha256(causal) or result.get("physical_ane_occupancy_measured") is not False):
            raise ValueError("experimental full-request causal execution differs")
    elif result.get("vae_causal_conv_experimental_enabled",False):
        raise ValueError("unexpected experimental VAE offload")
    if (result.get("actual_denoise_steps") != 1 or result.get("conditioning_cache_hit") or
            result.get("execution") != execution or result.get("numerical_recipe") != recipe or
            result.get("dit_projection_dtype") != "fp16" or result.get("dit_residual_dtype") != "fp32"):
        raise ValueError("complete native request did not execute its declared precision/route")
    if hybrid:
        frames = 1 + ((args.frames - 1 + 3) // 4) * 4
        tokens = ((frames - 1) // 4 + 1) * (args.height // 16) * (args.width // 16)
        if (result.get("coreml_prefix_calls") != 32*((tokens+511)//512) or
                result.get("coreml_prefix_video_blocks") != 32 or
                result.get("coreml_prefix_compute_units") != "cpuAndNeuralEngine" or
                result.get("coreml_prefix_manifest_sha256") != sha256(args.ane_manifest)):
            raise ValueError("hybrid complete request lacked the declared all-block partition execution")
    elif result.get("coreml_prefix_calls",0) != 0:
        raise ValueError("pure-GPU complete request executed an unexpected Core ML prefix")
    residency = getattr(args,"residency","resident")
    options = getattr(args,"vae_options",None)
    if options is not None:
        expected = dict(encoder="buffered",decoder="buffered",convolution="singleton2d",history="compact",tile_rows=16384,allocator_cache_bytes=4 << 30,
            spatial="full",encoder_spatial="full",spatial_rows=64,spatial_disk_budget_bytes=8 << 30)
        expected.update(options)
        if (result.get("vae_encoder_recipe","buffered") != expected["encoder"] or
            result.get("streamed_input_frames",0) != (args.frames if expected["encoder"] == "streaming" else 0) or
            result.get("vae_decoder_recipe") != expected["decoder"] or
            result.get("vae_conv_recipe") != expected["convolution"] or
            result.get("vae_history_recipe") != expected["history"] or
            result.get("vae_conv_tile_rows_limit") != expected["tile_rows"] or
            result.get("allocator_cache_limit_bytes") != expected["allocator_cache_bytes"] or
            result.get("streamed_output_frames") != (args.frames if expected["decoder"] == "streaming" else 0)):
            raise ValueError("native VAE request controls did not execute as declared")
        if (result.get("vae_spatial_recipe","full") != expected["spatial"] or
            (expected["spatial"] == "file_rows" and
             (result.get("vae_spatial_rows") != expected["spatial_rows"] or
              result.get("vae_spatial_disk_budget_bytes") != expected["spatial_disk_budget_bytes"] or
              result.get("vae_spatial_file_live_bytes_after_decode") != 0 or
              result.get("vae_spatial_host_live_bytes_after_decode") != 0 or
              result.get("vae_spatial_cleanup_failures") != 0))):
            raise ValueError("native spatial schedule/settings/retirement differ from request")
        if (result.get("vae_encoder_spatial_recipe","full")!=expected["encoder_spatial"] or
            (expected["encoder_spatial"]=="file_rows" and
             (result.get("vae_encoder_spatial_file_live_bytes_after_encode")!=0 or
              result.get("vae_encoder_spatial_host_live_bytes_after_encode")!=0 or
              result.get("vae_encoder_spatial_cleanup_failures")!=0))):
            raise ValueError("native encoder spatial schedule/retirement differs from request")
    if "weight_residency_recipe" in result and result["weight_residency_recipe"] != ("stage_dit" if residency == "component_staged" else "resident"):
        raise ValueError("native request executed a different residency than requested")


def workflow(args, filename, index):
    def node(kind, **inputs):
        return dict(class_type=kind, inputs=inputs)
    graph = {
        "1": node("UNETLoader", unet_name="seedvr2_3b_fp16.safetensors", weight_dtype="default"),
        "2": node("VAELoader", vae_name="seedvr2_ema_vae_fp16.safetensors"),
    }
    if args.kind == "image":
        graph["3"] = node("LoadImage", image=filename)
        images = ["3", 0]
    else:
        graph["3"] = node("LoadVideo", file=filename)
        graph["4"] = node("GetVideoComponents", video=["3", 0])
        images = ["4", 0]
    graph.update({
        "5": node("ImageScale", image=images, upscale_method="bicubic", width=args.width, height=args.height, crop="disabled"),
        "6": node("SeedVR2Preprocess", resized_images=["5", 0]),
        "7": node("VAEEncode", pixels=["6", 0], vae=["2", 0]),
        "8": node("SeedVR2Conditioning", model=["1", 0], vae_conditioning=["7", 0]),
        "9": node("KSampler", model=["1", 0], positive=["8", 0], negative=["8", 1], latent_image=["7", 0],
                  seed=args.seed + index, steps=1, cfg=1.0, sampler_name="euler", scheduler="simple", denoise=1.0),
        "10": node("VAEDecode", samples=["9", 0], vae=["2", 0]),
        "11": node("SeedVR2PostProcessing", images=["10", 0], original_resized_images=["5", 0], color_correction_method="none"),
    })
    if args.kind == "image":
        graph["12"] = node("SaveImage", images=["11", 0], filename_prefix=f"request-{index}")
    else:
        graph["12"] = node("CreateVideo", images=["11", 0], fps=args.fps, bit_depth=8, color_space="sRGB", codec="none")
        graph["13"] = node("SaveVideo", video=["12", 0], filename_prefix=f"request-{index}", format="mp4", codec={"codec": "h264"})
    return graph


def verify_output(path, args):
    if args.kind == "image":
        from PIL import Image
        with Image.open(path) as image:
            if image.size != (args.width, args.height):
                raise ValueError("complete output dimensions differ")
        return dict(width=args.width, height=args.height, frames=1)
    ffprobe = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
    info = json.loads(subprocess.check_output([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate,duration", "-of", "json", str(path)], text=True))["streams"][0]
    numerator, denominator = map(int, info["r_frame_rate"].split("/"))
    if (info["width"], info["height"], int(info["nb_read_frames"]), numerator / denominator) != (
            args.width, args.height, args.frames, args.fps):
        raise ValueError("complete video output frame/FPS/dimension mismatch")
    return info


def validate_native_build_binding(library,identity):
    import ctypes as C
    loaded=C.CDLL(str(library))
    loaded.tc_runtime_build_identity.restype=C.c_void_p
    loaded.tc_string_free.argtypes=[C.c_void_p]
    loaded.tc_string_free.restype=None
    pointer=loaded.tc_runtime_build_identity()
    if not pointer:raise ValueError("native runtime build identity is unavailable")
    try:actual=C.string_at(pointer).decode()
    finally:loaded.tc_string_free(pointer)
    if not isinstance(identity.get("runtime_build_id"),str) or actual!=identity["runtime_build_id"]:
        raise ValueError("native library and declared build identity do not match")
    return actual


def record_setup(args):
    if args.output.exists():
        raise ValueError("a fresh observation directory is required")
    args.output.mkdir(parents=True)
    sources = [args.model / "diffusion_models/seedvr2_3b_fp16.safetensors",
               args.model / "vae/seedvr2_ema_vae_fp16.safetensors"]
    source_json = json.loads((args.model / "source.json").read_text())
    expected = {row["path"]: row for row in source_json["files"]}
    for path in sources:
        row = expected[str(path.relative_to(args.model))]
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError("original ModelScope source no longer matches its verified provenance")
    report = dict(schema="seedvr2-full-request-observer-v1", route=args.route, status="running", runs=[],
        source_path=str(args.source), source_sha256=sha256(args.source), model_provenance=source_json,
        workload=dict(kind=args.kind, width=args.width, height=args.height, frames=args.frames, fps=args.fps,
                      seed=args.seed, steps=1, sampler="euler", scheduler="simple", cfg=1, color_correction="none"),
        timing_scope="complete per-request import/resize/encode/sampler/decode/export; constructor/readiness/cold separately",
        qualification_passed=False,
        limit="one route observer, not a paired cold/warm cohort or standalone proof of GPU speed versus ComfyUI")
    if args.route in ("native-gpu", "native-hybrid"):
        build_manifest = getattr(args,"native_build_manifest",None) or args.library.parent / "runtime-build/runtime-build-manifest.json"
        if not build_manifest.is_file():
            raise ValueError("native runtime build identity manifest is missing")
        identity = json.loads(build_manifest.read_text())
        runtime_id=validate_native_build_binding(args.library,identity)
        write_json(args.output / "native-build-manifest.json", identity)
        report["native_artifact_identity"] = dict(library_sha256=sha256(args.library),
            runtime_build_manifest_sha256=sha256(build_manifest),
            verified_runtime_build_id=runtime_id,
            environment={key: os.environ.get(key) for key in (
                "TURBOCIDER_SEEDVR2_VAE_CONV_RECIPE", "TURBOCIDER_SEEDVR2_VAE_NORM_RECIPE",
                "TURBOCIDER_SEEDVR2_VAE_HISTORY_RECIPE", "TURBOCIDER_SEEDVR2_VAE_POINTWISE_RECIPE",
                "TURBOCIDER_SEEDVR2_VAE_CONV_TILE_ROWS", "TURBOCIDER_SEEDVR2_ALLOCATOR_CACHE_BYTES")})
        report["native_artifact_identity"]["dit_environment"] = {key:os.environ.get(key) for key in (
            "TURBOCIDER_SEEDVR2_DIT_RMS_RECIPE","TURBOCIDER_SEEDVR2_DIT_TIME_RECIPE",
            "TURBOCIDER_SEEDVR2_DIT_ATTENTION_RECIPE","TURBOCIDER_SEEDVR2_DIT_ROPE_RECIPE",
            "TURBOCIDER_SEEDVR2_DIT_RESIDUAL_RECIPE","TURBOCIDER_SEEDVR2_DIT_LINEAR_RECIPE")}
        report["native_artifact_identity"]["residency"] = getattr(args,"residency","resident")
        if getattr(args,"experimental_causal_manifest",None):
            report["native_artifact_identity"]["experimental_causal_manifest_sha256"]=sha256(args.experimental_causal_manifest)
            report["native_artifact_identity"]["experimental_causal_compute"]=args.experimental_causal_compute
        if args.route == "native-hybrid":
            report["native_artifact_identity"]["prefix_manifest_sha256"] = sha256(args.ane_manifest)
    write_json(args.output / "report.json", report)
    return report


def native(args, report):
    from image_framework_native import generate_requests
    requests = []
    suffix = ".mp4" if args.kind == "video" else ".png"
    for index in range(args.count):
        source = args.output / f"input-{index}{args.source.suffix}"
        shutil.copy2(args.source, source)
        requests.append(native_request(args, source, args.output / f"output-{index}{suffix}", index))
    observed = generate_requests(args.library, "seedvr2-3b", args.model, requests, args.output, args.route)
    report.update(observed)
    for row in report["runs"]:
        index = row["index"]
        result = row["metrics"]
        check_native_execution(args,result)
        if getattr(args,"capture_raw_video",False):
            capture=result.get("video_raw_capture") or {}
            path=args.output/"raw-pixels/capture.json"
            if (capture.get("manifest")!=str(path) or capture.get("frames")!=args.frames or
                capture.get("diagnostic_only") is not True or capture.get("sha256")!=sha256(path)):
                raise ValueError("complete streaming request did not bind its raw capture")
        row["output"] = verify_output(Path(result["output"]), args)
        row["output_sha256"] = sha256(Path(result["output"]))
    write_json(args.output / "report.json", report)


def configure_raw_capture(args):
    if getattr(args,"capture_raw_video",False):
        options=args.vae_options or {}
        if (args.route!="native-gpu" or args.kind!="video" or args.count!=1 or
            options.get("encoder")!="streaming" or options.get("decoder")!="streaming"):
            raise ValueError("raw capture requires one explicit GPU-backbone streaming video diagnostic")
        os.environ["TURBOCIDER_SEEDVR2_VIDEO_RAW_CAPTURE"]=str(args.output.absolute()/"raw-pixels")
        os.environ["TURBOCIDER_SEEDVR2_VIDEO_PIXEL_HASH"]="1"
    elif os.environ.get("TURBOCIDER_SEEDVR2_VIDEO_RAW_CAPTURE"):
        raise ValueError("inherited raw capture must be explicitly recorded by --capture-raw-video")


async def comfy(args, report):
    import aiohttp
    import yaml
    from mlxserve_local_model_probe import listener_pid
    instance = args.output / "instance"
    for name in ("input", "output", "temp", "user"):
        (instance / name).mkdir(parents=True)
    for index in range(args.count):
        shutil.copy2(args.source, instance / "input" / f"input-{index}{args.source.suffix}")
    (args.output / "models.yaml").write_text(yaml.safe_dump(dict(seedvr2_observer=dict(
        base_path=str(args.model), diffusion_models="diffusion_models", vae="vae"))))
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    command = [str(args.comfy_python), "-B", "-u", str(args.comfy / "main.py"), "--listen", "127.0.0.1", "--port", str(port),
        "--base-directory", str(instance), "--input-directory", str(instance / "input"), "--output-directory", str(instance / "output"),
        "--temp-directory", str(instance / "temp"), "--user-directory", str(instance / "user"),
        "--database-url", "sqlite:///:memory:", "--extra-model-paths-config", str(args.output / "models.yaml"),
        "--disable-all-custom-nodes", "--disable-api-nodes", "--disable-auto-launch", "--preview-method", "none", "--fp16-vae"]
    if args.route == "comfy-sdpa-gpu":
        command += ["--use-pytorch-cross-attention", "--gpu-only", "--fp16-unet"]
    env = os.environ.copy()
    env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    env["PYTHONPATH"] = str(args.comfy_extra)
    report["comfy_commit"] = subprocess.check_output(["git", "-C", str(args.comfy), "rev-parse", "HEAD"], text=True).strip()
    report["comfy_local_changes"] = subprocess.check_output(["git", "-C", str(args.comfy), "diff", "--"], text=True)
    report["comfy_seedvr2_vae_sha256"] = sha256(args.comfy / "comfy/ldm/seedvr/vae.py")
    write_json(args.output / "launch.json", dict(command=command, port=port))
    with (args.output / "server.stdout.txt").open("x") as stdout, (args.output / "server.stderr.txt").open("x") as stderr:
        server = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
        try:
            address = f"http://127.0.0.1:{port}"
            started = time.perf_counter()
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900), trust_env=False) as session:
                while True:
                    if server.poll() is not None:
                        raise RuntimeError("owned ComfyUI server exited before readiness; inspect server.stderr.txt")
                    if listener_pid(port) == {server.pid}:
                        try:
                            async with session.get(address + "/system_stats") as response:
                                response.raise_for_status()
                                stats = await response.json()
                            break
                        except aiohttp.ClientConnectionError:
                            pass
                    if time.perf_counter() - started > 180:
                        raise TimeoutError("owned server readiness deadline exceeded")
                    await asyncio.sleep(.2)
                if not any(device.get("type") == "mps" for device in stats["devices"]):
                    raise ValueError("a real MPS server is required; CPU fallback is not a baseline")
                report.update(readiness_seconds=time.perf_counter()-started, system_stats=stats)
                async with session.get(address + "/object_info") as response:
                    response.raise_for_status()
                    objects = await response.json()
                graph = workflow(args, f"input-0{args.source.suffix}", 0)
                if any(node["class_type"] not in objects for node in graph.values()):
                    raise ValueError("a required stock SeedVR2 workflow node is unavailable")
                write_json(args.output / "object-info.json", {node["class_type"]: objects[node["class_type"]] for node in graph.values()})
                client = uuid.uuid4().hex
                async with session.ws_connect(address + "/ws?clientId=" + client) as ws:
                    for index in range(args.count):
                        if listener_pid(port) != {server.pid}:
                            raise RuntimeError("owned server listener identity changed")
                        graph = workflow(args, f"input-{index}{args.source.suffix}", index)
                        write_json(args.output / f"request-{index}.json", graph)
                        print(json.dumps(dict(starting=index, route=args.route)), flush=True)
                        start = time.perf_counter()
                        async with session.post(address + "/prompt", json=dict(prompt=graph, client_id=client)) as response:
                            response.raise_for_status()
                            submitted = await response.json()
                        if "prompt_id" not in submitted:
                            raise ValueError(json.dumps(submitted))
                        prompt_id = submitted["prompt_id"]
                        events = []
                        async with asyncio.timeout(900):
                            async for message in ws:
                                if message.type != aiohttp.WSMsgType.TEXT:
                                    continue
                                event = json.loads(message.data)
                                if event.get("data", {}).get("prompt_id") != prompt_id:
                                    continue
                                events.append(dict(seconds=time.perf_counter()-start, **event))
                                if event["type"] in ("execution_error", "execution_interrupted"):
                                    write_json(args.output / f"events-{index}.json", events)
                                    raise RuntimeError(json.dumps(event))
                                if event["type"] == "execution_success":
                                    break
                            else:
                                raise RuntimeError("websocket closed before successful execution")
                        wall = time.perf_counter() - start
                        for _ in range(100):
                            async with session.get(address + "/history/" + prompt_id) as response:
                                response.raise_for_status()
                                history = (await response.json()).get(prompt_id)
                            if history:
                                break
                            await asyncio.sleep(.01)
                        if not history or history["status"]["status_str"] != "success":
                            raise ValueError("complete successful history was not observed")
                        write_json(args.output / f"events-{index}.json", events)
                        write_json(args.output / f"history-{index}.json", history)
                        executed = {e["data"]["node"] for e in events if e["type"] == "executing" and e["data"].get("node")}
                        cached = {node for e in events if e["type"] == "execution_cached" for node in e["data"]["nodes"]}
                        required = {"3", "5", "6", "7", "8", "9", "10", "11", "12"}
                        if args.kind == "video": required |= {"4", "13"}
                        if not required.issubset(executed) or required & cached:
                            raise ValueError("media/encode/sampler/decode/export must genuinely execute every request")
                        files = [path for path in (instance / "output").rglob(f"request-{index}*") if path.suffix == (".mp4" if args.kind == "video" else ".png")]
                        if len(files) != 1:
                            raise ValueError("exactly one complete output file is required")
                        row = dict(index=index, seed=args.seed+index, request_wall_including_export=wall,
                            executed_nodes=sorted(executed), cached_nodes=sorted(cached),
                            output_path=str(files[0]), output_sha256=sha256(files[0]), output=verify_output(files[0], args))
                        report["runs"].append(row)
                        write_json(args.output / "report.json", report)
                        print(json.dumps(row), flush=True)
        finally:
            if server.poll() is None:
                server.terminate()
                try: server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill(); server.wait(timeout=10)
            report["server_terminal"] = server.poll() is not None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=("native-gpu", "native-hybrid", "comfy-default", "comfy-sdpa-gpu"), required=True)
    parser.add_argument("--ane-manifest", type=Path)
    parser.add_argument("--residency", choices=("resident", "component_staged"), default="resident")
    parser.add_argument("--vae-options",type=json.loads,help="explicit native request-local seedvr2_vae object")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--kind", choices=("image", "video"), required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, default=ROOT / "build/native/libturbocider.dylib")
    parser.add_argument("--native-build-manifest",type=Path,help="explicit matching build identity for a relocated/packaged native library")
    parser.add_argument("--experimental-causal-manifest",type=Path,help="isolated full-request VAE experiment; not a public App setting")
    parser.add_argument("--experimental-causal-compute",choices=("ne","cpu"),default="ne")
    parser.add_argument("--capture-raw-video",action="store_true",help="experimental build only: bounded preencoder pixels; NOT a timing/peak benchmark")
    parser.add_argument("--comfy", type=Path)
    parser.add_argument("--comfy-python", type=Path)
    parser.add_argument("--comfy-extra", type=Path)
    args = parser.parse_args()
    if args.experimental_causal_manifest:
        if args.route!="native-gpu":parser.error("causal experiment requires an explicit GPU backbone observer")
        args.experimental_causal_manifest=args.experimental_causal_manifest.resolve(strict=True)
        os.environ["TURBOCIDER_SEEDVR2_VAE_CAUSAL_MANIFEST"]=str(args.experimental_causal_manifest)
        os.environ["TURBOCIDER_SEEDVR2_VAE_CAUSAL_COMPUTE"]=args.experimental_causal_compute
    elif os.environ.get("TURBOCIDER_SEEDVR2_VAE_CAUSAL_MANIFEST"):
        parser.error("experimental causal environment must be explicitly recorded by --experimental-causal-manifest")
    if args.native_build_manifest is not None:
        if not args.route.startswith("native-"):parser.error("native build manifest applies only to native observations")
        args.native_build_manifest=args.native_build_manifest.resolve(strict=True)
    if args.vae_options is not None and (not isinstance(args.vae_options,dict) or not args.route.startswith("native-")):
        parser.error("VAE controls require a JSON object and a native route")
    if args.route == "native-hybrid":
        if args.ane_manifest is None or not args.ane_manifest.is_file():
            parser.error("hybrid observation requires an existing explicit prefix manifest")
        args.ane_manifest = args.ane_manifest.absolute()
    elif args.ane_manifest is not None:
        parser.error("prefix manifest applies only to native-hybrid observation")
    if args.route.startswith("comfy-") and args.residency != "resident":
        parser.error("reference residency must remain the explicitly recorded ordinary setting")
    for name in ("model", "source", "output", "library"):
        setattr(args, name, getattr(args, name).absolute())
    for name in ("comfy", "comfy_python", "comfy_extra"):
        if getattr(args, name) is not None: setattr(args, name, getattr(args, name).absolute())
    if args.route.startswith("comfy-") and any(getattr(args, name) is None for name in ("comfy", "comfy_python", "comfy_extra")):
        parser.error("explicit existing ComfyUI checkout, Python and extra dependency paths are required")
    if not 1 <= args.count <= 10 or args.width % 16 or args.height % 16:
        parser.error("count must be 1...10; output dimensions must be multiples of sixteen")
    configure_raw_capture(args)
    report = record_setup(args)
    if args.capture_raw_video:
        report["diagnostic_raw_capture"]=True
        report["timing_scope"]="complete streaming request WITH frame/tensor diagnostic IO; not a speed/peak qualification"
        report["native_artifact_identity"]["raw_capture"]="bounded-fp16-and-preencoder-rgb24-v1"
    try:
        if args.route in ("native-gpu", "native-hybrid"): native(args, report)
        else: asyncio.run(comfy(args, report))
        report["status"] = "complete"
        if len(report["runs"]) > 1:
            report["warm_median_seconds"] = statistics.median(row["request_wall_including_export"] for row in report["runs"][1:])
    except BaseException as error:
        report.update(status="failed", failure=str(error))
        raise
    finally:
        write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
