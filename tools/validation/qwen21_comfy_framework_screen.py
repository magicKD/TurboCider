#!/usr/bin/env python3
"""Same-source Qwen2.1 native GPU/frozen and owned ComfyUI MPS presets."""
import argparse
import json
import math
from pathlib import Path
import socket
import shutil
import statistics
import subprocess
import sys

import numpy as np

from flux9_bf16_framework_screen import PROMPTS, generation, check_generations, registered_hashes, write_json
from image_framework_native import generate_requests
from qwen21_mflux_framework_screen import requests as native_requests
from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_sampled
from z_comfy_framework_screen import comfy_worker, launch_command as z_launch

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/Comfy-Org-Qwen-Image-2.1"
FROZEN = ROOT / "outputs/benchmark-best-configs/qwen21-frozen-w6144-r1024-cache-20261011/manifest-99f1b3499b89b96f5d04a23cb91c3f3496083e1d66b09c6dd2e4500d0d6ca509.json"
ROUTES = ("native-gpu", "native-frozen", "comfy-default", "comfy-gpu", "comfy-sdpa-gpu")
ORDER = (*ROUTES, *reversed(ROUTES))
STEPS = 40


def sigmas():
    # Native scheduler.hpp and upstream base scheduler both stretch the last
    # nonzero node to .02. ManualSigmas avoids the generic Comfy simple table.
    mu = np.float32((.9 - .5) / (8192 - 256) * (512 * 512 / 256 - 256) + .5)
    initial = np.linspace(1, 1 / STEPS, STEPS, dtype=np.float32)
    e = np.exp(mu)
    shifted = e / (e + (np.float32(1) / initial - np.float32(1)))
    complement = np.float32(1) - shifted
    result = np.float32(1) - complement / (complement[-1] / np.float32(.98))
    return [*map(float, result), 0.]


def workflow(prompt, index):
    def node(kind, **inputs):
        return dict(class_type=kind, inputs=inputs)
    return {
        "1": node("UNETLoader", unet_name="qwen_image_2.1_bf16.safetensors", weight_dtype="default"),
        "2": node("CLIPLoader", clip_name="qwen3vl_8b_bf16.safetensors", type="qwen_image", device="default"),
        "3": node("VAELoader", vae_name="qwen_image_2.1_vae_bf16.safetensors"),
        "4": node("CLIPTextEncode", clip=["2", 0], text=prompt),
        "5": node("ConditioningZeroOut", conditioning=["4", 0]),
        # Stock sampling fixes empty channels/downscale to the model format.
        "6": node("EmptySD3LatentImage", width=512, height=512, batch_size=1),
        "7": node("QwenImage21Cache", model=["1", 0], device="auto", dtype="default"),
        "8": node("SamplerCustom", model=["7", 0], positive=["4", 0], negative=["5", 0],
                  latent_image=["6", 0], add_noise=True, noise_seed=30 if index == 3 else 29,
                  cfg=1.0, sampler=["11", 0], sigmas=["12", 0]),
        "9": node("VAEDecode", samples=["8", 0], vae=["3", 0]),
        "10": node("SaveImage", images=["9", 0], filename_prefix=f"request-{index}"),
        "11": node("KSamplerSelect", sampler_name="euler"),
        "12": node("ManualSigmas", sigmas=" ".join(format(v, ".9g") for v in sigmas())),
    }


def validate_comfy_row(row, index):
    wall = row.get("request_wall_including_export")
    if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
        raise ValueError("finite complete Qwen PNG wall required")
    if (row.get("index"), row.get("seed"), row.get("history_status")) != (index, 30 if index == 3 else 29, "success"):
        raise ValueError("actual Qwen sequence/seed/history differs")
    if row.get("actual_steps") != list(range(1, STEPS + 1)):
        raise ValueError("all40 sampler steps required, not cached output")
    if any(n not in row.get("executed_nodes", []) or n in row.get("cached_nodes", []) for n in ("8", "9", "10")):
        raise ValueError("actual sampler/VAE/SaveImage must execute")
    if index != 3 and ("4" not in row["executed_nodes"] or "4" in row["cached_nodes"]):
        raise ValueError("fresh Qwen prompt must really encode")


def validate_trial(trial, count=4):
    report, route = trial["report"], trial["route"]
    if route not in ROUTES or len(report["runs"]) != count or trial["memory"]["verified"]["complete"] is not True:
        raise ValueError("known complete Qwen trial and independent memory required")
    if route.startswith("comfy-"):
        if report.get("status") != "complete" or report.get("server_terminal") is not True:
            raise ValueError("owned ComfyUI must complete and exit")
        for i, row in enumerate(report["runs"]):
            validate_comfy_row(row, i)
    else:
        for i, row in enumerate(report["runs"]):
            m = row["metrics"]
            if (m["model"], m["operation"], m["width"], m["height"], m["steps"], m["seed"], m["actual_denoise_steps"],
                    m["prompt_cache_hit"]) != ("qwen-image-2.1", "image.generate", 512, 512, STEPS, 30 if i == 3 else 29, STEPS, i == 3):
                raise ValueError("actual full native Qwen task/cache differs")
            wall = row["request_wall_including_export"]
            if type(wall) not in (float, int) or not math.isfinite(wall) or wall <= 0:
                raise ValueError("finite positive complete native wall required")
            h = m.get("hybrid")
            if route == "native-gpu" and h:
                raise ValueError("pure GPU cannot execute hybrid work")
            if route == "native-frozen" and (not h or h.get("runtime_calls_session_total") != (i + 1) * 39 * 32 or
                    h.get("checkpoint_sha256_verified") is not True or h.get("runtime_failed") is not False or
                    h.get("runtime_failures_session_total") != 0 or h.get("runtime_weight") is not None or
                    h.get("ane_mlp_range") != [0, 6144] or h.get("compute_units") != "cpuAndNeuralEngine"):
                raise ValueError("actual source-bound frozen Qwen FFN calls required")


def launch_command(comfy, python, output, route, port):
    if route not in ROUTES[2:]:
        raise ValueError("declared Qwen Comfy preset required")
    command = z_launch(comfy, python, output, "comfy-sdpa-gpu" if route == "comfy-gpu" else route, port)
    if route == "comfy-gpu":
        command.remove("--use-pytorch-cross-attention")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=ROUTES)
    parser.add_argument("--comfy-extra", type=Path)
    for name in ("comfy", "comfy-python", "native-python", "library"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--count", type=int, choices=(1, 4), default=4)
    parser.add_argument("--reference-only", action="store_true", help="reuse published native bests; execute only ComfyUI presets")
    parser.add_argument("--order", default=",".join(ORDER))
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker:
        if args.worker.startswith("comfy-"):
            comfy_worker(args.output, graph_factory=workflow, prompts=PROMPTS, validate_row=validate_comfy_row, seed_key="noise_seed")
        else:
            rows = native_requests(args.output, STEPS, args.count, args.worker, FROZEN)
            if args.count == 4:
                rows[3]["seed"] = 30
            report = generate_requests(args.library, "qwen-image-2.1", MODEL, rows, args.output, args.worker)
            write_json(args.output / "report.json", report)
        return
    if args.output.exists() or any(getattr(args, n) is None for n in ("comfy", "comfy_python", "native_python", "library", "comfy_extra")):
        parser.error("explicit isolated existing ComfyUI/packages/native library and fresh output required")
    args.comfy, args.library, args.comfy_extra = (p.resolve(strict=True) for p in (args.comfy, args.library, args.comfy_extra))
    args.comfy_python, args.native_python = args.comfy_python.absolute(), args.native_python.absolute()
    order = args.order.split(",")
    if args.reference_only and order == list(ORDER):
        order = list(ROUTES[2:])
    if (any(r not in ROUTES for r in order) or args.reference_only and order != list(ROUTES[2:]) or
            not args.reference_only and args.count == 4 and order != list(ORDER)):
        parser.error("declared precheck or complete five-route palindrome required")
    build_path = args.library.parent / "runtime-build/runtime-build-manifest.json"
    build = json.loads(build_path.read_text())
    if any(sha256_file(ROOT / r["path"]) != r["sha256"] for r in build["inputs"]["sources"]):
        raise ValueError("native library source inputs differ")
    sources = [MODEL / p for p in ("diffusion_models/qwen_image_2.1_bf16.safetensors",
        "text_encoders/qwen3vl_8b_bf16.safetensors", "vae/qwen_image_2.1_vae_bf16.safetensors")]
    tracked = subprocess.check_output(["git", "-C", str(args.comfy), "ls-files", "*.py", "comfy/text_encoders/qwen25_tokenizer/*"], text=True).splitlines()
    code = [args.comfy / p for p in tracked if (args.comfy / p).is_file()]
    extras = [p for p in args.comfy_extra.rglob("*") if p.is_file() and p.stat().st_size > 0 and p.suffix not in (".pyc", ".lock")]
    ane = [p for p in FROZEN.parent.rglob("*") if p.is_file() and p.suffix != ".lock"]
    own = [ROOT / "tools/validation" / n for n in ("z_comfy_framework_screen.py", "qwen21_mflux_framework_screen.py",
        "image_framework_native.py", "runtime_ane_memory.py", "runtime_ane_common.py", "flux9_bf16_framework_screen.py")]
    files = sorted(set([*sources, *code, *extras, *ane, *own, Path(__file__).resolve(), args.library, build_path]))
    identities = {str(p): generation(p) for p in files}
    hashes = registered_hashes(files, ROOT / "outputs/benchmark-source-registrations/unsloth-image.json")
    args.output.mkdir(parents=True)
    data = dict(schema="tc-qwen21-comfy-reference-screen-v1" if args.reference_only else "tc-qwen21-comfy-five-route-screen-v1",
        date="2026-10-12", status="running", steps=STEPS, count=args.count, reference_only=args.reference_only,
        order=order, trials=[], qualification_passed=False, library_sha256=hashes[str(args.library)],
        native_inputs_matched=len(build["inputs"]["sources"]), source_sha256=hashes, model_sha256={str(p): hashes[str(p)] for p in sources},
        comfy_commit=subprocess.check_output(["git", "-C", str(args.comfy), "rev-parse", "HEAD"], text=True).strip(),
        comfy_local_changes=subprocess.check_output(["git", "-C", str(args.comfy), "status", "--short"], text=True).splitlines(),
        schedule=dict(sigmas=sigmas(), rule="40Euler,mu(.5-.9 at256-8192),terminal.02 then0", cfg=1, seed=29, repeat_seed=30),
        timing_scope="full512 PNG: native C API/export or owned ComfyUI HTTP submit/WebSocket success/history/download/export; readiness/cold/repeat separate",
        representation_scope="same original BF16 encoder/DiT/VAE; ManualSigmas40Euler/cfg1,default lossless auto prefix KV cache; stock vs GPU residency vs SDPA/GPU; frozen W8A8 approximate; independent RNG/token/arithmetic/cache lifecycles",
        source_policy="initial source-generation SHA reused by metadata, no per-inference model hash; reference/user service/models unchanged; custom/API nodes disabled; repeat seed30 forces actual40steps/VAE")
    target = args.output / "summary.json"
    write_json(target, data)
    try:
        for index, route in enumerate(order):
            check_generations(identities)
            if shutil.disk_usage(ROOT).free < 2 << 30:
                raise RuntimeError("2GiB disk reserve required before the next owned process")
            output = args.output / f"{index}-{route}"
            output.mkdir()
            env = benchmark_environment()
            env.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
            env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
            python = args.comfy_python if route.startswith("comfy-") else args.native_python
            roles = ()
            if route.startswith("comfy-"):
                env["PYTHONPATH"] = str(args.comfy_extra)
                (output / "instance/user").mkdir(parents=True)
                import yaml
                (output / "models.yaml").write_text(yaml.safe_dump(dict(turbocider_benchmark=dict(
                    base_path=str(MODEL), diffusion_models="diffusion_models", text_encoders="text_encoders", vae="vae"))))
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                write_json(output / "launch.json", dict(command=launch_command(args.comfy, python, output, route, port),
                    port=port, comfy=str(args.comfy), count=args.count))
                roles = (("comfy_server", str(python.resolve(strict=True))), ("listener_probe", "/usr/sbin/lsof"))
            command = [str(python), "-B", str(Path(__file__).resolve()), "--worker", route, "--output", str(output),
                       "--library", str(args.library), "--count", str(args.count)]
            print(json.dumps(dict(starting=index, route=route)), flush=True)
            with (output / "stdout.txt").open("x") as out, (output / "stderr.txt").open("x") as err:
                memory = run_sampled(command, repo=ROOT, output=output, stem="run", env=env, stdout=out, stderr=err,
                    timeout=1800, interval_ms=100, max_gap_ms=500, async_evidence=True, child_roles=roles)
            check_generations(identities)
            trial = dict(index=index, route=route, report=json.loads((output / "report.json").read_text()), memory=memory, png_sha256=[])
            validate_trial(trial, args.count)
            from PIL import Image
            for i in range(args.count):
                path = output / f"{i}.png"
                with Image.open(path) as image:
                    image.load()
                    if image.size != (512, 512):
                        raise ValueError("complete full-size PNG required")
                trial["png_sha256"].append(sha256_file(path))
            data["trials"].append(trial)
            write_json(target, data)
        if args.count == 4:
            data["medians"] = {}
            for route in dict.fromkeys(order):
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
