#!/usr/bin/env python3
"""Serial source-matched Z BF16 GPU/frozen/mflux/Unsloth CPU+GPU-text palindrome."""
import argparse
import json
import math
from pathlib import Path
import socket
import statistics
import subprocess
import sys

from flux9_bf16_framework_screen import generation,check_generations,registered_hashes,write_json
from mflux_original_layout import validate_layout
from runtime_ane_common import benchmark_environment,sha256_file
from runtime_ane_memory import run_sampled
from unsloth_image_reference_screen import launch_command,sampler_roles,validate_generation,placement
from z_base_formats_screen import CHECKPOINTS,COMMON,FROZEN
from z_mflux_framework_screen import route_environment,validate_runs

ROOT = Path(__file__).resolve().parents[2]
ROUTES = ("native-gpu","native-frozen","mflux-bf16","unsloth-gpu-text","unsloth-cpu-text")
ORDER = (*ROUTES,*reversed(ROUTES))


def compiled_source_files(root):
    # Cache lock files have no model payload and may be touched by the
    # compiler. Keep every actual compiled artifact and manifest bound.
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix != ".lock")


def validate_trial(trial):
    route,report = trial["route"],trial["report"]
    if route not in ROUTES or len(report.get("runs",[])) != 4 or trial["memory"]["verified"]["complete"] is not True:
        raise ValueError("complete known four-PNG trial and independent memory required")
    if route.startswith("unsloth-"):
        if report.get("server_terminal") is not True or report.get("status") != "complete_precheck_only":
            raise ValueError("successful owned reference server must be terminal")
        for index,row in enumerate(report["runs"]):
            if row["index"] != index or row.get("prompt_is_repeat") is not (index==3):
                raise ValueError("actual reference cold/two-fresh/repeat sequence differs")
            wall = row.get("request_wall_including_export")
            if type(wall) not in (float,int) or not math.isfinite(wall) or wall<=0:
                raise ValueError("finite positive complete reference PNG wall required")
    else:
        validate_runs(report,route,8,4)
        if route == "native-frozen":
            for index,row in enumerate(report["runs"]):
                hybrid = row["metrics"]["hybrid"]
                if (hybrid.get("runtime_calls_session_total") != 256*(index+1) or
                        hybrid.get("checkpoint_sha256_verified") is not True or
                        hybrid.get("ane_mlp_range") != [0,5120] or hybrid.get("mlp_width") != 10240 or
                        hybrid.get("compute_units") != "cpuAndNeuralEngine" or hybrid.get("block_count") != 32 or
                        hybrid.get("warmup_calls_session_total") != 0):
                    raise ValueError("all256 actual source-bound frozen FFN calls/request required")
        if route == "mflux-bf16":
            if report.get("reference_cache_limit_bytes") != 1024<<20:
                raise ValueError("actual stock current reference cache differs")
            coverage = report.get("required_weight_coverage",{})
            if set(coverage) != {"transformer","text_encoder","vae"} or any(
                    value.get("required_targets",0)<=0 or value.get("covered_targets") != value["required_targets"] or value.get("missing") != []
                    for value in coverage.values()):
                raise ValueError("complete current mflux source coverage required")


def validate_cohort(data):
    if data.get("status") != "complete_diagnostic" or data.get("order") != list(ORDER) or len(data.get("trials",[])) != len(ORDER):
        raise ValueError("complete full five-route palindrome required")
    for index,trial in enumerate(data["trials"]):
        if trial["index"] != index or trial["route"] != ORDER[index]:
            raise ValueError("actual full route order differs")
        validate_trial(trial)
    for route in ROUTES:
        trials = [trial for trial in data["trials"] if trial["route"] == route]
        values = [row["request_wall_including_export"] for trial in trials for row in trial["report"]["runs"][1:3]]
        median = data["medians"][route]
        if median["samples"] != values or median["fresh_warm_seconds"] != statistics.median(values):
            raise ValueError("all four actual fresh samples must determine the median")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference","binary","library","mflux","reference-model","python","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh paired cohort required")
    for name in ("reference","binary","library","mflux","reference_model"):
        setattr(args,name,getattr(args,name).resolve(strict=True))
    args.output = args.output.resolve()
    args.python = args.python.absolute()
    layout = validate_layout(args.reference_model)
    if layout["family"] != "z":
        raise ValueError("verified unchanged BF16 Z mflux view required")
    build_path = args.library.parent / "runtime-build/runtime-build-manifest.json"
    build = json.loads(build_path.read_text())
    if any(sha256_file(ROOT / row["path"]) != row["sha256"] for row in build["inputs"]["sources"]):
        raise ValueError("current native source must match exact library build inputs")
    dit,encoder,vae = CHECKPOINTS["bf16"],COMMON / "split_files/text_encoders/qwen_3_4b.safetensors",COMMON / "split_files/vae/ae.safetensors"
    sources = sorted(set([dit,encoder,vae,*[p for p in (COMMON / "tokenizer").glob("*") if p.is_file()]]))
    source_hashes = registered_hashes(sources,ROOT / "outputs/benchmark-source-registrations/unsloth-image.json")
    reference_files = [args.binary,args.reference / "studio/backend/core/inference/sd_cpp_args.py",
        args.reference / "studio/backend/core/inference/diffusion_memory.py",args.binary.parent / "UNSLOTH_BUILD.txt"]
    tools = [Path(__file__).resolve(),ROOT / "tools/validation/unsloth_image_reference_screen.py",ROOT / "tools/validation/z_mflux_framework_screen.py",
        ROOT / "tools/validation/image_framework_native.py",args.library,build_path,FROZEN]
    mflux_files = sorted((args.mflux / "src").rglob("*.py"))
    ane_files = compiled_source_files(FROZEN.parent)
    derived_files = sorted(p for p in args.reference_model.rglob("*") if p.is_file() and p.name != "benchmark-source-registrations.json")
    all_files = sorted(set([*sources,*reference_files,*tools,*mflux_files,*ane_files,*derived_files]))
    identities = {str(p):generation(p) for p in all_files}
    hashes = registered_hashes(all_files,ROOT / "outputs/benchmark-source-registrations/unsloth-image.json")
    args.output.mkdir(parents=True)
    data = dict(schema="tc-z-unsloth-five-route-screen-v1",status="running",qualification_passed=False,goal_complete=False,
        date="2026-10-12",steps=8,count=4,order=list(ORDER),trials=[],library_sha256=hashes[str(args.library)],
        native_inputs_matched=len(build["inputs"]["sources"]),model_sha256={str(p):source_hashes[str(p)] for p in sources},
        source_sha256=hashes,layout_receipt_sha256=sha256_file(args.reference_model / "layout-receipt.json"),
        unsloth_commit=subprocess.check_output(["git","-C",str(args.reference),"rev-parse","HEAD"],text=True).strip(),
        unsloth_backend_build=(args.binary.parent / "UNSLOTH_BUILD.txt").read_text(),
        mflux_commit=subprocess.check_output(["git","-C",str(args.mflux),"rev-parse","HEAD"],text=True).strip(),
        timing_scope="complete512 PNG wall: native C API/export, mflux image/save, sd-server HTTP/PNG reply/export; constructor/readiness/cold/repeat separate",
        representation_scope="same original BF16 encoder/DiT/VAE; native frozen W8A8 is approximate; independent runtimes/noise engines/templates are not pixel-identical; defaultCPU text and opt-in Metal text both retained",
        source_policy="initial source-generation SHA then durable metadata reuse; no model hashes in individual inference; original assets/reference source unchanged")
    target = args.output / "summary.json"
    write_json(target,data)
    try:
        for index,route in enumerate(ORDER):
            check_generations(identities)
            output = args.output / f"{index}-{route}"
            output.mkdir()
            if route.startswith("unsloth-"):
                text = "gpu" if route == "unsloth-gpu-text" else "default"
                with socket.socket() as reservation:
                    reservation.bind(("127.0.0.1",0))
                    port = reservation.getsockname()[1]
                launch = launch_command(args.reference,args.binary,dit,encoder,vae,"z-image-turbo",text,port)
                write_json(output / "launch.json",dict(command=launch,port=port,model="z-image-turbo",count=4,encoder_placement=text))
                command = [str(args.python),"-B",str(ROOT / "tools/validation/unsloth_image_reference_screen.py"),"--worker","--output",str(output)]
                env = benchmark_environment()
                roles = sampler_roles(args.binary)
            else:
                command = [str(args.python),"-B",str(ROOT / "tools/validation/z_mflux_framework_screen.py"),"--child",route,
                    "--reference-model",str(args.reference_model),"--library",str(args.library),"--mflux",str(args.mflux),
                    "--output",str(output),"--steps","8","--count","4","--reference-cache-mib","1024"]
                env = route_environment(route)
                env.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",PYTHONDONTWRITEBYTECODE="1",MPLCONFIGDIR=str(args.output / "matplotlib-cache"))
                roles = (("python_helper",str(args.python.resolve(strict=True))),) if route.startswith("mflux-") else ()
            print(json.dumps(dict(starting=index,route=route)),flush=True)
            with (output / "stdout.txt").open("x") as out,(output / "stderr.txt").open("x") as err:
                memory = run_sampled(command,repo=ROOT,output=output,stem="run",env=env,stdout=out,stderr=err,
                    timeout=1200,interval_ms=100,max_gap_ms=500,async_evidence=True,child_roles=roles)
            check_generations(identities)
            report = json.loads((output / "report.json").read_text())
            trial = dict(index=index,route=route,report=report,memory=memory)
            validate_trial(trial)
            if route.startswith("unsloth-"):
                placement((output / "server.stdout.txt").read_text()+(output / "server.stderr.txt").read_text(),text)
                for request_index,row in enumerate(report["runs"]):
                    payload = json.loads((output / f"request-{request_index}.json").read_text())
                    validate_generation(row["info"],payload,(output / f"request-{request_index}.log").read_text())
            from PIL import Image
            trial["png_sha256"] = []
            for request_index in range(4):
                path = output / f"{request_index}.png"
                with Image.open(path) as image:
                    image.load()
                    if image.size != (512,512):
                        raise ValueError("full size PNG required")
                trial["png_sha256"].append(sha256_file(path))
            data["trials"].append(trial)
            write_json(target,data)
        data["medians"] = {}
        for route in ROUTES:
            trials = [trial for trial in data["trials"] if trial["route"] == route]
            samples = [row["request_wall_including_export"] for trial in trials for row in trial["report"]["runs"][1:3]]
            data["medians"][route] = dict(samples=samples,fresh_warm_seconds=statistics.median(samples),
                cold_seconds=[trial["report"]["runs"][0]["request_wall_including_export"] for trial in trials],
                repeat_seconds=[trial["report"]["runs"][3]["request_wall_including_export"] for trial in trials],
                constructor_seconds=[trial["report"].get("ready_seconds",trial["report"].get("constructor_seconds")) for trial in trials])
        data["status"] = "complete_diagnostic"
        validate_cohort(data)
    except BaseException as error:
        data.update(status="failed",error=str(error))
        raise
    finally:
        write_json(target,data)
    print(json.dumps(dict(status=data["status"],medians=data["medians"])))


if __name__ == "__main__":
    main()
