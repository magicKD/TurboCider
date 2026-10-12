#!/usr/bin/env python3
"""Serial same-payload DiT Q4/g32 versus original BF16, native and stock mflux."""
import argparse
import copy
import json
from pathlib import Path
import statistics
import subprocess
import sys

from flux9_bf16_framework_screen import (
    check_generations, generation, native_child, mflux_child, registered_hashes, requests, validate_native, write_json)
from mflux_original_layout import read_header
from mflux_weight_coverage import inspect_saved_dit_coverage, inspect_saved_full_coverage
from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_memory import run_sampled

ROOT = Path(__file__).resolve().parents[2]
ORDER = ("native-bf16","native-q4","mflux-bf16","mflux-q4",
         "mflux-q4","mflux-bf16","native-q4","native-bf16")
PREDECODE_ORDER = ("native-bf16","native-q4","native-predecoded","mflux-bf16","mflux-q4",
                   "mflux-q4","mflux-bf16","native-predecoded","native-q4","native-bf16")


def quantized_route(route):
    return route in ("native-q4","native-predecoded","mflux-q4")


def validate_pack(pack, original):
    receipt = json.loads((pack / "conversion.json").read_text())
    if (receipt.get("status") != "complete_conversion_only" or receipt.get("bits") != 4 or
            receipt.get("group_size") != 32 or receipt.get("converted_components") not in (["transformer"],["transformer","text_encoder"])):
        raise ValueError("complete known matched Q4/g32 conversion required")
    components = receipt["converted_components"]
    for name in components:
        record = receipt["components"][name]
        tensors,linears = (539,153) if name == "transformer" else (902,252)
        if (record.get("full_compressed_payload_equal") is not True or record.get("stored_tensors") != tensors or
                record.get("quantized_linears") != linears):
            raise ValueError("complete original9B learned weights plus all affine planes required")
        if (pack / "native" / name / "config.json").resolve(strict=True) != (original / name / "config.json").resolve(strict=True):
            raise ValueError("actual native geometry must retain the original9B config")
    for layout in ("native","mflux"):
        for component in [name for name in ("text_encoder","vae","tokenizer") if name not in components]:
            if (pack / layout / component).resolve(strict=True) != (original / component).resolve(strict=True):
                raise ValueError("original dense encoder/VAE/tokenizer must be shared unchanged")
        for name in components:
            path = pack / layout / name / "0.safetensors"
            record = receipt["components"][name]
            tensors,linears = (539,153) if name == "transformer" else (902,252)
            if path.stat().st_size != record["bytes"]:
                raise ValueError("complete compressed source extent required")
            header,_ = read_header(path)
            if len(header)-1 != tensors or sum(key.endswith(".scales") for key in header) != linears:
                raise ValueError("complete quantized/native source tensor descriptors required")
            for key, value in header.items():
                if key.endswith(".scales"):
                    prefix = key[:-7]
                    weight, bias = header[prefix+".weight"],header[prefix+".biases"]
                    if (weight["dtype"] != "U32" or value["dtype"] != "BF16" or bias["dtype"] != "BF16" or
                            len(weight["shape"]) != 2 or value["shape"] != bias["shape"] or
                            weight["shape"][0] != value["shape"][0] or weight["shape"][1] != 4*value["shape"][1]):
                        raise ValueError("exact saved Q4/g32 packed/scales/affine-bias planes required")
    return receipt


def validate_report(report, route):
    if route not in PREDECODE_ORDER or report.get("engine") != route or len(report.get("runs",[])) != 4:
        raise ValueError("actual complete known four-request route required")
    if route.startswith("native"):
        validate_native(report["runs"])
        count = 96 if route in ("native-bf16","native-predecoded") else 0
        if any(row.get("actual_compiled_single_blocks") != count for row in report["runs"]):
            raise ValueError("actual dense compiled versus packed eager branch differs")
        if any(row.get("native_event_counts",{}).get("transformer_block") != 128 for row in report["runs"]):
            raise ValueError("all8 dual plus24 single blocks across4 steps must execute")
        if route == "native-predecoded" and any(
                row["native_event_counts"].get("affine_predecoded_gpu_route") != 1 or
                row["native_event_counts"].get("affine_predecode_projection",0) != (153 if index == 0 else 0)
                for index,row in enumerate(report["runs"])):
            raise ValueError("one complete153-projection BF16 predecode then retained compiled route required")
    else:
        bits = 4 if route == "mflux-q4" else None
        if report.get("quantization") != bits or report.get("precision") != "mlx.core.bfloat16":
            raise ValueError("actual stored representation/activation precision differs")
        coverage = report.get("required_weight_coverage",{})
        if set(coverage) != {"vae","transformer","text_encoder"} or any(
                row.get("required_targets",0) <= 0 or row.get("covered_targets") != row.get("required_targets") or
                row.get("missing") != [] for row in coverage.values()):
            raise ValueError("actual complete loaded-model/source coverage required")
        if route == "mflux-q4" and (coverage["transformer"].get("bits"),coverage["transformer"].get("group_size"),
                                      coverage["transformer"].get("quantized_linears")) != (4,32,153):
            raise ValueError("actual153 DiT Q4/g32 linears required")
        if route == "mflux-q4" and report.get("quantized_components") == ["transformer","text_encoder"] and (
                coverage["text_encoder"].get("bits"),coverage["text_encoder"].get("group_size"),
                coverage["text_encoder"].get("quantized_linears"),coverage["text_encoder"].get("required_targets")) != (4,32,252,902):
            raise ValueError("all252 actual encoder Q4/g32 linears and902 saved parameters required")
        if any(row.get("actual_steps") != [0,1,2,3] or row.get("actual_quantization") != bits or row.get("loras")
               for row in report["runs"]):
            raise ValueError("actual base four-step generation required")


def child(args):
    route = args.child
    receipt = validate_pack(args.pack,args.model)
    full = receipt["converted_components"] == ["transformer","text_encoder"]
    output = args.output
    call = copy.copy(args)
    call.native_encoder_eval_interval = 0
    call.native_encoder_fused_attention = True
    call.reference_cache_mib = 0
    call.stage_probe = False
    call.ane_manifest = None
    call.model = args.pack / ("native" if route.startswith("native") else "mflux") if quantized_route(route) else args.model
    if route.startswith("native"):
        call.child = "native-compiled" if route in ("native-bf16","native-predecoded") else "native"

        def factory(destination, compiled, manifest):
            result = requests(destination,compiled,manifest)
            if quantized_route(route):
                for request in result:
                    request["allow_approximation"] = True
            return result

        report = native_child(call,output,request_factory=factory)
    else:
        report = mflux_child(call,output,stored_bits=4 if route.endswith("q4") else None,
                            coverage_provider=(inspect_saved_full_coverage if full else inspect_saved_dit_coverage) if route.endswith("q4") else None)
    report.update(engine=route,model_root=str(call.model),
        quantized_components=receipt["converted_components"] if quantized_route(route) else [],
        representation=("DiT153+encoder252 affine Q4/g32 linears; original dense BF16 embedding/VAE" if full else
                        "DiT153 affine Q4/g32 linears; original dense BF16 encoder/VAE") if quantized_route(route) else "original BF16 encoder/DiT/VAE",
        dit_runtime_recipe="source Q4/g32 decoded once to BF16,153 dense projections then96 compiled singles/request" if route == "native-predecoded" else "original packed/dense route",
        library_sha256=sha256_file(args.library) if route.startswith("native") else None)
    validate_report(report,route)
    write_json(output / "report.json",report)


def parent(args):
    from PIL import Image
    receipt = validate_pack(args.pack,args.model)
    args.output.mkdir(parents=True)
    models = sorted([*args.model.rglob("*.safetensors"),*args.model.rglob("*.json"),*args.model.rglob("*.jinja"),
        *(args.pack / layout / name / "0.safetensors" for layout in ("native","mflux") for name in receipt["converted_components"]),args.pack / "conversion.json"])
    reference = sorted((args.mflux / "src").rglob("*.py"))
    local = [args.library,Path(__file__).resolve(),ROOT / "tools/validation/flux9_affine_pack.py",
        ROOT / "tools/validation/flux9_bf16_framework_screen.py",ROOT / "tools/validation/mflux_weight_coverage.py",
        ROOT / "tools/validation/mflux_original_layout.py"]
    identities = {str(path):generation(path) for path in (*models,*reference,*local)}
    hashes = registered_hashes(models,ROOT / "outputs/benchmark-source-registrations/flux9-bf16.json")
    hashes.update({str(path):sha256_file(path) for path in (*reference,*local)})
    for layout in ("native","mflux"):
        for name in receipt["converted_components"]:
            path = args.pack / layout / name / "0.safetensors"
            if hashes[str(path)] != receipt["components"][name][layout+"_sha256"]:
                raise ValueError("actual converted artifact differs from whole-payload equality proof")
    for name,digest in receipt["source_sha256"].items():
        if hashes.get(name) != digest:
            raise ValueError("actual original source differs from conversion provenance")
    data = dict(schema="tc-flux9-matched-affine-screen-v1",status="running",qualification_passed=False,goal_complete=False,
        date="2026-10-12",order=args.order.split(","),trials=[],conversion=receipt,
        conversion_sha256=hashes[str(args.pack / "conversion.json")],
        model_sha256={str(path.relative_to(ROOT)):hashes[str(path)] for path in models},
        local_source_sha256={str(path.relative_to(ROOT)):hashes[str(path)] for path in local},
        mflux_source_sha256={str(path.relative_to(args.mflux)):hashes[str(path)] for path in reference},
        mflux_commit=subprocess.check_output(["git","-C",str(args.mflux),"rev-parse","HEAD"],text=True).strip(),
        task=dict(model="flux2-klein-9b",size=512,steps=4,seed=42,dynamic_text=False,loras=[]),
        timing_scope="complete PNG wall; separate constructor/cold, two fresh prompts, immediate repeated-prompt cache hit separately",
        source_policy="first source-generation SHA then durable metadata reuse; stat guards at arm boundaries; no per-inference model hash",
        scope="same original base; both Q4 routes share byte-identical g32 compressed payload in declared converted components; original BF16 embedding/VAE; not stock all-module affine or GGUF; no ANE route for this quantized candidate")
    target = args.output / "summary.json"
    write_json(target,data)
    try:
        for index,route in enumerate(data["order"]):
            check_generations(identities)
            destination = args.output / f"{index}-{route}"
            destination.mkdir()
            env = benchmark_environment()
            for key in tuple(env):
                if key.startswith(("TURBOCIDER_FLUX_","TURBOCIDER_FLUX9_","TURBOCIDER_QWEN3_")):
                    del env[key]
            env.update(PYTHONPATH=str(args.extra),PYTHONDONTWRITEBYTECODE="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",
                MPLCONFIGDIR=str(args.output / "matplotlib-cache"))
            if route.startswith("native"):
                env.update(TURBOCIDER_QWEN3_EVAL_INTERVAL="0",TURBOCIDER_QWEN3_FUSED_SDPA="1")
            if route == "native-predecoded":
                env["TURBOCIDER_FLUX9_PREDECODE_AFFINE"] = "1"
            command = [str(args.python),"-B",str(Path(__file__).resolve()),"--model",str(args.model),"--pack",str(args.pack),
                "--library",str(args.library),"--mflux",str(args.mflux),"--output",str(destination),"--child",route]
            print(json.dumps(dict(starting=index,route=route)),flush=True)
            with (destination / "stdout.txt").open("x") as out,(destination / "stderr.txt").open("x") as err:
                memory = run_sampled(command,repo=ROOT,output=destination,stem="run",env=env,stdout=out,stderr=err,
                    timeout=900,interval_ms=100,max_gap_ms=500,async_evidence=True,
                    child_roles=(("python_helper",str(args.python.resolve(strict=True))),) if route.startswith("mflux") else ())
            check_generations(identities)
            report = json.loads((destination / "report.json").read_text())
            validate_report(report,route)
            pngs = []
            for request in range(4):
                path = destination / f"{request}.png"
                with Image.open(path) as image:
                    image.load()
                    if image.size != (512,512):
                        raise ValueError("full-size unresized PNG required")
                pngs.append(sha256_file(path))
            data["trials"].append(dict(index=index,route=route,report=report,memory=memory,png_sha256=pngs))
            write_json(target,data)
        data["medians"] = {}
        for route in sorted(set(data["order"])):
            trials = [trial for trial in data["trials"] if trial["route"] == route]
            samples = [row["request_wall_including_export"] for trial in trials for row in trial["report"]["runs"][1:3]]
            data["medians"][route] = dict(samples=samples,fresh_warm_seconds=statistics.median(samples),
                cold_seconds=[trial["report"]["runs"][0]["request_wall_including_export"] for trial in trials],
                repeated_prompt_seconds=[trial["report"]["runs"][3]["request_wall_including_export"] for trial in trials],
                constructor_seconds=[trial["report"]["constructor_seconds"] for trial in trials])
        data["status"] = "complete_diagnostic"
    except BaseException as error:
        data.update(status="failed",error=str(error))
        raise
    finally:
        write_json(target,data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model","pack","mflux","library","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--python",type=Path)
    parser.add_argument("--extra",type=Path)
    parser.add_argument("--order",default=",".join(ORDER))
    parser.add_argument("--child",choices=tuple(dict.fromkeys(PREDECODE_ORDER)))
    args = parser.parse_args()
    for name in ("model","pack","mflux","library"):
        setattr(args,name,getattr(args,name).resolve(strict=True))
    args.output = args.output.absolute()
    if args.child:
        child(args)
    else:
        if (args.output.exists() or not args.python or not args.extra or
                args.order.split(",") not in (list(ORDER),list(PREDECODE_ORDER),["native-q4","mflux-q4"],["native-q4","native-predecoded"])):
            parser.error("fresh output, reference environment and exact serial order required")
        args.python,args.extra = args.python.absolute(),args.extra.resolve(strict=True)
        parent(args)


if __name__ == "__main__":
    main()
