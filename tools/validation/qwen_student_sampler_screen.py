#!/usr/bin/env python3
"""Authored five vs default six nodes, same pinned r128/joint GPU recipe.

This changes NFE and quality tier, not equivalent-work kernel throughput.
Timing mode primes each edit tier after every switch, scores actual strong
prefix hits on both arms, and verifies six -> five -> six restoration.
Quality mode freezes a broader prompt/seed matrix and disables prefix reuse.
SSIM/difference describe changes; they do not measure instruction adherence.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys

from flux9_bf16_framework_screen import write_json
from image_framework_native import generate_requests
from qwen21_dit_cache_report import image_metrics
from qwen_student_prefix_screen import PROMPTS, adapter_pin
from runtime_ane_common import benchmark_environment, sha256_file
from runtime_ane_load import LoadObservation
from runtime_ane_memory import run_sampled

ROOT = Path(__file__).resolve().parents[2]
FLAG = "TURBOCIDER_QWEN21_STUDENT_FIVE_STEP"
FIVE = "experimental authored five-step student sampler; distinct quality tier"
SIX = "six-step student schedule"
QUALITY_CASES = (
    ("photo", "generate", PROMPTS["generate"]),
    ("english-text", "generate", 'A clean front-facing shop sign, the exact words "GREEN TEA" in large dark green uppercase letters on a plain cream background. Crisp typography, no other text.'),
    ("chinese-text", "generate", '一张极简海报，白色背景，中间用大号清晰的中文黑体写着“秋日茶会”，下方放着一个蓝色陶瓷茶壶。只有这四个汉字，不要其他文字。'),
    ("spatial", "generate", "Exactly three separate objects on a white tabletop: a red cube on the left, a blue sphere in the center, a yellow cone on the right. Front view, studio lighting, no other objects."),
    ("edit-color", "edit", PROMPTS["edit"]),
    ("edit-material", "edit", "Change only the teapot material to transparent clear glass. Preserve its shape, handle, spout, position, wooden table and background."),
    ("edit-add-object", "edit", "Add exactly one small yellow lemon on the table to the left of the blue teapot. Preserve the teapot, its shape and blue ceramic material, the table and background."),
)


def points(seed, mode="timing"):
    if mode == "quality":
        values = []
        for case_id, case, prompt in QUALITY_CASES:
            for i, (steps, value) in enumerate(((6, seed), (5, seed), (5, seed+1), (6, seed+1))):
                values.append(dict(case_id=case_id, case=case, prompt=prompt, steps=steps, seed=value,
                                   conditioning_hit=i != 0, prefix=None, score=False, label="quality-only"))
        return values
    if mode != "timing":
        raise ValueError("known sampler screen mode required")
    values = []
    for i, (steps, value) in enumerate(((6, seed), (5, seed), (6, seed), (5, seed),
                                       (5, seed+1), (6, seed+1), (6, seed), (5, seed))):
        values.append(dict(case_id="photo", case="generate", prompt=PROMPTS["generate"], steps=steps,
                           seed=value, conditioning_hit=i != 0, prefix=None, score=i >= 2,
                           label="setup" if i < 2 else "generation-score"))
    # A single bank must miss on every sampler change, then hit in its own
    # arithmetic. Never time a six-step miss against a five-step hit.
    edit = ((6,seed,False,False), (6,seed,True,True),
            (5,seed,False,False), (5,seed,True,True), (5,seed+1,True,True),
            (6,seed+1,False,False), (6,seed+1,True,True), (6,seed,True,True),
            (5,seed,False,False), (5,seed,True,True),
            (6,seed,False,False), (6,seed,True,False))
    for i, (steps, value, hit, score) in enumerate(edit):
        values.append(dict(case_id="edit-color", case="edit", prompt=PROMPTS["edit"], steps=steps,
                           seed=value, conditioning_hit=i != 0, prefix=hit, score=score,
                           label="edit-score" if score else "prime-or-restoration"))
    return values


def requests(args):
    values = []
    for i, point in enumerate(points(args.seed, args.mode)):
        request = dict(model="qwen-image-2.1", operation="image."+point["case"], prompt=point["prompt"],
            width=512, height=512, steps=point["steps"], frames=1, audio=False, seed=point["seed"],
            execution="gpu", residency="resident", allow_approximation=True, qwen21_dit_cache="off",
            lora_strategy="inference_time", loras=[dict(path=str(args.lora), strength=1., role="transformer")],
            output=str(args.output / f"{i}.png"))
        if point["case"] == "edit":
            request.update(qwen21_reference_size=512,
                           inputs=[dict(kind="image", role="reference", path=str(args.reference))])
        values.append(request)
    return values


def validate(report, seed, mode="timing"):
    expected = points(seed, mode)
    if len(report["runs"]) != len(expected):
        raise ValueError("complete frozen sampler sequence required")
    for row, point in zip(report["runs"], expected):
        m = row["metrics"]
        steps = point["steps"]
        if ((m["model"], m["operation"], m["width"], m["height"], m["steps"], m["actual_denoise_steps"], m["seed"]) !=
                ("qwen-image-2.1", "image."+point["case"], 512, 512, steps, steps, point["seed"])):
            raise ValueError("actual sampler/task differs")
        wall = row["request_wall_including_export"]
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
            raise ValueError("finite complete request wall required")
        if (m.get("runtime_backend") != "mlx_cpp_metal" or m.get("runtime_precision") != "bf16" or
                m.get("encoder_runtime_backend") != "mlx_cpp_metal" or
                m.get("hybrid") or m.get("encoder_hybrid") or m.get("student_ffn_reuse") or m.get("qwen21_dbcache") or
                m.get("lora_applied_projections") != 227 or m.get("lora_strategy") != "inference_time"):
            raise ValueError("complete ordinary GPU student and original decoder required")
        selection = m["acceleration_selection"]
        if "joint BF16 LoRA A/B operands" not in selection:
            raise ValueError("fixed joint A/B and F32 ranks required")
        if ((FIVE in selection) != (steps == 5) or (SIX in selection) != (steps == 6)):
            raise ValueError("actual sampler receipt differs")
        labels = m["plan"]["algorithm_approximations"]
        tag = "qwen21_viggle_v021_r128_" + ("5step_sampler_experimental" if steps == 5 else "6step_distillation")
        other = "qwen21_viggle_v021_r128_" + ("6step_distillation" if steps == 5 else "5step_sampler_experimental")
        if tag not in labels or other in labels:
            raise ValueError("planned sampler label differs")
        if m["prompt_cache_hit"] is not point["conditioning_hit"]:
            raise ValueError("conditioning lifecycle differs")
        prefix = point["prefix"]
        if (("edit prefix KV snapshot hit" in selection) != (prefix is True) or
                ("edit prefix KV snapshot miss" in selection) != (prefix is False)):
            raise ValueError("actual per-tier prefix lifecycle differs")


def child(args):
    sequence = points(args.seed, args.mode)
    def before(index, request):
        point = sequence[index]
        os.environ[FLAG] = str(int(point["steps"] == 5))
        os.environ["TURBOCIDER_QWEN21_LORA_BF16_AB"] = "1"
        os.environ["TURBOCIDER_QWEN21_PREFIX_SNAPSHOT"] = str(int(point["prefix"] is not None))
        print(json.dumps(dict(index=index, **point), ensure_ascii=False), flush=True)
    report = generate_requests(args.library, "qwen-image-2.1", args.model, requests(args), args.output,
                               "authored-five-vs-six", before_request=before)
    validate(report, args.seed, args.mode)
    report["status"] = "complete"
    write_json(args.output / "report.json", report)


def analyze(output, report, seed, mode="timing"):
    import numpy as np
    from PIL import Image, ImageDraw
    validate(report, seed, mode)
    sequence = points(seed, mode)
    pixels = []
    for i in range(len(sequence)):
        with Image.open(output / f"{i}.png") as im:
            im.load()
            if im.size != (512,512):
                raise ValueError("complete 512 image required")
            pixels.append(np.asarray(im.convert("RGBA")))
    pairs = ((2,3), (5,4), (9,11), (14,12)) if mode == "timing" else tuple(
        (base+a, base+b) for base in range(0,len(sequence),4) for a,b in ((0,1),(3,2)))
    rows = []
    for a,b in pairs:
        pa,pb = sequence[a],sequence[b]
        if any(pa[k] != pb[k] for k in ("case_id","case","prompt","seed")) or (pa["steps"],pb["steps"]) != (6,5):
            raise ValueError("unmatched sampler quality pair")
        x,y = pixels[a],pixels[b]
        rows.append(dict(case_id=pa["case_id"], seed=pa["seed"], six_index=a, five_index=b,
            rgb=image_metrics(x[:,:,:3],y[:,:,:3]),
            alpha_max_abs=int(np.abs(x[:,:,3].astype(int)-y[:,:,3]).max()),
            alpha_ranges=[[int(z[:,:,3].min()),int(z[:,:,3].max())] for z in (x,y)],
            rgba_equal=bool(np.array_equal(x,y))))
    result = dict(quality_differences=rows, instruction_quality_qualified=False,
                  scope="same pinned model/adapter and joint arithmetic; five vs six NFE; no teacher40 denominator")
    if mode == "timing":
        for a,b in ((2,6),(3,7),(9,15),(9,19),(11,17)):
            if not np.array_equal(pixels[a],pixels[b]):
                raise ValueError("return-to-sampler changed output")
        timing = []
        for case in ("generate","edit"):
            samples = {steps: [row["request_wall_including_export"] for row,point in zip(report["runs"],sequence)
                              if point["case"] == case and point["steps"] == steps and point["score"]]
                       for steps in (6,5)}
            medians = {k:statistics.median(v) for k,v in samples.items()}
            timing.append(dict(case=case, six_seconds=samples[6],five_seconds=samples[5],
                six_median=medians[6],five_median=medians[5],reduction_percent=100*(1-medians[5]/medians[6])))
        result.update(timing=timing,same_tier_restored=True,
                      timing_scope="conditioning-hot; actual strong prefix hits in BOTH edit arms; setup/prime excluded")
    sheet = Image.new("RGB", (512, len(rows)*276), "white")
    draw = ImageDraw.Draw(sheet)
    for i,row in enumerate(rows):
        for j,key in enumerate(("six_index","five_index")):
            image = Image.fromarray(pixels[row[key]],"RGBA")
            # Gray display background; original PNG retains its alpha channel.
            background = Image.new("RGBA",image.size,(220,220,220,255))
            background.alpha_composite(image)
            sheet.paste(background.convert("RGB").resize((256,256)),(j*256,i*276+20))
            draw.text((j*256,i*276+3),f'{row["case_id"]} seed{row["seed"]}: {6-j} nodes',fill="black")
    sheet.save(output / "comparison.png")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--model", type=Path, default=ROOT/"models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--lora", type=Path, default=ROOT/"models/Viggle-Qwen-Image-2.1-viggle-turbo/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors")
    p.add_argument("--reference", type=Path, default=ROOT/"results/qwen21/teapot-blue-edit-native-512.png")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=73)
    p.add_argument("--mode", choices=("timing","quality"), default="timing")
    p.add_argument("--child", action="store_true")
    args = p.parse_args()
    for key in ("library","model","lora","reference"):
        setattr(args,key,getattr(args,key).resolve(strict=True))
    args.output = args.output.resolve()
    if args.child:
        child(args); return
    pin = adapter_pin(args.lora)
    if pin["rank"] != "r128" or sha256_file(args.lora) != pin["sha256"]:
        p.error("pinned original r128 required")
    args.output.mkdir(parents=True,exist_ok=False)
    trial = args.output/"trial"; trial.mkdir()
    sources = [Path(__file__),ROOT/"tools/validation/image_framework_native.py",
               ROOT/"native/models/qwen21/scheduler.hpp",ROOT/"native/models/qwen21/viggle_adapter.hpp",
               ROOT/"native/models/qwen21/diagnostic_options.hpp",ROOT/"native/models/qwen21_module.cpp",
               ROOT/"native/models/qwen21/pipeline.cpp",ROOT/"native/platform/apple/results.mm"]
    hashes = {str(x.relative_to(ROOT)):sha256_file(x) for x in sources}
    assets = {key:sha256_file(getattr(args,key)) for key in ("library","lora","reference")}
    summary = dict(schema="tc-qwen-student-sampler-screen-v1",status="running",mode=args.mode,
        qualification_passed=False,instruction_quality_qualified=False,source_sha256=hashes,
        asset_sha256=assets,adapter_rank=pin["rank"],seed=args.seed,
        author_readme_revision="b667ee50696e8f4ccd161e02f8af0e43bb5665ed",
        raw_nodes={"six":[1,.9375,.875,.75,.5,.25],"five":[1,.875,.75,.5,.25]},
        frozen_sequence=points(args.seed,args.mode),
        scope="sampler quality/NFE tradeoff, not equivalent-work acceleration; no App/default change")
    write_json(args.output/"source.json",{str(x.relative_to(ROOT)):x.read_text() for x in sources})
    write_json(args.output/"summary.json",summary)
    command = [sys.executable,"-B",str(Path(__file__).resolve()),"--child","--library",str(args.library),
               "--model",str(args.model),"--lora",str(args.lora),"--reference",str(args.reference),
               "--seed",str(args.seed),"--mode",args.mode,"--output",str(trial)]
    observer = LoadObservation(args.output/"load.jsonl")
    try:
        with (args.output/"stdout.log").open("x") as stdout, (args.output/"stderr.log").open("x") as stderr:
            summary["memory"] = run_sampled(command,repo=ROOT,output=args.output,stem="trial",
                env=benchmark_environment(),stdout=stdout,stderr=stderr,timeout=1800,
                interval_ms=100,max_gap_ms=500,observer=observer)
        report = json.loads((trial/"report.json").read_text())
        summary["analysis"] = analyze(trial,report,args.seed,args.mode)
        summary["status"] = "complete"
    except Exception as error:
        summary.update(status="failed",error=str(error))
    try:
        summary["load"] = observer.verify()
    except ValueError as error:
        summary["load_error"] = str(error)
    if any(sha256_file(x) != hashes[str(x.relative_to(ROOT))] for x in sources):
        summary.update(status="failed",error="source changed during observation")
    if any(sha256_file(getattr(args,key)) != value for key,value in assets.items()):
        summary.update(status="failed",error="asset changed during observation")
    write_json(args.output/"summary.json",summary)
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    if summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
