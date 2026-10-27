#!/usr/bin/env python3
"""Qwen512 decoder convolution/norm experiments, matched r128/joint GPU controls.

Original FP32 state/norm/attention and reference encoding remain unchanged.
Whole-request and standalone VAE measurements are explicitly separate modes.
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
MARKER = "VAE decoder FP16 convolution operands/output with FP32 state"
FUSED_MARKER = "VAE FP32 fused channel L2 norm/SiLU with NHWC convolution input"


def points(seed):
    # case, candidate enabled, seed; first two per task are excluded setup/prime.
    values = []
    for case in ("generate", "edit"):
        values.extend((case, half, value) for half, value in (
            (False, seed), (True, seed), (False, seed), (True, seed),
            (True, seed+1), (False, seed+1), (False, seed), (True, seed)))
    return values


def requests(args):
    values = []
    for i, (case, _, seed) in enumerate(points(args.seed)):
        request = dict(model="qwen-image-2.1", operation="image."+case, prompt=PROMPTS[case],
            width=512, height=512, steps=6, frames=1, audio=False, seed=seed,
            execution="gpu", residency="resident", allow_approximation=True,
            qwen21_dit_cache="off", lora_strategy="inference_time",
            loras=[dict(path=str(args.lora), strength=1., role="transformer")], output=str(args.output / f"{i}.png"))
        if case == "edit":
            request.update(qwen21_reference_size=512,
                           inputs=[dict(kind="image", role="reference", path=str(args.reference))])
        if i in (0, 8):
            request["dump_tensors"] = str(args.output / ("dump-generate" if i == 0 else "dump-edit"))
        values.append(request)
    return values


def validate(report, seed, recipe="fp16", control="original"):
    if len(report["runs"]) != 16:
        raise ValueError("all original/FP16/return-to-original requests required")
    for i, (row, (case, half, value)) in enumerate(zip(report["runs"], points(seed))):
        m = row["metrics"]
        if ((m["model"], m["operation"], m["width"], m["height"], m["steps"], m["actual_denoise_steps"], m["seed"]) !=
                ("qwen-image-2.1", "image."+case, 512, 512, 6, 6, value)):
            raise ValueError("actual complete six-step workload differs")
        wall = row["request_wall_including_export"]
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
            raise ValueError("finite full request wall required")
        if (m.get("hybrid") or m.get("encoder_hybrid") or m.get("student_ffn_reuse") or m.get("qwen21_dbcache")
                or m.get("lora_applied_projections") != 227 or m.get("lora_strategy") != "inference_time"):
            raise ValueError("same complete GPU student required")
        selection = m["acceleration_selection"]
        if "joint BF16 LoRA A/B operands" not in selection:
            raise ValueError("fixed joint LoRA arithmetic required")
        selected_recipe = recipe if half else control
        expected_half = selected_recipe in ("fp16", "combined")
        expected_fused = selected_recipe in ("fused", "combined")
        if ((MARKER in selection) != expected_half or ("+vae_fp16_conv_fp32_state" in m["runtime_precision"]) != expected_half
                or (FUSED_MARKER in selection) != expected_fused
                or ("+vae_fp32_fused_norm_silu_nhwc" in m["runtime_precision"]) != expected_fused):
            raise ValueError("actual decoder arithmetic differs from configured arm")
        if m["prompt_cache_hit"] is not (i not in (0, 8)):
            raise ValueError("conditioning lifecycle differs")
        if ("edit prefix KV snapshot hit" in selection) != (i >= 10):
            raise ValueError("strong-prefix control differs")
        if i == 9 and "edit prefix KV snapshot miss" not in selection:
            raise ValueError("first non-dump edit must prime its prefix")


def child(args):
    def before(index, request):
        case, half, seed = points(args.seed)[index]
        selected_recipe = args.candidate if half else args.control
        os.environ["TURBOCIDER_QWEN21_LORA_BF16_AB"] = "1"
        os.environ["TURBOCIDER_QWEN21_VAE_FP16_CONV"] = str(int(selected_recipe in ("fp16", "combined")))
        os.environ["TURBOCIDER_QWEN21_VAE_FUSED_NORM"] = str(int(selected_recipe in ("fused", "combined")))
        os.environ["TURBOCIDER_QWEN21_PREFIX_SNAPSHOT"] = "1" if case == "edit" else "0"
        print(json.dumps(dict(index=index, case=case, candidate=half, decoder_recipe=selected_recipe, seed=seed)), flush=True)
    report = generate_requests(args.library, "qwen-image-2.1", args.model, requests(args), args.output,
                               "vae-precision", before_request=before)
    validate(report, args.seed, args.candidate, args.control)
    report["status"] = "complete"
    write_json(args.output / "report.json", report)


def pixels(path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        image.load()
        if image.size != (512, 512):
            raise ValueError("complete512 image required")
        return np.asarray(image.convert("RGBA"))


def quality(a, b):
    import numpy as np
    result = dict(rgb=image_metrics(a[:, :, :3], b[:, :, :3]),
                  alpha_max_abs=int(np.abs(a[:, :, 3].astype(int)-b[:, :, 3]).max()),
                  rgba_equal=bool(np.array_equal(a, b)))
    result["passed"] = (result["rgb"]["ssim"] >= .995 and result["rgb"]["mae_255"] <= 1.
                        and result["alpha_max_abs"] <= 2)
    return result


def analyze(output, report, seed, recipe="fp16", control="original"):
    import numpy as np
    validate(report, seed, recipe, control)
    for a, b in ((2, 6), (3, 7), (10, 14), (11, 15)):
        if not np.array_equal(pixels(output / f"{a}.png"), pixels(output / f"{b}.png")):
            raise ValueError("return-to-arm changed output")
    rows = []
    for label, a, b in (("generation-same-seed", 2, 3), ("generation-new-seed", 5, 4),
                        ("edit-same-seed", 10, 11), ("edit-new-seed", 13, 12)):
        rows.append(dict(case=label, quality=quality(pixels(output / f"{a}.png"), pixels(output / f"{b}.png"))))
    wall = lambda i: report["runs"][i]["request_wall_including_export"]
    timing = []
    for case, original, half in (("generate", [2, 5, 6], [3, 4, 7]), ("edit-prefix-hit", [10, 13, 14], [11, 12, 15])):
        a, b = [wall(i) for i in original], [wall(i) for i in half]
        ma, mb = statistics.median(a), statistics.median(b)
        label = "half" if recipe == "fp16" else "candidate"
        baseline = "original" if control == "original" else "control"
        timing.append(dict(case=case,
                           reduction_percent=100*(1-mb/ma),
                           **{baseline+"_seconds":a,baseline+"_median":ma,
                              baseline+"_decode_seconds":[report["runs"][i]["metrics"]["timings_seconds"]["vae_decode"] for i in original],
                              label+"_seconds":b,label+"_median":mb,
                              label+"_decode_seconds":[report["runs"][i]["metrics"]["timings_seconds"]["vae_decode"] for i in half]}))
    return dict(quality=rows, timing=timing, quality_passed=all(row["quality"]["passed"] for row in rows),
                same_arm_restored=True, scope="conditioning-hot; edit strong prefix hit on both arms; not cold setup")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--model", type=Path, default=ROOT / "models/Comfy-Org-Qwen-Image-2.1")
    p.add_argument("--lora", type=Path, default=ROOT / "models/Viggle-Qwen-Image-2.1-viggle-turbo/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors")
    p.add_argument("--reference", type=Path, default=ROOT / "results/qwen21/teapot-blue-edit-native-512.png")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=73)
    p.add_argument("--child", action="store_true")
    p.add_argument("--candidate", choices=("fp16", "fused", "combined"), default="fp16")
    p.add_argument("--control", choices=("original", "fp16"), default="original")
    p.add_argument("--component-binary", type=Path)
    p.add_argument("--latents", type=Path)
    p.add_argument("--weights", type=Path, default=ROOT / "models/Comfy-Org-Qwen-Image-2.1/vae/qwen_image_2.1_vae_bf16.safetensors")
    args = p.parse_args()
    for key in ("library", "model", "lora", "reference", "component_binary", "latents", "weights"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve(strict=True))
    args.output = args.output.resolve()
    if args.control == "fp16" and args.candidate != "combined":
        p.error("the stronger FP16 control is only for adding fused norm to that same convolution recipe")
    if args.child:
        child(args); return
    component = args.component_binary is not None
    if component and args.control != "original":
        p.error("standalone components retain their original decoder control")
    if component and args.latents is None:
        p.error("component mode needs actual latent input")
    pin = adapter_pin(args.lora)
    if not component and (pin["rank"] != "r128" or sha256_file(args.lora) != pin["sha256"]):
        p.error("pinned r128 required")
    args.output.mkdir(parents=True, exist_ok=False)
    summary = dict(schema="tc-qwen-vae-precision-screen-v1", status="running", qualification_passed=False,
        mode="component" if component else "full-request", candidate_recipe=args.candidate, control_recipe=args.control,
        library_sha256=sha256_file(args.library),
        adapter_sha256=None if component else pin["sha256"], reference_sha256=sha256_file(args.reference),
        gates=dict(minimum_request_reduction_percent=3., minimum_rgb_ssim=.995, maximum_rgb_mae_255=1., maximum_alpha_abs=2))
    sources = [Path(__file__), ROOT / "tools/native/qwen21_vae_precision_probe.mm",
               ROOT / "tools/native/build_qwen21_vae_precision_probe.sh", ROOT / "native/models/qwen21/vae.cpp",
               ROOT / "native/models/qwen21/vae.hpp", ROOT / "native/models/qwen21/metal/vae_norm_silu.hpp"]
    identities = {str(path.relative_to(ROOT)): sha256_file(path) for path in sources}
    summary["source_sha256"] = identities
    write_json(args.output / "source.json", {str(path.relative_to(ROOT)): path.read_text() for path in sources})
    write_json(args.output / "summary.json", summary)
    trial = args.output / "trial"
    if component:
        command = [str(args.component_binary), str(args.weights), str(args.latents), str(trial), str(args.reference)]
        if args.candidate != "fp16":
            command.append(args.candidate)
        summary["binary_sha256"] = sha256_file(args.component_binary)
        summary["input_file_sha256"] = sha256_file(args.latents)
    else:
        trial.mkdir()
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--child", "--library", str(args.library),
            "--model", str(args.model), "--lora", str(args.lora), "--reference", str(args.reference),
            "--seed", str(args.seed), "--output", str(trial), "--candidate", args.candidate,"--control",args.control]
    observer = LoadObservation(args.output / "load.jsonl")
    try:
        with (args.output / "stdout.log").open("x") as stdout, (args.output / "stderr.log").open("x") as stderr:
            summary["memory"] = run_sampled(command, repo=ROOT, output=args.output, stem="trial", env=benchmark_environment(),
                stdout=stdout, stderr=stderr, timeout=1800, interval_ms=100, max_gap_ms=500, observer=observer)
        if component:
            native = json.loads((trial / "native.json").read_text())
            if native["status"] != "complete" or native.get("reference_encoder_exact") is not True:
                raise ValueError("complete VAE and exact reference encoder evidence required")
            if native.get("candidate_recipe", "fp16") != args.candidate:
                raise ValueError("actual component candidate recipe differs")
            candidate_png = trial / "candidate.png"
            if not candidate_png.is_file():
                candidate_png = trial / "half.png"
            q = quality(pixels(trial / "original.png"), pixels(candidate_png))
            timing = {str(half): [row["seconds"] for row in native["samples"] if row.get("candidate", row["half"]) is half and row["iteration"] >= 0]
                      for half in (False, True)}
            summary["analysis"] = dict(quality=q, decode_seconds=timing, scope="constructor+VAE only, not request")
            fallback_calls = sum(row.get("fp16_nonfinite_fallback", False) for row in native["samples"])
            summary["analysis"]["fp16_nonfinite_fallback_calls"] = fallback_calls
            summary["analysis"]["candidate_speed_claim_allowed"] = fallback_calls == 0 and q["passed"]
        else:
            summary["analysis"] = analyze(trial, json.loads((trial / "report.json").read_text()), args.seed, args.candidate,args.control)
        summary["status"] = "complete"
    except Exception as error:
        summary.update(status="failed", error=str(error))
    try:
        summary["load"] = observer.verify()
    except ValueError as error:
        summary["load_error"] = str(error)
    for path in sources:
        if sha256_file(path) != identities[str(path.relative_to(ROOT))]:
            summary.update(status="failed", error="source changed during measurement")
    if sha256_file(args.library) != summary["library_sha256"]:
        summary.update(status="failed", error="library changed during measurement")
    if component and sha256_file(args.component_binary) != summary["binary_sha256"]:
        summary.update(status="failed", error="component binary changed during measurement")
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    if summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
