#!/usr/bin/env python3
"""Audit actual same-payload FLUX9B DiT Q4/native/reference whole requests."""
import argparse
import json
from pathlib import Path
import statistics

from flux9_framework_evidence import portable, rgb
from flux9_matched_affine_screen import ORDER, validate_report
from runtime_ane_common import sha256_file
from runtime_ane_image_compare import rgb_metrics
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def portable_tree(value):
    if isinstance(value,dict):
        return {portable(key):portable_tree(item) for key,item in value.items()}
    if isinstance(value,list):
        return [portable_tree(item) for item in value]
    return portable(value)


def cohort(data):
    if data.get("status") != "complete_diagnostic" or data.get("order") != list(ORDER) or len(data.get("trials",[])) != 8:
        raise ValueError("complete exact four-route matched-precision palindrome required")
    for index,trial in enumerate(data["trials"]):
        if trial["index"] != index or trial["route"] != ORDER[index]:
            raise ValueError("actual measured route order differs")
        validate_report(trial["report"],trial["route"])
    for route in dict.fromkeys(ORDER):
        trials = [trial for trial in data["trials"] if trial["route"] == route]
        samples = [row["request_wall_including_export"] for trial in trials for row in trial["report"]["runs"][1:3]]
        declared = data["medians"][route]
        if declared["samples"] != samples or declared["fresh_warm_seconds"] != statistics.median(samples):
            raise ValueError("all actual four fresh samples must determine the median")
    return data["medians"]


def audit(screen,build_manifest):
    data = json.loads((screen / "summary.json").read_text())
    medians = cohort(data)
    build = json.loads(build_manifest.read_text())
    for source in build["inputs"]["sources"]:
        if sha256_file(ROOT / source["path"]) != source["sha256"]:
            raise ValueError("native build source identity changed")
    library = build_manifest.parent.parent / "libturbocider.dylib"
    if sha256_file(library) != data["local_source_sha256"][str(library.relative_to(ROOT))]:
        raise ValueError("actual native library identity differs")
    for name,digest in data["local_source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("actual comparison helper source changed")
    result = dict(schema="tc-flux9-matched-affine-validation-v1",status="complete_diagnostic",date="2026-10-12",
        qualification_passed=False,goal_complete=False,hardware="M4 Max64GiB/macOS26.6.2",task=data["task"],
        scope=data["scope"],timing_scope=data["timing_scope"],source_policy=data["source_policy"],
        model_sha256=data["model_sha256"],local_source_sha256=data["local_source_sha256"],
        mflux_source_sha256=data["mflux_source_sha256"],mflux_commit=data["mflux_commit"],
        native_inputs_matched=len(build["inputs"]["sources"]),conversion=portable(data["conversion"]),
        conversion_sha256=data["conversion_sha256"],evidence=str((screen / "summary.json").relative_to(ROOT)),
        evidence_sha256=sha256_file(screen / "summary.json"),order=data["order"],medians=medians,trials=[],quality=[])
    images = {}
    for trial in data["trials"]:
        directory = screen / f'{trial["index"]}-{trial["route"]}'
        if json.loads((directory / "report.json").read_text()) != trial["report"]:
            raise ValueError("actual raw child report differs from collected summary")
        memory = trial["memory"]
        actual = memory["verified"]
        if actual["complete"] is not True or actual["command_exit_code"] != 0 or actual["reasons"]:
            raise ValueError("complete successful independent memory proof required")
        for name in ("report","evidence"):
            if sha256_file(directory / memory[name]) != memory[name+"_sha256"]:
                raise ValueError("memory report/raw evidence changed")
        if len(trial["png_sha256"]) != 4:
            raise ValueError("four actual full PNG receipts required")
        for index,digest in enumerate(trial["png_sha256"]):
            path = directory / f"{index}.png"
            if sha256_file(path) != digest:
                raise ValueError("actual full PNG changed")
            images[trial["index"],index] = rgb(path)
            requested = json.loads((directory / f"request-{index}.json").read_text())
            if (requested["model"],requested["operation"],requested["width"],requested["height"],requested["steps"],requested["seed"]) != (
                    "flux2-klein-9b","image.generate",512,512,4,42) or requested.get("loras"):
                raise ValueError("same base source/task required")
            if trial["route"] == "native-q4" and (requested.get("allow_approximation") is not True or requested["compile_gpu"]):
                raise ValueError("packed quantized GPU recipe must remain explicit and eager")
        result["trials"].append(dict(index=trial["index"],route=trial["route"],report=portable(trial["report"]),
            memory={key:actual[key] for key in ("complete","tree_peak_phys_footprint_bytes","compression_bytes","decompression_bytes","swap_in_bytes","swap_out_bytes")},
            memory_report_sha256=memory["report_sha256"],memory_evidence_sha256=memory["evidence_sha256"],png_sha256=trial["png_sha256"]))
    for left,right,label in ((0,1,"native-q4-vs-native-bf16"),(7,6,"native-q4-vs-native-bf16"),
            (2,3,"mflux-q4-vs-mflux-bf16"),(5,4,"mflux-q4-vs-mflux-bf16"),
            (0,2,"native-vs-mflux-bf16"),(7,5,"native-vs-mflux-bf16"),
            (1,3,"native-vs-mflux-q4"),(6,4,"native-vs-mflux-q4")):
        for index in range(4):
            x,y = images[left,index],images[right,index]
            result["quality"].append(dict(comparison=label,left_process=left,right_process=right,request_index=index,
                mae_u8=float(np.mean(np.abs(x.astype(np.float64)-y))),**rgb_metrics(x,y)))
    result["forward_reverse_rgb_determinism"] = []
    for route in dict.fromkeys(ORDER):
        indices = [trial["index"] for trial in data["trials"] if trial["route"] == route]
        result["forward_reverse_rgb_determinism"].append(dict(route=route,exact_requests=[
            bool(np.array_equal(images[indices[0],index],images[indices[1],index])) for index in range(4)]))
    gpu,quant,ref,qref = (medians[name]["fresh_warm_seconds"] for name in ("native-bf16","native-q4","mflux-bf16","mflux-q4"))
    result["comparison"] = dict(native_bf16_over_q4=gpu/quant,mflux_bf16_over_q4=ref/qref,
        reference_over_native_bf16=ref/gpu,reference_over_native_q4=qref/quant,
        native_q4_reduction_percent=100*(1-quant/gpu),mflux_q4_reduction_percent=100*(1-qref/ref))
    result["quality_scope"] = "full unresized RGB and native opaque alpha verified; same engine precision drift versus cross-engine differences separated; no broad perceptual qualification or same-noise claim"
    result = portable_tree(result)
    if "/Users/" in json.dumps(result):
        raise ValueError("machine-specific paths in portable evidence")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("screen","build-manifest","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh evidence output required")
    result = audit(args.screen.resolve(),args.build_manifest.resolve())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as target:
        json.dump(result,target,indent=2,allow_nan=False);target.write("\n")
    print(json.dumps(dict(status=result["status"],comparison=result["comparison"])))


if __name__ == "__main__":
    main()
