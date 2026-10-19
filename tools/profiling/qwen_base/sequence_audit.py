#!/usr/bin/env python3
"""CPU audit: same fixed ANE graph across generation,1ref,2ref,generation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image

import analyze
import collect
from collect import ROOT, model_snapshot, sha256_file, write_json


def arithmetic_recipe(summary,report):
    projection=summary.get("ffn_projection","descriptor");activation=summary.get("ffn_activation","exp")
    if (projection not in collect.FFN_PROJECTIONS or activation not in collect.FFN_ACTIVATIONS or
            report.get("ffn_projection","descriptor")!=projection or report.get("ffn_activation","exp")!=activation or
            (activation!="exp" and (projection!="descriptor" or summary.get("fixed_program") is not True))):
        raise ValueError("actual fixed activation/projection recipe must match the complete variable-length window")
    return projection,activation


def geometry(requests, metrics):
    if len(requests) != 4 or len(metrics) != 4:
        raise ValueError("all four heterogeneous requests required")
    normalized = [{k: v for k, v in row.items() if k != "output"} for row in requests]
    if normalized[0] != normalized[3]:
        raise ValueError("last generation must exactly restore the first actual request")
    operations = ["image.generate", "image.edit", "image.edit", "image.generate"]
    counts = [0, 1, 2, 0]
    result = []
    for index, (request, metric) in enumerate(zip(requests, metrics)):
        if (request.get("operation") != operations[index] or len(request.get("inputs", [])) != counts[index] or
                request.get("seed") != 73 or request.get("loras") or
                metric.get("operation") != operations[index] or metric.get("reference_tokens") != counts[index] * 4096):
            raise ValueError("actual standard1024 reference counts and complete operations must match the matrix")
        phase = metric.get("qwen_ffn_phases", {})
        prefill, decode = phase.get("prefill", {}), phase.get("decode", {})
        total = prefill.get("actual_rows")
        text = metric.get("text_tokens")
        if (type(total) is not int or type(text) is not int or text <= 0 or
                total != 1024 + counts[index] * 4096 + text or decode.get("actual_rows") != 1024 or
                prefill.get("steps_this_request") != 1 or decode.get("steps_this_request") != 39 or
                prefill.get("completed_channel_blocks_this_request") != 32 or
                decode.get("completed_channel_blocks_this_request") != 1248 or
                prefill.get("runtime_calls_this_request") != 32 * math.ceil(total / 1024) or
                decode.get("runtime_calls_this_request") != 1248):
            raise ValueError("actual text/image sequence geometry required")
        result.append(dict(index=index, operation=operations[index], references=counts[index],
                           reference_tokens=metric["reference_tokens"], text_tokens=text, actual_prefill_rows=total,
                           predicted_runtime_calls=32 * (39 + math.ceil(total / 1024))))
    return result


def pixels(path):
    with Image.open(path) as image:
        image.load()
        if image.size != (512, 512):
            raise ValueError("complete512 output required")
        return np.asarray(image.convert("RGBA"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Comfy-Org-Qwen-Image-2.1")
    parser.add_argument("--source-archive", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh output required")
    summary = json.loads((args.root / "summary.json").read_text())
    report = json.loads((args.root / "trial/report.json").read_text())
    if (summary["status"] != "complete" or report["status"] != "complete" or summary["scenario"] != "sequence-matrix" or
            report["scenario"] != "sequence-matrix" or summary["route"] != "runtime" or summary["mode"] != "timeline"):
        raise ValueError("complete explicit heterogeneous runtime diagnostic required")
    if summary["assets"]["library"] != sha256_file(args.library) or summary["model_snapshot"] != model_snapshot(args.model):
        raise ValueError("unchanged native/model identity required")
    manifest = json.loads((args.library.parent / "runtime-build/runtime-build-manifest.json").read_text())
    if report["loaded_runtime_id"] != manifest["runtime_build_id"]:
        raise ValueError("loaded native identity differs")
    for entry in manifest["inputs"]["sources"]:
        if sha256_file(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("native source differs from the linked runtime")
    historical = {}
    for path in args.source_archive:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():
                    historical.setdefault(member.name, set()).add(archive.extractfile(member).read())
    source = json.loads((args.root / "source.json").read_text())
    if set(source) != set(summary["source_sha256"]):
        raise ValueError("complete profiling source snapshot required")
    for name, digest in summary["source_sha256"].items():
        raw = source[name].encode()
        if hashlib.sha256(raw).hexdigest() != digest or not (raw == (ROOT / name).read_bytes() or raw in historical.get(name, set())):
            raise ValueError("profiling tool current/archive identity differs")
    metrics = [row["metrics"] for row in report["runs"]]
    joint = summary.get("private_qkv", False)
    if report.get("private_qkv", False) is not joint:
        raise ValueError("actual joint-model contract differs from the parent")
    fixed = summary.get("fixed_program",False)
    if report.get("fixed_program",False) is not fixed:
        raise ValueError("actual fixed-program contract differs from the parent")
    projection,activation=arithmetic_recipe(summary,report)
    collect.validate(metrics, "runtime", "timeline", summary["channels"], private_qkv=joint,fixed_program=fixed,
        ffn_projection=projection,ffn_activation=activation)
    requests = [json.loads((args.root / f"trial/request-{i}.json").read_text()) for i in range(4)]
    shapes = geometry(requests, metrics)
    observer = json.loads((args.root / "trial/observer.json").read_text())
    if observer["source_sha256"] != summary["source_sha256"]["tools/profiling/qwen_base/observer.mm"]:
        raise ValueError("compiled observer source differs")
    observed = analyze.observer_contract(observer, report["runs"], True, expected_models=2 if joint else 1)
    if not observed["hot_no_compile_observed"] or not observed["hot_no_model_load_observed"]:
        raise ValueError("shape/task changes compiled or loaded another hot ANE model")
    previous = 0
    for index, shape in enumerate(shapes):
        actual = metrics[index]["hybrid"]["runtime_calls_session_total"]
        if actual - previous != shape["predicted_runtime_calls"]:
            raise ValueError("actual prediction chunks do not match the complete variable-length sequence")
        previous = actual
        api = observed["requests"][index]["api_calls"]["ane.evaluate_api"]
        qkv_calls = 1248 if joint else 0
        if api != shape["predicted_runtime_calls"] + qkv_calls + (3 if index == 0 else 0):
            raise ValueError("actual ANE evaluations differ from workload plus first capability probes")
        shape.update(runtime_calls=shape["predicted_runtime_calls"], private_qkv_calls=qkv_calls, wall_seconds=report["runs"][index]["request_wall_including_export"],
                     api_calls=observed["requests"][index]["api_calls"])
    images = [pixels(args.root / f"trial/{i}.png") for i in range(4)]
    if not np.array_equal(images[0], images[3]):
        raise ValueError("returning from longer reference shapes changed original generation RGBA")
    for name in ("report", "evidence"):
        if sha256_file(args.root / summary["memory"][name]) != summary["memory"][name + "_sha256"]:
            raise ValueError("memory evidence digest changed")
    if not summary["memory"]["verified"]["complete"] or summary["memory"]["verified"]["command_exit_code"] != 0:
        raise ValueError("complete successful process memory evidence required")
    if summary["analysis"]["hot_median_seconds"] is not None:
        raise ValueError("heterogeneous task timings must not be pooled")
    result = dict(schema="tc-qwen-base-sequence-audit-v1", status="complete", qualification_passed=False,
                  library_sha256=summary["assets"]["library"], native_sources_verified=len(manifest["inputs"]["sources"]),
                  matrix=shapes, observer=observed, private_qkv=joint, fixed_program=fixed,ffn_projection=projection,ffn_activation=activation,generation_rgba_restored_exact=True,
                  one_vs_two_duplicate_reference_rgba_exact=bool(np.array_equal(images[1], images[2])),
                  memory=summary["memory"]["verified"], summary_sha256=sha256_file(args.root / "summary.json"),
                  source_archive_sha256={str(p): sha256_file(p) for p in args.source_archive}, script_sha256=sha256_file(Path(__file__)),
                  scope="actual four complete requests; duplicate references test sequence length, not multi-image semantic quality; no cross-task speedup median")
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
