#!/usr/bin/env python3
"""Independent CPU recheck of matched fixed-QKV + fixed-FFN full requests."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

import numpy as np
from PIL import Image, ImageDraw

import analyze
import collect
from audit import normalized, pixels, quality
from collect import ROOT, model_snapshot, sha256_file, write_json


def archive_contract(path, manifest):
    with tarfile.open(path, "r:gz") as archive:
        members = {m.name: m for m in archive.getmembers()}
        if set(members) != {e["path"] for e in manifest["inputs"]["sources"]}:
            raise ValueError("complete bound native source archive required")
        for entry in manifest["inputs"]["sources"]:
            payload = archive.extractfile(members[entry["path"]]).read()
            if hashlib.sha256(payload).hexdigest() != entry["sha256"] or sha256_file(ROOT / entry["path"]) != entry["sha256"]:
                raise ValueError("native current/archive/source identity differs")


def window(root, library, manifest, enabled, mode, case, prefix, historical=None, *, channels=6144, fixed_program=False, ffn_projection="descriptor",ffn_activation="exp"):
    if type(channels) is not int or channels not in (4096,5120,6144):
        raise ValueError("explicit supported fixed FFN partition required")
    summary = json.loads((root / "summary.json").read_text())
    report = json.loads((root / "trial/report.json").read_text())
    if (summary["status"] != "complete" or report["status"] != "complete" or
            summary["mode"] != mode or summary["route"] != "runtime" or summary["case"] != case or
            summary["channels"] != channels or summary.get("private_qkv") is not enabled or
            report.get("private_qkv") is not enabled or summary.get("runtime_prefix_snapshot") is not prefix or
            summary.get("fixed_program",False) is not fixed_program or report.get("fixed_program",False) is not fixed_program or
            summary.get("ffn_projection","descriptor")!=ffn_projection or report.get("ffn_projection","descriptor")!=ffn_projection or
            summary.get("ffn_activation","exp")!=ffn_activation or report.get("ffn_activation","exp")!=ffn_activation or
            summary.get("resident_prefix") or summary["scenario"] != "repeat" or len(report["runs"]) != 3 or
            summary["assets"]["library"] != sha256_file(library) or report["loaded_runtime_id"] != manifest["runtime_build_id"]):
        raise ValueError("complete matched original-base fixed-partition joint/off window required")
    if summary["model_snapshot"] != model_snapshot(ROOT / "models/Comfy-Org-Qwen-Image-2.1"):
        raise ValueError("original model generation changed")
    snapshot = json.loads((root / "source.json").read_text())
    if set(snapshot) != set(summary["source_sha256"]):
        raise ValueError("complete profiling source snapshot required")
    for name, digest in summary["source_sha256"].items():
        raw = snapshot[name].encode()
        if hashlib.sha256(raw).hexdigest() != digest or not (raw == (ROOT / name).read_bytes() or raw in (historical or {}).get(name, set())):
            raise ValueError("bound profiling source changed")
    metrics = [r["metrics"] for r in report["runs"]]
    collect.validate(metrics, "runtime", mode, channels, prefix, False, enabled, fixed_program,ffn_projection,ffn_activation)
    previous = 0
    for index, metric in enumerate(metrics):
        request = json.loads((root / f"trial/request-{index}.json").read_text())
        if (request.get("seed") != (73, 74, 73)[index] or request.get("loras") or request.get("audio") is not False or
                request.get("operation") != "image." + case or metric.get("operation") != request["operation"] or
                (case == "generate" and request.get("inputs")) or
                (case == "edit" and (len(request.get("inputs", [])) != 1 or request.get("qwen21_reference_size") != 1024))):
            raise ValueError("actual full-base generation/edit seed/geometry contract differs")
        calls = metric["hybrid"]["runtime_calls_session_total"]
        expected = 1280 if prefix and index > 0 else 32 * (39 + (6 if case == "edit" else 2))
        if calls - previous != expected:
            raise ValueError("actual FFN submissions differ from complete forty-step geometry")
        previous = calls
    images = [pixels(root / f"trial/{i}.png") for i in range(3)]
    if np.array_equal(images[0], images[1]) or np.array_equal(images[1], images[2]):
        raise ValueError("changed seeds returned stale identical output")
    if not prefix and not np.array_equal(images[0], images[2]):
        raise ValueError("same-seed generation did not restore after alternate input")
    for name in ("report", "evidence"):
        if sha256_file(root / summary["memory"][name]) != summary["memory"][name + "_sha256"]:
            raise ValueError("memory evidence digest changed")
    if not summary["memory"]["verified"]["complete"] or summary["memory"]["verified"]["command_exit_code"] != 0:
        raise ValueError("complete successful owned process window required")
    hot = [r["request_wall_including_export"] for r in report["runs"][1:]]
    if statistics.median(hot) != summary["analysis"]["hot_median_seconds"]:
        raise ValueError("complete request wall denominator differs")
    deltas = analyze.native_deltas(report["runs"])
    if deltas != summary["analysis"]["native_runtime_deltas"]:
        raise ValueError("FFN per-request native deltas differ")
    observed = None
    if mode == "timeline":
        raw = json.loads((root / "trial/observer.json").read_text())
        if raw["source_sha256"] != summary["source_sha256"]["tools/profiling/qwen_base/observer.mm"]:
            raise ValueError("actual observer binary source differs")
        observed = analyze.observer_contract(raw, report["runs"], True, expected_models=2 if enabled else 1)
        if (observed != summary["analysis"]["observer"] or not observed["hot_no_compile_observed"] or
                not observed["hot_no_model_load_observed"]):
            raise ValueError("actual fixed-model hot compile/load or observed analysis differs")
    receipt = dict(cohort=root.name, mode=mode, private_qkv=enabled, channels=channels,fixed_program=fixed_program,ffn_projection=ffn_projection,ffn_activation=ffn_activation,
        first_seconds=report["runs"][0]["request_wall_including_export"], hot_seconds=hot,
        hot_median_seconds=statistics.median(hot), observer=observed, ffn_deltas=deltas,
        qkv_receipts=[r.get("qwen_private_qkv") for r in metrics], memory=summary["memory"]["verified"],
        load_error=summary.get("load_error"), png_sha256=[sha256_file(root / f"trial/{i}.png") for i in range(3)],
        summary_sha256=sha256_file(root / "summary.json"))
    return dict(root=root, summary=summary, report=report, images=images, receipt=receipt)


def matched(off, on):
    a, b = off["summary"]["environment"], on["summary"]["environment"]
    if set(a) != set(b) or {k for k in a if a[k] != b[k]} != {"TURBOCIDER_QWEN21_PRIVATE_QKV"}:
        raise ValueError("only the explicit QKV switch may differ in the matched environments")
    for key in ("library", "manifest", "reference"):
        if off["summary"].get("assets", {}).get(key) != on["summary"].get("assets", {}).get(key):
            raise ValueError("matched native/manifest/reference assets differ")
    for i in range(3):
        if normalized(off["root"] / f"trial/request-{i}.json") != normalized(on["root"] / f"trial/request-{i}.json"):
            raise ValueError("actual matched requests differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("off", "on", "library", "native-source-archive", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--case", choices=("generate", "edit"), required=True)
    parser.add_argument("--timeline", type=Path)
    parser.add_argument("--reverse-off", type=Path)
    parser.add_argument("--gpu-reference", type=Path, help="frozen GPU PNG directory; quality only, never a fresh timing denominator")
    parser.add_argument("--tool-source-archive", action="append", type=Path, default=[])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh audit directory required")
    manifest = json.loads((args.library.parent / "runtime-build/runtime-build-manifest.json").read_text())
    archive_contract(args.native_source_archive, manifest)
    historical = {}
    for path in args.tool_source_archive:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():
                    historical.setdefault(member.name, set()).add(archive.extractfile(member).read())
    prefix = args.case == "edit"
    def load(root, enabled, mode):
        return window(root, args.library, manifest, enabled, mode, args.case, prefix, historical)
    off, on = load(args.off, False, "stock"), load(args.on, True, "stock")
    matched(off, on)
    cohorts = [off["receipt"], on["receipt"]]
    if args.timeline:
        timeline = load(args.timeline, True, "timeline")
        if any(not np.array_equal(a, b) for a, b in zip(timeline["images"], on["images"])):
            raise ValueError("timeline and stock returned different same-config images")
        cohorts.append(timeline["receipt"])
    if args.reverse_off:
        reverse = load(args.reverse_off, False, "stock")
        matched(reverse, on)
        if any(not np.array_equal(a, b) for a, b in zip(reverse["images"], off["images"])):
            raise ValueError("reverse control did not restore baseline outputs")
        cohorts.append(reverse["receipt"])
    comparisons = [quality(off["images"][i], on["images"][i]) for i in range(3)]
    absolute = None
    if args.gpu_reference:
        absolute = [quality(pixels(args.gpu_reference / f"{i}.png"), on["images"][i]) for i in range(3)]
    base = off["receipt"]["hot_median_seconds"]
    candidate = on["receipt"]["hot_median_seconds"]
    result = dict(schema="tc-qwen-joint-fixed-qkv-ffn-audit-v1", status="complete", qualification_passed=False,
        case=args.case, runtime_prefix_snapshot=prefix, runtime_build_id=manifest["runtime_build_id"],
        script_sha256=sha256_file(Path(__file__)),
        library_sha256=sha256_file(args.library), native_source_archive_sha256=sha256_file(args.native_source_archive),
        tool_source_archives={str(p): sha256_file(p) for p in args.tool_source_archive},
        cohorts=cohorts, incremental_quality=comparisons, frozen_gpu_quality=absolute,
        stock_hot_wall_reduction=1-candidate/base,
        limitations=["Small pilot without device exclusivity or thermal/statistical qualification.",
            "W8/A8 and FP16 program output are approximate; pixel metrics do not establish semantic image quality.",
            "Private API host selectors, not ANE hardware counters; physical INT8 and physical device overlap remain unverified.",
            "Stage lifetime includes upstream/norm completion; overlapping spans cannot be summed into wall time."])
    args.output.mkdir(parents=True)
    sheet = Image.new("RGB", (1024, 3*544), "white")
    draw = ImageDraw.Draw(sheet)
    for i in range(3):
        for j, (label, record) in enumerate((("FFN only", off), ("FFN + QKV", on))):
            draw.text((512*j+8, 544*i+8), f"{label}, seed {(73,74,73)[i]}, request {i}", fill="black")
            sheet.paste(Image.fromarray(record["images"][i][:,:,:3]), (512*j,544*i+32))
    sheet.save(args.output / "comparison.png")
    write_json(args.output / "audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
