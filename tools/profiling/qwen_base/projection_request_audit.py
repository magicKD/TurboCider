#!/usr/bin/env python3
"""Independent matched complete-request FFN reduction audit; never a GPU speed claim."""
import argparse
import json
import math
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image, ImageDraw

from audit import normalized, pixels, quality
import collect
import joint_audit
from collect import sha256_file, write_json
from quality_audit import matched_request


def frozen_gpu_report_contract(report):
    # The original base-runtime collector predates the explicit status/route
    # envelope. Admit that exact legacy format, not arbitrary missing status.
    legacy = set(report) == {"engine", "constructor_seconds", "runs"} and report.get("engine") == "gpu"
    current = report.get("status") == "complete" and report.get("route") == "gpu"
    rows = report.get("runs", [])
    if (not (legacy or current) or len(rows) != 3 or [row.get("index") for row in rows] != [0, 1, 2] or
            any(type(row.get("request_wall_including_export")) not in (int, float) or
                not math.isfinite(row["request_wall_including_export"]) or row["request_wall_including_export"] <= 0 for row in rows)):
        raise ValueError("three complete ordered frozen GPU request results required for quality-only control")


def contract(value, projection):
    summary = value["summary"]
    if (projection not in collect.FFN_PROJECTIONS or
            summary.get("ffn_projection") != projection or
            summary.get("ffn_activation","exp")!="exp" or
            summary.get("channels") != 6144 or summary.get("private_qkv") is not True or
            summary.get("fixed_program") is not True or summary.get("resident_prefix") is not False or
            summary.get("case") not in ("generate", "edit") or summary.get("mode") not in ("stock", "timeline") or
            summary.get("runtime_prefix_snapshot") is not (summary["case"] == "edit")):
        raise ValueError("explicit fixed6144 joint projection and matched edit-prefix policy required")
    expected = collect.environment("runtime", summary["mode"], 6144, summary["case"] == "edit",
                                   False, True, True, projection)
    if "ffn_activation" not in summary and "TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION" not in summary.get("environment",{}):
        expected.pop("TURBOCIDER_PRIVATE_ANE_W8_ACTIVATION")
    if summary.get("environment") != expected:
        raise ValueError("complete captured projection environment differs from the declared recipe")


def matched(baseline, candidate, projection):
    if projection not in collect.FFN_PROJECTIONS[1:]:
        raise ValueError("explicit non-default FFN projection required")
    contract(baseline, "descriptor")
    contract(candidate, projection)
    a, b = baseline["summary"], candidate["summary"]
    if a["case"] != b["case"] or a["mode"] != "stock" or b["mode"] != "stock":
        raise ValueError("matched complete stock request cases required")
    x, y = a["environment"], b["environment"]
    if set(x) != set(y) or {k for k in x if x[k] != y[k]} != {"TURBOCIDER_PRIVATE_ANE_W8_PROJECTION"}:
        raise ValueError("only the immutable FFN projection policy may differ")
    if any(a["assets"][key] != b["assets"][key] for key in ("library", "manifest", "reference")):
        raise ValueError("matched library, shape and reference assets differ")
    for index in range(3):
        if normalized(baseline["root"] / f"trial/request-{index}.json") != normalized(candidate["root"] / f"trial/request-{index}.json"):
            raise ValueError("projection changed an actual prompt, seed, geometry or execution request")


def repeated(stock, repeat, projection, mode):
    contract(stock, projection)
    contract(repeat, projection)
    a, b = stock["summary"], repeat["summary"]
    if (a["mode"] != "stock" or b["mode"] != mode or a["case"] != b["case"] or
            any(a["assets"][key] != b["assets"][key] for key in ("library", "manifest", "reference"))):
        raise ValueError("same-recipe reverse/timeline assets and case required")
    for index in range(3):
        if (normalized(stock["root"] / f"trial/request-{index}.json") != normalized(repeat["root"] / f"trial/request-{index}.json") or
                not np.array_equal(stock["images"][index], repeat["images"][index])):
            raise ValueError("reverse/timeline did not restore complete same-recipe requests and PNGs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "library", "native-source-archive", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--projection", choices=collect.FFN_PROJECTIONS[1:], required=True)
    parser.add_argument("--case", choices=("generate", "edit"), required=True)
    parser.add_argument("--timeline", type=Path)
    parser.add_argument("--reverse-baseline", type=Path)
    parser.add_argument("--gpu-reference", type=Path, help="frozen PNGs for quality only; never a timing denominator")
    parser.add_argument("--historical-descriptor", type=Path, help="older same-arithmetic request/PNG control only, never a speed denominator")
    parser.add_argument("--tool-source-archive", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh independent audit directory required")
    manifest = json.loads((args.library.parent / "runtime-build/runtime-build-manifest.json").read_text())
    joint_audit.archive_contract(args.native_source_archive, manifest)
    historical = {}
    for path in args.tool_source_archive:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile():
                    historical.setdefault(member.name, set()).add(archive.extractfile(member).read())

    def load(root, projection, mode="stock"):
        return joint_audit.window(root, args.library, manifest, True, mode, args.case, args.case == "edit", historical,
                                  fixed_program=True, ffn_projection=projection)

    baseline, candidate = load(args.baseline, "descriptor"), load(args.candidate, args.projection)
    matched(baseline, candidate, args.projection)
    cohorts = [baseline["receipt"], candidate["receipt"]]
    for root, stock, projection, mode in ((args.timeline, candidate, args.projection, "timeline"),
                                         (args.reverse_baseline, baseline, "descriptor", "stock")):
        if root is None:
            continue
        value = load(root, projection, mode)
        repeated(stock, value, projection, mode)
        cohorts.append(value["receipt"])
    historical_descriptor = None
    if args.historical_descriptor:
        previous = json.loads((args.historical_descriptor / "report.json").read_text())
        if previous.get("status") != "complete" or len(previous.get("runs", [])) != 3:
            raise ValueError("complete frozen descriptor request control required")
        collect.validate([row["metrics"] for row in previous["runs"]], "runtime", "stock", 6144,
                         args.case == "edit", False, True, True, "descriptor")
        for i in range(3):
            if (normalized(args.historical_descriptor / f"request-{i}.json") != normalized(baseline["root"] / f"trial/request-{i}.json") or
                    not np.array_equal(pixels(args.historical_descriptor / f"{i}.png"), baseline["images"][i])):
                raise ValueError("new default did not preserve the complete frozen same-arithmetic request and RGBA")
        historical_descriptor = dict(report_sha256=sha256_file(args.historical_descriptor / "report.json"),
            png_sha256=[sha256_file(args.historical_descriptor / f"{i}.png") for i in range(3)],
            rgba_exact=True, scope="older descriptor quality-only control; no old timing reused")
    frozen = None
    frozen_receipt = None
    if args.gpu_reference:
        gpu_report = json.loads((args.gpu_reference / "report.json").read_text())
        frozen_gpu_report_contract(gpu_report)
        collect.validate([row["metrics"] for row in gpu_report["runs"]], "gpu", "stock")
        for i in range(3):
            matched_request(normalized(args.gpu_reference / f"request-{i}.json"),
                            normalized(baseline["root"] / f"trial/request-{i}.json"))
        frozen = {projection: [quality(pixels(args.gpu_reference / f"{i}.png"), value["images"][i]) for i in range(3)]
                  for projection, value in (("descriptor", baseline), (args.projection, candidate))}
        frozen_receipt = dict(report_sha256=sha256_file(args.gpu_reference / "report.json"),
            png_sha256=[sha256_file(args.gpu_reference / f"{i}.png") for i in range(3)],
            request_sha256=[sha256_file(args.gpu_reference / f"request-{i}.json") for i in range(3)],
            scope="frozen ordinary GPU PNGs and exact same prompts/seeds/geometry; never a timing denominator")
    result = dict(schema="tc-ffn-projection-complete-request-audit-v1", status="complete", qualification_passed=False,
        case=args.case, projection=args.projection, runtime_prefix_snapshot=args.case == "edit",
        runtime_build_id=manifest["runtime_build_id"], library_sha256=sha256_file(args.library),
        native_source_archive_sha256=sha256_file(args.native_source_archive), script_sha256=sha256_file(Path(__file__)),
        tool_source_archives={str(path): sha256_file(path) for path in args.tool_source_archive}, cohorts=cohorts,
        stock_hot_wall_reduction=1-candidate["receipt"]["hot_median_seconds"]/baseline["receipt"]["hot_median_seconds"],
        incremental_quality=[quality(baseline["images"][i], candidate["images"][i]) for i in range(3)],
        frozen_gpu_quality=frozen, frozen_gpu_receipt=frozen_receipt, historical_descriptor=historical_descriptor,
        limits=["Small non-exclusive-device pilot, not statistical, thermal or semantic image qualification.",
                "One fixed FFN program plus one fixed QKV program per owner; internal MIL MatMul count is not predict count or physical tiling.",
                "Different reduction may change approximate W8/A8/FP16 output; PNG equality is required only for same-recipe replay.",
                "Stock is the performance denominator; timeline is diagnostic. Host API spans do not prove physical INT8 or device overlap."])
    args.output.mkdir(parents=True)
    sheet = Image.new("RGB", (1024, 3*544), "white")
    draw = ImageDraw.Draw(sheet)
    for i in range(3):
        for j, (projection, value) in enumerate((("descriptor", baseline), (args.projection, candidate))):
            draw.text((j*512+8, i*544+8), f"FFN {projection} + QKV512, seed {(73,74,73)[i]}, request{i}", fill="black")
            sheet.paste(Image.fromarray(value["images"][i][:, :, :3]), (j*512, i*544+32))
    sheet.save(args.output / "comparison.png")
    write_json(args.output / "audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
