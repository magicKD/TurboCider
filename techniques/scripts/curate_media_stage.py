#!/usr/bin/env python3
"""Read explicitly selected source-bound stage runs, never run a candidate.

Reconstruct complete PNG/synthetic input bytes and orientation hashes, verify
snapshots/binaries/assets/terminal receipts, and independently recompute fixed
pair statistics. No original GPU FP32 tensor re-execution or hardware attestation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs"))
from media_stage import SourceRefused, oriented_rgb, png_rgb, sha, source_bgra, synthetic_rgb, verify


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            value.update(data)
    return value.hexdigest()


def local(root, relative):
    part = Path(relative)
    path = (root / part).resolve()
    if part.is_absolute() or ".." in part.parts or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("bounded explicit local evidence asset required")
    return path


def curate(args):
    root, repo, mlx = args.root.resolve(), ROOT.parent, args.mlx_root.resolve()
    manifest_path = args.native_manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["inputs"]["sources"]:
        if digest(local(repo, item["path"])) != item["sha256"]:
            raise ValueError("current native source differs from explicit source manifest")
    file_meta, file_rgb = png_rgb(args.source_png)
    refusals = {}
    for path in args.rejected_source:
        try:
            png_rgb(path)
        except SourceRefused as error:
            refusals[error.summary["file_sha256"]] = {"kind": "nonopaque-source-before-native", "reason": str(error), **error.summary,
                                                       "native_work_started_for_this_source": False}
        else:
            raise ValueError("explicit refused-source must actually be refused")
    selected = args.run + args.build_failure
    if not args.run or len(selected) != len(set(selected)):
        raise ValueError("unique explicit runs/failures required")
    result = {"schema": "tc-media-stage-evidence-v1", "curator_sha256": digest(Path(__file__)),
              "contract_sha256": digest(ROOT / "labs/media_stage.py"), "production_qualified": False,
              "whole_request_qualified": False, "physical_overlap_qualified": False, "raw_pixels_or_paths_exported": False,
              "gpu_fp32_outputs_reexecuted_by_verifier": False, "independent_png_sample_bytes_verified": True,
              "historical_wrapper_oracle_min_is_not_a_deployable_policy": True,
              "source_manifest_sha256": digest(manifest_path), "source_manifest_runtime_build_id": manifest["runtime_build_id"],
              "native_source_files_verified": len(manifest["inputs"]["sources"]), "runs": [], "build_failures": []}
    checked_assets = set()
    for name in selected:
        path = local(root, name + "/report.json")
        document = json.loads(path.read_text())
        failed = name in args.build_failure
        launch_path = local(root, name + "-launch/receipt.json")
        terminal = json.loads(launch_path.read_text())
        if terminal.get("owned_process_exit_confirmed") is not True or terminal.get("returncode") != int(failed):
            raise ValueError("owned wrapper terminal receipt differs")
        if document["source_manifest_sha256"] != digest(manifest_path) or document["file_source"] != file_meta:
            raise ValueError("complete source PNG or native source identity differs")
        for record in document["source_admission_rejections"]:
            if record != refusals.get(record["file_sha256"]):
                raise ValueError("source refusal is not independently reproduced")
        if len(document["source_admission_rejections"]) != len(refusals):
            raise ValueError("all explicit source refusals retained")
        for filename, expected in document["source_sha256"].items():
            if digest(local(path.parent, "source/" + filename)) != expected:
                raise ValueError("per-run collection source snapshot differs")
        for filename, expected in document["product_sources"].items():
            if digest(local(repo, filename)) != expected:
                raise ValueError("actual linked decoder or product stage recipe differs")
        assets = document["mlx_assets"]
        asset_id = sha(json.dumps(assets, sort_keys=True).encode())
        if asset_id not in checked_assets:
            inventory = [str(p.relative_to(mlx)) for p in sorted((mlx / "include").rglob("*")) if p.is_file()]
            if inventory != [row["path"] for row in assets["headers"]]:
                raise ValueError("complete installed MLX header inventory differs")
            for item in assets["headers"]:
                if digest(local(mlx, item["path"])) != item["sha256"]:
                    raise ValueError("installed MLX header differs")
            for filename, expected in assets["libraries"].items():
                if digest(local(mlx, "lib/" + filename)) != expected:
                    raise ValueError("installed MLX library differs")
            checked_assets.add(asset_id)
        config_path = local(path.parent, "config.json")
        config = json.loads(config_path.read_text())
        if digest(config_path) != document["raw_config_sha256"] or Path(config["source_png"]).resolve() != args.source_png.resolve():
            raise ValueError("actual config source binding differs")
        item = {"id": name, "raw_report_sha256": digest(path), "supervisor_sha256": digest(launch_path),
                "owned_wrapper_exit_confirmed": True, "report": document}
        if failed:
            log = local(path.parent, "build.stderr.log")
            text = log.read_text()
            if (document["status"] != "failed" or document.get("build_returncode") != 1 or document["native"] is not None
                    or "Undefined symbols" not in text or "_CGImageGetWidth" not in text or (path.parent / "native.json").exists()):
                raise ValueError("retain the explicit CoreGraphics link failure, not a hardware result")
            item.update(build_stderr_sha256=digest(log), independent_validation={"device_work_started": False, "cause": "missing explicit CoreGraphics linkage", "hardware_unsupported": False})
            result["build_failures"].append(item)
            continue
        if (document["status"] != "measured" or document["native_returncode"] != 0 or document["native_child_exit_confirmed"] is not True
                or not all(document.get(key) is True for key in ("file_source_unchanged", "source_inputs_unchanged", "product_sources_unchanged", "mlx_assets_unchanged"))):
            raise ValueError("complete normal source/lifecycle protocol required")
        native_path = local(path.parent, "native.json")
        if (digest(local(path.parent, "media-stage")) != document["binary_sha256"] or digest(native_path) != document["raw_native_sha256"]
                or json.loads(native_path.read_text()) != document["native"] or document["native"]["source_config_sha256"] != digest(config_path)):
            raise ValueError("actual executable/config/native content differs")
        proof = verify(document["native"])
        # Historical wrapper retained its then-defined comparisons, including a per-trial oracle min.
        # Check every old value without rewriting it; fixed-arm comparisons are explicitly added here.
        old = document["comparisons"]
        if {key: proof[key] for key in old if key != "component_comparisons"} != {key: old[key] for key in old if key != "component_comparisons"}:
            raise ValueError("historical complete-call/control summary differs")
        if len(old["component_comparisons"]) != len(proof["component_comparisons"]):
            raise ValueError("all original scenario comparisons required")
        for original, current in zip(old["component_comparisons"], proof["component_comparisons"]):
            if any(current.get(key) != value for key, value in original.items()):
                raise ValueError("historical paired arithmetic differs")
        seed = document["seed"]
        if (config["seed"] != seed or config["measured_groups"] != document["measured_groups"]
                or document["native"]["seed"] != seed or document["native"]["measured_groups"] != document["measured_groups"]):
            raise ValueError("actual seed/group config differs")
        for kind, rgb, width, height in (("file", file_rgb, file_meta["width"], file_meta["height"]),
                                          ("synthetic", synthetic_rgb(seed=seed), 513, 257)):
            ids = document["source_identities"][kind]
            bgra = [sha(source_bgra(rgb, width, height, g)[1]) for g in (0, 1)]
            if ids["bgra_sha256_by_generation"] != bgra:
                raise ValueError("full logical/padded input differs from independent construction")
            for code in range(8):
                hashes = [sha(oriented_rgb(rgb, width, height, code, g)[2]) for g in (0, 1)]
                scenario = next(s for s in document["native"]["scenarios"] if s["source"] == kind and s["orientation"] == code)
                if (ids["width"] != width or ids["height"] != height or scenario["source_wh"] != [width, height]
                        or ids["rgb_sha256_by_orientation"][str(code)] != hashes or scenario["rgb_sha256_by_generation"] != hashes or scenario["bgra_sha256_by_generation"] != bgra):
                    raise ValueError("all orientations/generations differ from independent forward scatter")
        item.update(independent_validation=proof, full_source_bytes_independently_verified=True)
        result["runs"].append(item)
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
    if any(value in encoded for value in ("/Users/", "/home/", "/private/var/folders/", "source_png")):
        raise ValueError("public evidence contains a raw path/config field")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source-png", "native-manifest", "mlx-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--rejected-source", type=Path, action="append", default=[])
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--build-failure", action="append", default=[])
    args = parser.parse_args()
    document = curate(args)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"runs": len(document["runs"]), "build_failures": len(document["build_failures"]),
                      "complete_stage_calls": sum(r["independent_validation"]["completed_stage_calls"] for r in document["runs"])}))


if __name__ == "__main__":
    main()
