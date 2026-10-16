#!/usr/bin/env python3
"""Explicit source-bound BGRA stage: product loop, public vImage and fused GPU.

Default is preparation only. --device builds a small owned client plus the real
product PE decoder/common utility, NOT the product library or a model. Existing
native source manifest/MLX tree and explicit local source PNG are required.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

from capture_qwen_ffn import source_match

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "labs"))
sys.path.insert(0, str(ROOT / "labs/metal"))
from media_stage import SourceRefused, oriented_rgb, png_rgb, sha, source_bgra, synthetic_rgb, verify
from mlx_adoption_probe import assets


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            value.update(part)
    return value.hexdigest()


def identities(rgb, width, height):
    return {"width": width, "height": height,
            "bgra_sha256_by_generation": [sha(source_bgra(rgb, width, height, g)[1]) for g in (0, 1)],
            "rgb_sha256_by_orientation": {str(code): [sha(oriented_rgb(rgb, width, height, code, g)[2]) for g in (0, 1)] for code in range(8)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", action="store_true")
    parser.add_argument("--source-png", type=Path, required=True)
    parser.add_argument("--rejected-source", type=Path, action="append", default=[])
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--mlx-root", type=Path, default=Path(sysconfig.get_paths()["purelib"]) / "mlx")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repeats", type=int, choices=(10, 30), default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.seed <= 10000:
        parser.error("bounded seed required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "tc-media-stage-wrapper-v1", "status": "starting", "device_requested": args.device,
              "production_qualified": False, "whole_request_qualified": False, "physical_overlap_qualified": False,
              "seed": args.seed, "measured_groups": args.repeats, "native": None, "comparisons": None,
              "source_admission_rejections": [], "native_child_exit_confirmed": False, "errors": []}

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    try:
        # New small-client operation: <=6MiB build/report estimate plus original 128MiB reserve.
        report["admission_minimum_free_mib"] = 140
        if shutil.disk_usage(output).free < 140 * 1024**2:
            raise ValueError("140MiB small-client admission required; no model/capture admission gate changed")
        for source in args.rejected_source:
            try:
                png_rgb(source)
            except SourceRefused as error:
                report["source_admission_rejections"].append({"kind": "nonopaque-source-before-native", "reason": str(error), **error.summary,
                                                             "native_work_started_for_this_source": False})
            else:
                raise ValueError("explicit rejected-source was not refused; do not relabel an accepted source")
        image, rgb = png_rgb(args.source_png)
        report["file_source"] = image
        native_manifest = args.native_manifest.resolve()
        manifest = json.loads(native_manifest.read_text())
        report["native_source_files_verified"] = source_match(manifest)
        report["source_manifest_sha256"] = digest(native_manifest)
        # Manifest identifies current source only; no product binary is linked or attested.
        report["source_manifest_runtime_build_id"] = manifest["runtime_build_id"]
        mlx = args.mlx_root.resolve()
        before_assets = assets(mlx)
        report["mlx_assets"] = before_assets
        identities_by_source = {"file": identities(rgb, image["width"], image["height"]),
                                "synthetic": identities(synthetic_rgb(seed=args.seed), 513, 257)}
        report["source_identities"] = identities_by_source
        config = {"source_png": str(args.source_png.resolve()), "source_file_sha256": image["file_sha256"],
                  "expected_file_rgb_sha256": image["rgb_sha256"], "source_identities": identities_by_source,
                  "seed": args.seed, "measured_groups": args.repeats}
        with (output / "config.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(config, indent=2) + "\n")
        report["raw_config_sha256"] = digest(output / "config.json")
        own = Path(__file__).resolve()
        sources = [own, own.with_suffix(".mm"), ROOT / "labs/media_stage.py", own.with_name("capture_qwen_ffn.py"), ROOT / "labs/metal/mlx_adoption_probe.py"]
        (output / "source").mkdir()
        report["source_sha256"] = {}
        for source in sources:
            shutil.copy2(source, output / "source" / source.name)
            report["source_sha256"][source.name] = digest(source)
        report["product_sources"] = {str(path.relative_to(REPO)): digest(path) for path in
            (REPO / "native/media/pe_image.mm", REPO / "native/media/rgb_video_reader.mm", REPO / "native/media/video_input.mm",
             REPO / "native/models/seedvr2/media.cpp", REPO / "native/core/common.cpp")}
        if not args.device:
            report["status"] = "prepared-only-no-device"
        else:
            report["macos_product_version"] = subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip()
            report["macos_build"] = subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip()
            report["sdk_version"] = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-version"], text=True).strip()
            report["compiler_version"] = subprocess.check_output(["xcrun", "clang++", "--version"], text=True).splitlines()[0]
            binary = output / "media-stage"
            command = ["xcrun", "clang++", "-std=c++20", "-O2", "-fobjc-arc", "-Wall", "-Wextra", "-mmacosx-version-min=26.2",
                       "-I", str(REPO), "-I", str(REPO / "native"), "-I", str(REPO / "native/core"), "-I", str(REPO / "bindings/c/include"),
                       "-isystem", str(mlx / "include"), str(output / "source/media_stage_probe.mm"),
                       str(REPO / "native/media/pe_image.mm"), str(REPO / "native/core/common.cpp"), "-L" + str(mlx / "lib"),
                       "-lmlx", "-ljaccl", "-framework", "Foundation", "-framework", "Metal", "-framework", "Accelerate", "-framework", "ImageIO", "-framework", "CoreGraphics",
                       "-Wl,-rpath," + str(mlx / "lib"), "-o", str(binary)]
            with (output / "build.stdout.log").open("x") as stdout, (output / "build.stderr.log").open("x") as stderr:
                built = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=120)
            report["build_returncode"] = built.returncode
            save()
            if built.returncode:
                raise ValueError("small native stage build failed; original owned logs retained")
            report["binary_sha256"] = digest(binary)
            if shutil.disk_usage(output).free < 134 * 1024**2:
                raise ValueError("128MiB reserve plus estimated native metadata required before GPU work")
            temporary = output / "task-tmp"
            temporary.mkdir(mode=0o700)
            environment = {k: v for k, v in os.environ.items() if not k.startswith(("TURBOCIDER_", "ANE_", "ANEFORGE_", "DYLD_", "MLX_DISABLE_COMPILE"))}
            environment["TMPDIR"] = str(temporary)
            with (output / "native.stdout.log").open("x") as stdout, (output / "native.stderr.log").open("x") as stderr:
                child = subprocess.run([str(binary), str(output / "config.json"), str(output / "native.json")], env=environment,
                                       stdout=stdout, stderr=stderr, timeout=180)
            report["native_returncode"] = child.returncode
            report["native_child_exit_confirmed"] = True
            if (output / "native.json").is_file():
                report["native"] = json.loads((output / "native.json").read_text())
                report["raw_native_sha256"] = digest(output / "native.json")
            report["status"] = "native-failed" if child.returncode else "measured"
            if child.returncode == 0:
                report["comparisons"] = verify(report["native"])
        report["file_source_unchanged"] = digest(args.source_png) == image["file_sha256"]
        report["source_inputs_unchanged"] = all(digest(path) == report["source_sha256"][path.name] for path in sources)
        report["native_source_files_verified_after"] = source_match(manifest)
        report["product_sources_unchanged"] = all(digest(REPO / path) == value for path, value in report["product_sources"].items())
        report["mlx_assets_unchanged"] = assets(mlx) == before_assets
        if not all(report[key] for key in ("file_source_unchanged", "source_inputs_unchanged", "product_sources_unchanged", "mlx_assets_unchanged")):
            raise ValueError("source/product/assets changed; no comparison qualified")
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["comparisons"] = None
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    save()
    print(json.dumps({"status": report["status"], "complete_stage_calls": (report["comparisons"] or {}).get("completed_stage_calls", 0), "errors": report["errors"]}))
    return 0 if report["status"] in ("measured", "prepared-only-no-device") else 1


if __name__ == "__main__":
    raise SystemExit(main())
