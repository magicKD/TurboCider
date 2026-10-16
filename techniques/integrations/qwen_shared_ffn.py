#!/usr/bin/env python3
"""Explicit real-source Qwen five-arm FFN benchmark or private-port inspection.

Existing checkpoint/captures/fixture and prior qualified MIL/blob stay read-only.
New-graph mode compiles an owned APFS clone; explicit prepared replay omits the
compile selector. No ordinary-copy fallback, download, product rebuild, cache
deletion, default change or security bypass.
"""
import argparse
import ast
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import struct
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from curate_qwen_private_ffn import native_prefix_hashes
sys.path.insert(0, str(ROOT / "labs/metal"))
from mlx_adoption_probe import assets


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            result.update(data)
    return result.hexdigest()


def local(root, name):
    value = Path(name); path = (root / value).resolve()
    if value.is_absolute() or ".." in value.parts or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("explicit bounded existing source required")
    return path


def span(path, key, shape, digest):
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("short safetensors header")
        size = struct.unpack("<Q", raw)[0]
        if not 2 <= size <= 16 * 1024**2 or 8 + size > path.stat().st_size:
            raise ValueError("bounded safetensors header required")
        node = json.loads(stream.read(size))[key]
        begin, end = node["data_offsets"]
        if (node["dtype"] != "BF16" or node["shape"] != shape or type(begin) is not int or type(end) is not int
                or begin < 0 or end - begin != math.prod(shape) * 2 or 8 + size + end > path.stat().st_size):
            raise ValueError("original BF16 span/shape differs")
        result = hashlib.sha256(); stream.seek(8 + size + begin); remaining = end - begin
        while remaining:
            data = stream.read(min(remaining, 1 << 20))
            if not data:
                raise ValueError("short tensor")
            result.update(data); remaining -= len(data)
    if result.hexdigest() != digest:
        raise ValueError("original BF16 tensor content differs")
    return {"file": str(path.resolve()), "offset": 8 + size + begin, "bytes": end - begin, "sha256": digest, "shape": shape, "dtype": "BF16"}


def reference_span(path):
    with path.open("rb") as stream:
        if stream.read(8) != b"\x93NUMPY\x01\x00":
            raise ValueError("explicit NPYv1 reference required")
        size = struct.unpack("<H", stream.read(2))[0]
        if not 1 <= size <= 65536:
            raise ValueError("reference header cap")
        header = ast.literal_eval(stream.read(size).decode("ascii")); offset = stream.tell()
    if header != {"descr": "<f4", "fortran_order": False, "shape": (1024, 4096)} or path.stat().st_size != offset + 1024 * 4096 * 4:
        raise ValueError("complete original FP32-encoded BF16 reference required")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(offset)
        for data in iter(lambda: stream.read(1 << 20), b""):
            result.update(data)
    return {"file": str(path), "offset": offset, "bytes": 1024 * 4096 * 4, "sha256": result.hexdigest(), "shape": [1024, 4096], "dtype": "F32"}


def clone_constant(source, destination):
    if source.is_symlink() or destination.exists() or destination.is_symlink():
        raise ValueError("fresh owned clone target and ordinary source file required")
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    function = libc.clonefile; function.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32); function.restype = ctypes.c_int
    if function(os.fsencode(source), os.fsencode(destination), 1):
        raise OSError(ctypes.get_errno(), "APFS clone failed; ordinary-copy fallback disabled")
    if (source.stat().st_ino, source.stat().st_dev) == (destination.stat().st_ino, destination.stat().st_dev):
        raise ValueError("clone must not be a hard link")


def prepare(args, output, report):
    fixture_path = args.fixture.resolve(); fixture = json.loads(fixture_path.read_text()); root = fixture_path.parent
    if (fixture.get("schema") != "tc-real-ffn-fixture-v1" or fixture.get("shape_mhf") != [1024, 4096, 12288]
            or fixture.get("source_layer") != 0 or fixture.get("loras") != []):
        raise ValueError("complete actual base layer0 fixture required")
    for name, digest in fixture["files"].items():
        if sha(local(root, name)) != digest:
            raise ValueError("fixture content differs")
    prior_root = args.prior_run.resolve(); prior = json.loads(local(prior_root, "report.json").read_text())
    if prior.get("status") != "measured" or prior.get("share") != 2048 or prior.get("shape_mhf") != fixture["shape_mhf"]:
        raise ValueError("explicit completed full-consumer prior run required")
    for key in ("original_gate_up_tensor_sha256", "original_down_tensor_sha256"):
        if prior["original_weight_sha256"][key] != fixture[key]:
            raise ValueError("prior native constant source differs")
    if native_prefix_hashes(root, fixture, 2048) != prior["native_weight_sha256"]:
        raise ValueError("all paired FP16 prefix bytes differ from independently checked fixture")
    sources = [local(prior_root, "ane-model/model.mil"), local(prior_root, "ane-model/weights.bin")]
    if [sha(path) for path in sources] != [prior["private_mil_sha256"], prior["private_weights_blob_sha256"]]:
        raise ValueError("prior MIL/blob content differs")
    checkpoint, capture = args.checkpoint.resolve(), args.capture_root.resolve()
    weights = [span(checkpoint, "transformer_blocks.0.img_mlp.gate_up.weight", [24576, 4096], fixture["original_gate_up_tensor_sha256"]),
               span(checkpoint, "transformer_blocks.0.img_mlp.out.weight", [4096, 12288], fixture["original_down_tensor_sha256"])]
    receipt_path = local(capture, "receipt.json")
    if sha(receipt_path) != fixture["capture_receipt_sha256"]:
        raise ValueError("capture receipt changed")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "captured" or receipt.get("loras") != []:
        raise ValueError("completed base forced-GPU capture required")
    points = []
    for name in args.point:
        selected = [point for point in fixture["points"] if point["id"] == name]
        if len(selected) != 1:
            raise ValueError("unique actual selected point required")
        info = selected[0]; basename = f"layer-0-step-{info['step']}.safetensors"
        records = [row for row in receipt["records"] if Path(row["path"]).name == basename]
        if len(records) != 1:
            raise ValueError("unique original activation required")
        activation = local(capture, records[0]["path"])
        if sha(activation) != info["capture_file_sha256"]:
            raise ValueError("original activation file differs")
        points.append({"id": name, "input": span(activation, "input", [1, 1024, 4096], info["original_input_tensor_sha256"]),
                       "reference": reference_span(local(root, info["original_reference"]))})
    if len(points) != len({point["id"] for point in points}):
        raise ValueError("duplicate source points refused")
    if args.mode == "benchmark":
        if len(points) < 2:
            raise ValueError("alternate at least two complete actual timesteps; no constant-input stale-output loophole")
        sys.path.insert(0, str(ROOT / "labs/ane"))
        from qwen_witness import selected
        from real_bf16_source import load as original
        import numpy as np
        inputs = [{"id": point["id"], "x": np.load(local(root, next(item["input"] for item in fixture["points"] if item["id"] == point["id"])),
                    allow_pickle=False, mmap_mode="r")} for point in points]
        gu, down = original(checkpoint, capture, fixture, inputs, np)
        prefix_gu = np.concatenate((gu[:2048], gu[12288:14336]), axis=0)
        prefix_down = np.ascontiguousarray(down[:, :2048])
        for point, values in zip(points, inputs):
            full_rows, full_cols, full_values, _ = selected(values["x_bf16_source"], gu, down, np)
            prefix_rows, prefix_cols, prefix_values, bounds = selected(values["x_bf16_source"], prefix_gu, prefix_down, np)
            point["oracle"] = {"full": {"rows": full_rows, "cols": full_cols, "values": full_values.tolist()},
                               "prefix": {"rows": prefix_rows, "cols": prefix_cols, "values": prefix_values.tolist()}}
            point["independent_prefix_domain"] = bounds
    model = output / "model"; model.mkdir()
    shutil.copy2(sources[0], model / "model.mil")
    clone_constant(sources[1], model / "weights.bin")
    if [sha(model / name) for name in ("model.mil", "weights.bin")] != [sha(path) for path in sources]:
        raise ValueError("owned cloned graph content differs")
    report.update(fixture_sha256=sha(fixture_path), prior_report_sha256=sha(prior_root / "report.json"),
                  private_mil_sha256=sha(sources[0]), private_blob_sha256=sha(sources[1]),
                  native_weight_sha256=prior["native_weight_sha256"], clone_not_hardlink=True,
                  capture_receipt_sha256=fixture["capture_receipt_sha256"], source_layer=0, loras=[], shape_mhf=fixture["shape_mhf"], share=2048)
    return {"schema": "tc-qwen-shared-manifest-v1", "mode": args.mode, "model_directory": str(model),
            "weights": weights, "points": points, "shape_mhf": fixture["shape_mhf"], "share": 2048, "measured_groups": args.repeats,
            "gates": {"relative_l2_max": .03, "cosine_min": .999}}


def comparisons(native):
    measured = {i: {} for i in range(native["measured_groups"])}
    for sample in native["samples"]:
        if sample["numeric_pass"] is not True:
            raise ValueError("no ratios after a numerical failure")
        if sample["phase"] == "measured":
            measured[sample["trial"]][sample["arm"]] = sample["wall_ns"]
    result = []
    for a, b in (("gpu", "copy-serial"), ("gpu", "copy-hybrid"), ("gpu", "shared-serial"), ("gpu", "shared-hybrid"),
                 ("copy-hybrid", "shared-hybrid"), ("shared-serial", "shared-hybrid")):
        values = [item[a] / item[b] for item in measured.values()]
        rng = random.Random(73); boot = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(2000))
        result.append({"baseline": a, "candidate": b, "paired_ratios": values, "median_paired_host_ratio": statistics.median(values),
                       "bootstrap_95pct": [boot[49], boot[1949]], "production_qualified": False})
    return result


def replay_prepared(args, report):
    root = args.prepared_run.resolve()
    if not root.is_relative_to((ROOT.parent / "outputs").resolve()):
        raise ValueError("only explicit owned prepared-run assets may be reused")
    before = json.loads(local(root, "report.json").read_text()); path = local(root, "manifest.json")
    if before.get("prior_report_sha256") != sha(local(args.prior_run.resolve(), "report.json")):
        raise ValueError("prepared prefix lineage differs from explicit prior run")
    if (before.get("status") != "measured" or before.get("raw_manifest_sha256") != sha(path)
            or before.get("source_inputs_unchanged") is not True or before.get("original_spans_unchanged") is not True
            or before.get("cloned_graph_unchanged") is not True or before.get("share") != 2048
            or before.get("shape_mhf") != [1024,4096,12288] or before.get("fixture_sha256") != sha(args.fixture.resolve())):
        raise ValueError("previous complete source-verified prepared run required")
    manifest = json.loads(path.read_text())
    if manifest.get("mode") != "benchmark" or manifest.get("schema") != "tc-qwen-shared-manifest-v1":
        raise ValueError("complete prepared actual-source benchmark manifest required")
    model = Path(manifest["model_directory"]).resolve()
    if model != (root / "model").resolve():
        raise ValueError("prepared graph must be the already-owned immutable source directory")
    if sha(model / "model.mil") != before["private_mil_sha256"] or sha(model / "weights.bin") != before["private_blob_sha256"]:
        raise ValueError("prepared MIL/blob changed")
    if any(Path(node["file"]).resolve() != args.checkpoint.resolve() for node in manifest["weights"]):
        raise ValueError("prepared checkpoint identity differs")
    points = [next((value for value in manifest["points"] if value["id"] == name), None) for name in args.point]
    if len(points) < 2 or any(value is None for value in points) or len(args.point) != len(set(args.point)):
        raise ValueError("two or more distinct prepared actual points required")
    if any(not Path(point["input"]["file"]).resolve().is_relative_to(args.capture_root.resolve()) for point in points):
        raise ValueError("prepared input escaped explicit capture root")
    manifest.update(mode="benchmark", skip_compile=True, measured_groups=args.repeats, points=points)
    for key in ("fixture_sha256", "prior_report_sha256", "private_mil_sha256", "private_blob_sha256", "native_weight_sha256",
                "capture_receipt_sha256", "source_layer", "loras", "shape_mhf", "share"):
        report[key] = before[key]
    report.update(prepared_source_report_sha256=sha(root / "report.json"), prepared_source_manifest_sha256=sha(path),
                  graph_asset_policy="read-only owned prepared assets; compile selector omitted, no compile fallback")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", action="store_true")
    parser.add_argument("--mode", choices=("inspect", "benchmark"), default="inspect")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--prepared-run", type=Path, help="explicit already-qualified owned graph/manifest; read-only replay omits compile selector")
    parser.add_argument("--mlx-root", type=Path, default=Path(sysconfig.get_paths()["purelib"]) / "mlx")
    parser.add_argument("--point", action="append", required=True)
    parser.add_argument("--repeats", type=int, choices=(10, 30), default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.prepared_run and args.mode != "benchmark":
        parser.error("prepared-run replay requires explicit --mode benchmark")
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "tc-qwen-shared-probe-v1", "status": "starting", "mode": args.mode, "device_requested": args.device,
              "production_qualified": False, "physical_overlap_qualified": False, "runtime_internal_copy_behavior": "unknown",
              "errors": [], "native": None, "source_sha256": {}, "comparisons": None}
    own = [Path(__file__), Path(__file__).with_suffix(".mm"), Path(__file__).with_name("qwen_shared_abi.h"), Path(__file__).with_name("qwen_shared_compute.h"),
           ROOT / "scripts/curate_qwen_private_ffn.py", ROOT / "labs/ane/qwen_witness.py", ROOT / "labs/metal/real_bf16_source.py",
           ROOT / "labs/metal/mlx_adoption_probe.py"]

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    try:
        minimum = 128 if args.prepared_run else 256
        report["admission"] = {"kind": "read-only-prepared-load" if args.prepared_run else "new-clone-and-compile", "minimum_free_mib": minimum}
        if platform.system() != "Darwin" or shutil.disk_usage(output).free < minimum * 1024**2:
            raise ValueError(f"macOS and {minimum} MiB free space required for selected admission; nothing deleted")
        (output / "source").mkdir()
        for path in own:
            shutil.copy2(path, output / "source" / path.name); report["source_sha256"][path.name] = sha(path)
        manifest = replay_prepared(args, report) if args.prepared_run else prepare(args, output, report)
        with (output / "manifest.json").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        report["raw_manifest_sha256"] = sha(output / "manifest.json")
        if not args.device:
            report["status"] = "prepared-only-no-device"
        else:
            mlx = args.mlx_root.resolve(); binary = output / "qwen-shared"
            before_assets = assets(mlx); report["mlx_assets"] = before_assets
            report["macos_product_version"] = subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip()
            report["macos_build"] = subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip()
            report["sdk_version"] = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-version"], text=True).strip()
            command = ["xcrun", "clang++", "-std=c++20", "-O2", "-fobjc-arc", "-Wall", "-Wextra", "-mmacosx-version-min=26.2",
                       "-isystem", str(mlx / "include"), str(output / "source/qwen_shared_ffn.mm"), "-L" + str(mlx / "lib"),
                       "-lmlx", "-ljaccl", "-Wl,-rpath," + str(mlx / "lib"), "-framework", "Foundation", "-framework", "Metal",
                       "-framework", "IOSurface", "-o", str(binary)]
            with (output / "build.stdout.log").open("x") as stdout, (output / "build.stderr.log").open("x") as stderr:
                built = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=120)
            report["build_returncode"] = built.returncode; save()
            if built.returncode:
                raise ValueError("native build failed; inspect owned log")
            report["binary_sha256"] = sha(binary)
            environment = {key: value for key, value in os.environ.items() if not key.startswith(("ANE_", "ANEFORGE_", "DYLD_", "TURBOCIDER_", "MPSGRAPH_", "MLX_DISABLE_COMPILE"))}
            with (output / "native.stdout.log").open("x") as stdout, (output / "native.stderr.log").open("x") as stderr:
                child = subprocess.run([str(binary), str(output / "manifest.json"), str(output / "native.json")],
                                       stdout=stdout, stderr=stderr, env=environment, timeout=180)
            report["native_returncode"] = child.returncode
            if (output / "native.json").is_file():
                report["native"] = json.loads((output / "native.json").read_text()); report["raw_native_sha256"] = sha(output / "native.json")
            report["mlx_assets_unchanged"] = before_assets == assets(mlx)
            report["status"] = report["native"]["status"] if child.returncode == 0 else "native-failed"
            if not report["mlx_assets_unchanged"]:
                raise ValueError("installed MLX assets changed")
            if report["status"] == "measured":
                report["comparisons"] = comparisons(report["native"])
        report["source_inputs_unchanged"] = all(sha(path) == report["source_sha256"][path.name] for path in own)
        graph = Path(manifest["model_directory"])
        report["cloned_graph_unchanged"] = sha(graph / "model.mil") == report["private_mil_sha256"] and sha(graph / "weights.bin") == report["private_blob_sha256"]
        report["original_spans_unchanged"] = True
        for node in manifest["weights"] + [item[key] for item in manifest["points"] for key in ("input", "reference")]:
            with Path(node["file"]).open("rb") as stream:
                stream.seek(node["offset"]); digest = hashlib.sha256(); remaining = node["bytes"]
                while remaining:
                    block = stream.read(min(remaining, 1 << 20))
                    if not block: raise ValueError("source span became short")
                    digest.update(block); remaining -= len(block)
            report["original_spans_unchanged"] &= digest.hexdigest() == node["sha256"]
        if not report["source_inputs_unchanged"] or not report["cloned_graph_unchanged"] or not report["original_spans_unchanged"]:
            raise ValueError("source graph changed")
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"; report["comparisons"] = None; report["errors"].append({"type": type(error).__name__, "message": str(error)})
    save(); print(json.dumps({"status": report["status"], "errors": report["errors"]}))
    return 0 if report["status"] in ("prepared-only-no-device", "ports-inspected-no-execution", "measured") else 1


if __name__ == "__main__":
    raise SystemExit(main())
