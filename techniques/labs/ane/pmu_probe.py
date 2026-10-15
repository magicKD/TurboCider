#!/usr/bin/env python3
"""Prepare a tiny owned affine MIL; --device explicitly opts into private PMU API.

No third-party source, NumPy, model downloads, shell evaluation, security changes,
global logs or cache deletion. Run the device CLI through run_experiment.py.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import struct
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmu import classify_stats


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mil(channels, width, kind="affine"):
    shape = f"[1, {channels}, 1, {width}]"
    if kind == "conv":
        return ("program(1.3)\n{\n"
                f"    func main<ios18>(tensor<fp16, {shape}> x) {{\n"
                f'        tensor<fp16, [{channels}, {channels}, 1, 1]> w = const()[name=string("w"), '
                f'val=tensor<fp16, [{channels}, {channels}, 1, 1]>(BLOBFILE(path=string("@model_path/weights.bin"), offset=uint64(64)))];\n'
                '        tensor<int32, [2]> pair = const()[name=string("pair"), val=tensor<int32, [2]>([1, 1])];\n'
                '        tensor<int32, [4]> padding = const()[name=string("padding"), val=tensor<int32, [4]>([0, 0, 0, 0])];\n'
                '        int32 groups = const()[name=string("groups"), val=int32(1)];\n'
                '        string pad_type = const()[name=string("pad_type"), val=string("custom")];\n'
                f'        tensor<fp16, {shape}> y = conv(x=x, weight=w, strides=pair, dilations=pair, '
                'pad=padding, pad_type=pad_type, groups=groups)[name=string("y")];\n'
                "    } -> (y);\n}\n")
    if kind != "affine":
        raise ValueError("unknown owned graph kind")
    return ("program(1.3)\n{\n"
            f"    func main<ios18>(tensor<fp16, {shape}> x) {{\n"
            '        fp16 scale = const()[name=string("scale"), val=fp16(0x1p+1)];\n'
            '        fp16 offset = const()[name=string("offset"), val=fp16(0x1p-2)];\n'
            f'        tensor<fp16, {shape}> scaled = mul(x=x, y=scale)[name=string("scaled")];\n'
            f'        tensor<fp16, {shape}> y = add(x=scaled, y=offset)[name=string("y")];\n'
            "    } -> (y);\n}\n")


def weights_blob(channels):
    """Original writer for the observed 64-byte-header/descriptor FP16 container.

    Format constants are not an Apple compatibility guarantee. No upstream code
    body is embedded. Values are exactly representable synthetic dyadics.
    """
    values = [((out * 7 + inp * 3) % 11 - 5) / 64 for out in range(channels) for inp in range(channels)]
    payload = struct.pack("<" + "e" * len(values), *values)
    header = struct.pack("<II", 1, 2).ljust(64, b"\0")
    descriptor = struct.pack("<IIQQ", 0xDEADBEEF, 1, len(payload), 128).ljust(64, b"\0")
    return header + descriptor + payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", action="store_true", help="explicit private AppleNeuralEngine execution")
    parser.add_argument("--mask", type=int, choices=(0, 2, 15), default=15)
    parser.add_argument("--compile-profiling", action="store_true", help="request enableProfiling at compile/load as an explicit diagnostic")
    parser.add_argument("--kind", choices=("affine", "conv"), default="affine")
    parser.add_argument("--channels", type=int, choices=(32, 64, 128), default=32)
    parser.add_argument("--width", type=int, choices=(32, 64, 128, 256, 512), default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    pins = {name: sha(root / name) for name in ("pmu_probe.py", "pmu_probe.mm")}
    helper_sha = sha(root.parent / "pmu.py")
    report = {"schema": "tc-private-ane-pmu-probe-v1", "status": "starting", "production_qualified": False,
              "counter_semantics_qualified": False, "requested_mask": args.mask,
              "recipe_kind": args.kind,
              "compile_profiling_requested": args.compile_profiling,
              "shape_nchw": [1, args.channels, 1, args.width], "source_sha256": pins,
              "contract_helper_sha256": helper_sha, "device_requested": args.device,
              "recipe": "synthetic finite FP16 affine y=2*x+0.25 or dense dyadic 1x1 conv; full independent numerical check",
              "formal_performance_score": False, "errors": [], "macos": platform.mac_ver()[0],
              "python": platform.python_version(), "native": None, "counter_capture_status": "not-run"}
    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save()
    try:
        if shutil.disk_usage(output).free < 256 * 1024**2:
            raise ValueError("256 MiB disk headroom required; no data deleted")
        (output / "source").mkdir()
        for name in pins:
            shutil.copy2(root / name, output / "source" / name)
        shutil.copy2(root.parent / "pmu.py", output / "source/pmu.py")
        model = output / "model"
        model.mkdir()
        with (model / "model.mil").open("x", encoding="utf-8") as stream:
            stream.write(mil(args.channels, args.width, args.kind))
        report["model_mil_sha256"] = sha(model / "model.mil")
        if args.kind == "conv":
            with (model / "weights.bin").open("xb") as stream:
                stream.write(weights_blob(args.channels))
            report["weights_blob_sha256"] = sha(model / "weights.bin")
        if not args.device:
            report["status"] = "prepared-only-no-device"
        else:
            if platform.system() != "Darwin":
                raise ValueError("private ANE probe requires macOS")
            binary = output / "pmu-probe"
            command = ["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-Wall", "-Wextra",
                       str(root / "pmu_probe.mm"), "-framework", "Foundation", "-framework", "IOSurface",
                       "-framework", "IOKit", "-o", str(binary)]
            now = time.perf_counter_ns()
            with (output / "build.stdout.log").open("x") as stdout, (output / "build.stderr.log").open("x") as stderr:
                built = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=120)
            report.update(build_wall_ns=time.perf_counter_ns() - now, build_returncode=built.returncode)
            save()
            if built.returncode:
                raise ValueError("native build failed; inspect local build log")
            report["binary_sha256"] = sha(binary)
            environment = {key: value for key, value in os.environ.items()
                           if not key.startswith(("ANE_", "ANEFORGE_", "DYLD_", "MPSGRAPH_", "TURBOCIDER_"))}
            with (output / "native.stdout.log").open("x") as stdout, (output / "native.stderr.log").open("x") as stderr:
                child = subprocess.run([str(binary), str(model), str(args.mask), str(args.channels), str(args.width),
                                        str(output / "native.json"), args.kind, str(int(args.compile_profiling))],
                                       env=environment, stdout=stdout, stderr=stderr, timeout=90)
            report["native_returncode"] = child.returncode
            if (output / "native.json").is_file():
                report["native"] = json.loads((output / "native.json").read_text())
                report["raw_native_sha256"] = sha(output / "native.json")
            report["status"] = "executed" if child.returncode == 0 else "native-failed"
            if child.returncode == 0:
                states = [classify_stats(row["stats"])["state"] for row in report["native"]["samples"]]
                report["counter_capture_status"] = ("not-requested" if args.mask == 0 else
                    "raw-only-unqualified" if states and all(state == "raw-available-unqualified" for state in states)
                    else "unavailable-for-this-run")
                if args.mask and report["counter_capture_status"] == "unavailable-for-this-run":
                    report["status"] = "execution-passed-pmu-unavailable"
        report["source_inputs_unchanged"] = (pins == {name: sha(root / name) for name in pins}
                                              and helper_sha == sha(root.parent / "pmu.py"))
        report["model_source_unchanged"] = sha(model / "model.mil") == report["model_mil_sha256"]
        if args.kind == "conv":
            report["model_source_unchanged"] &= sha(model / "weights.bin") == report["weights_blob_sha256"]
        if not report["source_inputs_unchanged"] or not report["model_source_unchanged"]:
            raise ValueError("probe source changed")
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    save()
    print(json.dumps({"status": report["status"], "counter_capture_status": report["counter_capture_status"], "errors": report["errors"]}))
    if report["status"] == "execution-passed-pmu-unavailable":
        return 3
    return 0 if report["status"] in ("executed", "prepared-only-no-device") else 1


if __name__ == "__main__":
    raise SystemExit(main())
