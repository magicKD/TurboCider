#!/usr/bin/env python3
"""Prepare an owned affine MIL; explicitly opt into Metal/private-ANE backing tests.

Standard library only. No download, upstream import, security change or cleanup.
Run --device through scripts/run_experiment.py with a fresh ignored output dir.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mil(channels, width):
    shape = f"[1, {channels}, 1, {width}]"
    return ("program(1.3)\n{\n"
            f"    func main<ios18>(tensor<fp16, {shape}> x) {{\n"
            '        fp16 scale = const()[name=string("scale"), val=fp16(0x1p+1)];\n'
            '        fp16 offset = const()[name=string("offset"), val=fp16(0x1p-2)];\n'
            f'        tensor<fp16, {shape}> scaled = mul(x=x, y=scale)[name=string("scaled")];\n'
            f'        tensor<fp16, {shape}> y = add(x=scaled, y=offset)[name=string("y")];\n'
            "    } -> (y);\n}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", action="store_true")
    parser.add_argument("--channels", type=int, choices=(32, 64), default=32)
    parser.add_argument("--width", type=int, choices=(33, 65), default=33)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repeats", type=int, choices=(8, 24), default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.seed <= 10000:
        parser.error("seed must be 0..10000")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    pins = {name: sha(root / name) for name in ("shared_surface_probe.py", "shared_surface_probe.mm")}
    report = {"schema": "tc-private-shared-probe-v1", "status": "starting", "errors": [],
              "device_requested": args.device, "shape_nchw": [1, args.channels, 1, args.width],
              "seed": args.seed, "measured_groups": args.repeats, "source_sha256": pins,
              "production_qualified": False, "physical_overlap_qualified": False,
              "runtime_internal_copy_behavior": "unknown", "native": None,
              "boundary": "GPU producer encode through GPU consumer completion; feature copies included for copy control; validation excluded",
              "macos": platform.mac_ver()[0], "python": platform.python_version()}

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    try:
        if shutil.disk_usage(output).free < 128 * 1024**2:
            raise ValueError("128 MiB headroom required; nothing deleted")
        (output / "source").mkdir()
        for name in pins:
            shutil.copy2(root / name, output / "source" / name)
        model = output / "model"
        model.mkdir()
        with (model / "model.mil").open("x", encoding="utf-8") as stream:
            stream.write(mil(args.channels, args.width))
        report["model_mil_sha256"] = sha(model / "model.mil")
        if not args.device:
            report["status"] = "prepared-only-no-device"
        else:
            if platform.system() != "Darwin":
                raise ValueError("macOS required; no fallback")
            report["sdk_version"] = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-version"], text=True).strip()
            report["compiler_version"] = subprocess.check_output(["xcrun", "clang++", "--version"], text=True).splitlines()[0]
            binary = output / "shared-probe"
            command = ["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-Wall", "-Wextra",
                       str(root / "shared_surface_probe.mm"), "-framework", "Foundation", "-framework", "Metal",
                       "-framework", "IOSurface", "-o", str(binary)]
            start = time.perf_counter_ns()
            with (output / "build.stdout.log").open("x") as stdout, (output / "build.stderr.log").open("x") as stderr:
                child = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=120)
            report.update(build_wall_ns=time.perf_counter_ns() - start, build_returncode=child.returncode)
            save()
            if child.returncode:
                raise ValueError("native build failed; inspect owned log")
            report["binary_sha256"] = sha(binary)
            environment = {key: value for key, value in os.environ.items()
                           if not key.startswith(("ANE_", "ANEFORGE_", "DYLD_", "MPSGRAPH_", "TURBOCIDER_"))}
            with (output / "native.stdout.log").open("x") as stdout, (output / "native.stderr.log").open("x") as stderr:
                child = subprocess.run([str(binary), str(model), str(args.channels), str(args.width), str(args.seed),
                                        str(args.repeats), str(output / "native.json"), "0"], env=environment,
                                       stdout=stdout, stderr=stderr, timeout=90)
            report["native_returncode"] = child.returncode
            if (output / "native.json").is_file():
                report["native"] = json.loads((output / "native.json").read_text())
                report["raw_native_sha256"] = sha(output / "native.json")
            report["status"] = "backing-and-consumer-validated" if child.returncode == 0 else "native-failed"
        report["source_inputs_unchanged"] = pins == {name: sha(root / name) for name in pins}
        report["model_source_unchanged"] = sha(model / "model.mil") == report["model_mil_sha256"]
        if not report["source_inputs_unchanged"] or not report["model_source_unchanged"]:
            raise ValueError("source changed during probe")
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
    save()
    print(json.dumps({"status": report["status"], "errors": report["errors"]}))
    return 0 if report["status"] in ("prepared-only-no-device", "backing-and-consumer-validated") else 1


if __name__ == "__main__":
    raise SystemExit(main())
