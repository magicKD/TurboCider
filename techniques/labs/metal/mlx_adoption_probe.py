#!/usr/bin/env python3
"""Explicit bounded native MLX external-memory adoption probe, never a model run.

Default only snapshots source. --device uses an already installed caller-selected
MLX tree; no install/download, library rebuild, private API or system changes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sysconfig
import time


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1 << 20), b""):
            value.update(data)
    return value.hexdigest()


def assets(root):
    result = {"headers": [], "libraries": {}}
    for path in sorted((root / "include").rglob("*")):
        if path.is_file():
            result["headers"].append({"path": str(path.relative_to(root)), "sha256": sha(path)})
    if not result["headers"]:
        raise ValueError("installed MLX headers required")
    for name in ("libmlx.dylib", "libjaccl.dylib", "mlx.metallib"):
        path = root / "lib" / name
        if not path.is_file():
            raise ValueError("explicit complete installed MLX assets required")
        result["libraries"][name] = sha(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", action="store_true")
    parser.add_argument("--mlx-root", type=Path, default=Path(sysconfig.get_paths()["purelib"]) / "mlx")
    parser.add_argument("--case", choices=("padded", "port"), default="padded")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.seed <= 10000:
        parser.error("bounded seed required")
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    pins = {name: sha(root / name) for name in ("mlx_adoption_probe.py", "mlx_adoption_probe.mm")}
    report = {"schema": "tc-mlx-adoption-probe-v1", "status": "starting", "errors": [], "device_requested": args.device,
              "case": args.case, "seed": args.seed, "source_sha256": pins, "native": None,
              "synthetic": True, "real_qwen": False, "formal_performance_score": False, "production_qualified": False,
              "physical_overlap_qualified": False, "runtime_internal_copy_behavior": "unknown", "python": platform.python_version()}

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    try:
        if shutil.disk_usage(output).free < 128 * 1024**2:
            raise ValueError("128 MiB disk headroom required; nothing deleted")
        (output / "source").mkdir()
        for name in pins:
            shutil.copy2(root / name, output / "source" / name)
        if not args.device:
            report["status"] = "prepared-only-no-device"
        else:
            if platform.system() != "Darwin":
                raise ValueError("native Metal/MLX probe requires macOS")
            mlx = args.mlx_root.resolve(); before = assets(mlx)
            report["mlx_assets"] = before
            report["macos_product_version"] = subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip()
            report["macos_build"] = subprocess.check_output(["sw_vers", "-buildVersion"], text=True).strip()
            report["sdk_version"] = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-version"], text=True).strip()
            report["compiler_version"] = subprocess.check_output(["xcrun", "clang++", "--version"], text=True).splitlines()[0]
            binary = output / "mlx-adoption"
            command = ["xcrun", "clang++", "-std=c++20", "-O2", "-fobjc-arc", "-Wall", "-Wextra", "-mmacosx-version-min=26.2",
                       "-isystem", str(mlx / "include"), str(output / "source/mlx_adoption_probe.mm"),
                       "-L" + str(mlx / "lib"), "-lmlx", "-ljaccl", "-Wl,-rpath," + str(mlx / "lib"),
                       "-framework", "Foundation", "-framework", "Metal", "-framework", "IOSurface", "-o", str(binary)]
            start = time.perf_counter_ns()
            with (output / "build.stdout.log").open("x") as stdout, (output / "build.stderr.log").open("x") as stderr:
                built = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=120)
            report.update(build_wall_ns=time.perf_counter_ns() - start, build_returncode=built.returncode)
            save()
            if built.returncode:
                raise ValueError("native client build failed; inspect owned log")
            report["binary_sha256"] = sha(binary)
            environment = {key: value for key, value in os.environ.items()
                           if not key.startswith(("DYLD_", "ANE_", "ANEFORGE_", "MPSGRAPH_", "TURBOCIDER_"))}
            with (output / "native.stdout.log").open("x") as stdout, (output / "native.stderr.log").open("x") as stderr:
                child = subprocess.run([str(binary), args.case, str(args.seed), str(output / "native.json"), "0"],
                                       env=environment, stdout=stdout, stderr=stderr, timeout=90)
            report["native_returncode"] = child.returncode
            if (output / "native.json").is_file():
                report["native"] = json.loads((output / "native.json").read_text())
                report["raw_native_sha256"] = sha(output / "native.json")
            report["mlx_assets_unchanged"] = before == assets(mlx)
            report["status"] = "actual-adoption-and-lifetime-validated" if child.returncode == 0 else "native-failed"
            if not report["mlx_assets_unchanged"]:
                raise ValueError("installed MLX assets changed")
        report["source_inputs_unchanged"] = pins == {name: sha(root / name) for name in pins}
        if not report["source_inputs_unchanged"]:
            raise ValueError("probe source changed")
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"; report["errors"].append({"type": type(error).__name__, "message": str(error)})
    save(); print(json.dumps({"status": report["status"], "errors": report["errors"]}))
    return 0 if report["status"] in ("prepared-only-no-device", "actual-adoption-and-lifetime-validated") else 1


if __name__ == "__main__":
    raise SystemExit(main())
