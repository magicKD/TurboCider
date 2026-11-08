#!/usr/bin/env python3
"""Compare GPU W8A8 and a fixed Private ANE channel split on one real Turbo FFN."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sysconfig

from qwen21_convrot_projection_benchmark import CASES, ROOT, extract


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen-Image-2.1-Turbo-INT8")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, choices=(256, 1024), default=1024)
    parser.add_argument("--channels", type=int, choices=(3072, 4096, 6144), default=4096)
    parser.add_argument("--repeats", type=int, choices=range(3, 16), default=11)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output, native = args.output.resolve(), args.native.resolve()
    identity = json.loads((native / "runtime-build/runtime-build-manifest.json").read_text())
    for entry in identity["inputs"]["sources"]:
        if digest(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("source/library ABI mismatch: " + entry["path"])
    mlx = Path(os.environ.get("MLX_ROOT", Path(sysconfig.get_paths()["purelib"]) / "mlx")).resolve()
    output.mkdir(parents=True)
    code = [
        "native/backends/ane_gpu.mm", "native/backends/private/ane_program.mm",
        "native/backends/private/ane_mil.cpp", "native/backends/private/ane_w8_executor.mm",
        "native/backends/ane_memory.cpp", "native/core/gguf_decode.cpp",
        "tools/benchmarks/qwen21_convrot_ane_ffn_benchmark.mm",
    ]
    binary = output / "ffn-probe"
    compile_args = [
        "xcrun", "clang++", "-std=c++20", "-O2", "-fobjc-arc", "-mmacosx-version-min=26.2",
        "-Wall", "-Wextra", "-Werror", "-Wno-unused-parameter", "-Wno-missing-field-initializers",
        "-Wno-deprecated-declarations", '-DTC_TEST_BOUND_NATIVE_ID="' + identity["runtime_build_id"] + '"',
        "-I", "native/core", "-I", "bindings/c/include", "-isystem", str(mlx / "include"),
        *code, "-L", str(native), "-lturbocider", "-L", str(mlx / "lib"), "-lmlx", "-ljaccl",
        "-framework", "Foundation", "-framework", "Metal", "-framework", "IOSurface",
        "-Wl,-rpath," + str(native), "-Wl,-rpath," + str(mlx / "lib"), "-o", str(binary),
    ]
    with (output / "build.log").open("w") as log:
        subprocess.run(compile_args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=90)
    sources = {}
    for case in ("dit_gate_up", "dit_down"):
        leaf, prefixes = CASES[case]
        sources[case] = extract(args.model / leaf, prefixes, output / (case + ".safetensors"))
    overrides = {
        "TURBOCIDER_PRIVATE_ANE_FIXED_PROGRAM": "1",
        "TURBOCIDER_PRIVATE_ANE_SCALE_CACHE": "0",
        "TURBOCIDER_PRIVATE_ANE_LAUNCH_FENCE": "1",
    }
    command = ["/usr/bin/time", "-l", str(binary), str(output / "dit_gate_up.safetensors"),
               str(output / "dit_down.safetensors"), str(output / "program-cache"), str(args.rows),
               str(args.channels), str(args.repeats)]
    report = {
        "scope": "One real Turbo DiT FFN, two deterministic synthetic BF16 inputs, component timing only",
        "compile": compile_args, "command": command, "environment": overrides,
        "source_tensors": sources, "source_sha256": {path: digest(ROOT / path) for path in code},
        "runtime_id": identity["runtime_build_id"], "library_sha256": digest(native / "libturbocider.dylib"),
        "probe_sha256": digest(binary), "status": "prepared",
    }
    report_path = output / "report.json"
    if not args.prepare_only:
        environment = {key: value for key, value in os.environ.items() if not key.startswith("TURBOCIDER_")}
        environment.update(overrides)
        report["swap_before"] = subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip()
        with (output / "measure.json").open("w") as stdout, (output / "time.log").open("w") as stderr:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, timeout=90)
        report["swap_after"] = subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip()
        peak = re.search(r"([0-9]+)\s+peak memory footprint", (output / "time.log").read_text())
        report["peak_footprint_bytes"] = int(peak.group(1)) if peak else None
        report["exit_code"] = result.returncode
        report["status"] = "failed"
        if result.returncode == 0:
            measurements = json.loads((output / "measure.json").read_text())
            medians = {name: statistics.median(measurements[name]) for name in
                       ("gpu_seconds", "hybrid_ready_seconds", "hybrid_with_stage_seconds", "stage_seconds")}
            report.update(status="complete", observations=measurements, medians=medians)
            report["gpu_over_hybrid_with_stage"] = medians["gpu_seconds"] / medians["hybrid_with_stage_seconds"]
            if peak is None or int(peak.group(1)) > 1 << 30:
                report["status"] = "memory-bound-exceeded"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(report_path)
    if report["status"] not in ("prepared", "complete"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
