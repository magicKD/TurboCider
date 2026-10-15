#!/usr/bin/env python3
"""Local Metal -> INT8 Core ML -> Metal shared-IO sample, not a full FFN.

Only files next to this script and public system frameworks are compiled.
See ane.md for the distinction from the historical product integration test.
"""
import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

LABS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LABS))
from common import emit, file_sha, report


def run(*, overlay=None, output=None):
    if output and (Path(output).exists() or Path(output).is_symlink()):
        raise ValueError("report already exists")
    if overlay:
        source = Path(overlay).resolve()
        if not (source / "coremltools/__init__.py").is_file():
            raise ValueError("overlay does not contain coremltools")
        sys.path.insert(0, str(source))
    with redirect_stdout(sys.stderr):
        import coremltools as ct
        import numpy as np
    if platform.system() != "Darwin" or not hasattr(ct.target, "macOS26"):
        emit(report("public-w8-shared-io", {"status": "unsupported", "reason": "public INT8 feature ABI needs macOS26/CoreML9",
                                           "coremltools": ct.__version__}, backend="public_coreml_unavailable"), output)
        return 3
    from graphs import shared_int8
    root = Path(__file__).resolve().parent
    source_pins = {name: file_sha(root / name) for name in ("graphs.py", "native_shared_io.mm")}
    child_env = {key: value for key, value in os.environ.items() if not key.startswith("TURBOCIDER_")}
    with tempfile.TemporaryDirectory(prefix="tc-technique-local-int8-") as temporary:
        folder = Path(temporary)
        package, compiled = folder / "graph.mlpackage", folder / "graph.mlmodelc"
        start = time.perf_counter()
        with redirect_stdout(sys.stderr):
            model = ct.convert(shared_int8(), convert_to="mlprogram", minimum_deployment_target=ct.target.macOS26,
                inputs=[ct.TensorType(name="x", shape=(128, 33), dtype=np.int8),
                        ct.TensorType(name="w", shape=(64, 128), dtype=np.int8)],
                compute_precision=ct.precision.FLOAT16, skip_model_load=True)
            model.save(str(package))
            ct.models.utils.compile_model(str(package), destination_path=str(compiled))
        export_compile_seconds = time.perf_counter() - start
        binary = folder / "shared-int8-lab"
        command = ["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-Wall", "-Wextra",
                   str(root / "native_shared_io.mm"), "-framework", "Foundation", "-framework", "Metal",
                   "-framework", "CoreML", "-framework", "CoreVideo", "-framework", "IOSurface", "-o", str(binary)]
        built = subprocess.run(command, env=child_env, capture_output=True, text=True, timeout=120)
        if built.returncode:
            emit(report("public-w8-shared-io", {"status": "native_build_failed", "stderr": built.stderr}), output)
            return 1
        result = subprocess.run([str(binary), str(compiled)], env={**child_env, "TMPDIR": str(folder)},
                                capture_output=True, text=True, timeout=120)
        symbols = subprocess.check_output(["strings", str(binary)], text=True)
        private = [name for name in ("_ANEClient", "_ANERequest", "_ANEIOSurfaceObject", "_ANESharedEvents") if name in symbols]
        try:
            native = json.loads(result.stdout)
        except ValueError:
            native = {"status": "invalid_native_report"}
        if source_pins != {name: file_sha(root / name) for name in source_pins}:
            raise ValueError("local native fixture inputs changed during the run")
        passed = result.returncode == 0 and native.get("passed") is True and not private
        details = {"status": "integration_passed" if passed else "integration_failed", "native": native,
                   "graph_kind": "single normalized INT8-input matmul, not SwiGLU or LoRA",
                   "coremltools": ct.__version__, "export_compile_seconds": export_compile_seconds,
                   "native_source_sha256_before_after": source_pins, "exporter_sha256": source_pins["graphs.py"],
                   "binary_sha256": file_sha(binary), "private_class_strings_found": private,
                   "runtime_temp_isolated": True, "build_warnings": built.stderr, "stderr": result.stderr,
                   "limits": "Serial ownership/ABI/parity sample, not physical ANE residency, INT8 MAC, overlap or end-to-end speed evidence."}
        del model
    emit(report("public-w8-shared-io", details, backend="public_coreml_native",
                scope="actual local Metal producer/public Core ML/Metal consumer shared-feature ABI"), output)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coreml-overlay", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    return run(overlay=args.coreml_overlay, output=args.output)


if __name__ == "__main__":
    raise SystemExit(main())
