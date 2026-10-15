#!/usr/bin/env python3
"""Export a tiny true Public W8A8 graph and run the repo's native shared-IO fixtures.

Python/CoreML9 creates the artifact, native IOSurface/MLMultiArray binds INT8.
This is actual integration/parity/lifetime testing, not a timing benchmark.
"""
import argparse
from contextlib import redirect_stdout
import importlib.util
from pathlib import Path
import os
import platform
import subprocess
import sys
import tempfile
import time

LABS = Path(__file__).resolve().parents[1]
ROOT = LABS.parents[1]
sys.path.insert(0, str(LABS))
from common import emit, file_sha, report


def run(*, overlay=None, output=None):
    if overlay:
        source = Path(overlay).resolve()
        if not (source / "coremltools/__init__.py").is_file():
            raise ValueError("overlay does not contain coremltools")
        sys.path.insert(0, str(source))
    with redirect_stdout(sys.stderr):
        import coremltools as ct
    if platform.system() != "Darwin" or not hasattr(ct.target, "macOS26"):
        emit(report("public-w8-shared-io", {"status": "unsupported", "reason": "native public INT8 ABI needs macOS26/CoreML9",
                                           "coremltools": ct.__version__}, backend="public_coreml_unavailable"), output)
        return 3
    exporter_path = ROOT / "tools/coreml/export_runtime_w8a8_ffn.py"
    spec = importlib.util.spec_from_file_location("tc_w8_shared_lab_exporter", exporter_path)
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    native_sources = ["native/backends/ane_public_w8.mm", "native/backends/ane_gpu.mm",
                      "tests/native/ane_public_w8_test.mm", "tests/native/ane_public_w8_executor_test.mm",
                      "tools/native/build_ane_public_w8_test.sh"]
    source_pins = {name: file_sha(ROOT / name) for name in native_sources}
    child_env = {key: value for key, value in os.environ.items() if not key.startswith("TURBOCIDER_")}
    with tempfile.TemporaryDirectory(prefix="tc-technique-public-w8-") as temporary:
        folder = Path(temporary).resolve()
        graph, build, runtime_tmp = folder / "graph", folder / "build", folder / "runtime-tmp"
        runtime_tmp.mkdir()
        start = time.perf_counter()
        with redirect_stdout(sys.stderr):
            metadata = exporter.export(graph, exporter.geometry(33, 128, 512, 512, True))
        export_compile_seconds = time.perf_counter() - start
        built = subprocess.run(["bash", "tools/native/build_ane_public_w8_test.sh"], cwd=ROOT,
            env={**child_env, "TURBOCIDER_NATIVE_OUT": str(build)}, text=True, capture_output=True, timeout=120)
        if built.returncode:
            emit(report("public-w8-shared-io", {"status": "native_build_failed", "stderr": built.stderr}, backend="public_coreml_native"), output)
            return built.returncode
        fixtures = []
        for name in ("ane-public-w8-test", "ane-public-w8-executor-test"):
            binary = build / name
            began = time.perf_counter()
            result = subprocess.run([str(binary), str(graph / "manifest.json")], cwd=ROOT,
                                    env={**child_env, "TMPDIR": str(runtime_tmp)},
                                    text=True, capture_output=True, timeout=120)
            elapsed = time.perf_counter() - began
            symbols = subprocess.check_output(["strings", str(binary)], text=True)
            unexpected = [value for value in ("_ANEClient", "_ANERequest", "_ANEIOSurfaceObject", "_ANESharedEvents") if value in symbols]
            passed = result.returncode == 0 and "PASS" in result.stdout and not unexpected
            fixtures.append({"fixture": name, "passed": passed, "exit_code": result.returncode,
                             "stdout": result.stdout, "stderr": result.stderr, "binary_sha256": file_sha(binary),
                             "private_class_strings_found": unexpected,
                             "whole_fixture_process_seconds_not_operator_latency": elapsed})
            if not passed:
                emit(report("public-w8-shared-io", {"status": "integration_failed", "fixtures": fixtures}, backend="public_coreml_native"), output)
                return 1
        if source_pins != {name: file_sha(ROOT / name) for name in native_sources}:
            raise ValueError("native fixture inputs changed during the run")
    emit(report("public-w8-shared-io", {"status": "integration_passed", "coremltools": ct.__version__,
        "graph_shape": [33, 128, 512], "lora_inputs": True,
        "export_compile_seconds": export_compile_seconds, "exporter_sha256": file_sha(exporter_path),
        "graph_input_names": list(metadata["inputs"]), "fixtures": fixtures,
        "native_source_sha256_before_after": source_pins,
        "runtime_temp_isolated": True,
        "limits": "Original tiny native fixtures, synthetic corrections. No model LoRA, E2E speedup, physical ANE placement/INT8 MAC or full release qualification."},
        scope="actual public GPU staging/CoreML/GPU restore integration with native INT8 shared IO", backend="public_coreml_native"), output)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coreml-overlay", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and (args.output.exists() or args.output.is_symlink()):
        parser.error("report already exists")
    return run(overlay=args.coreml_overlay, output=args.output)


if __name__ == "__main__":
    raise SystemExit(main())
