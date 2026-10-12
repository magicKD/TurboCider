#!/usr/bin/env python3
"""Compile/run one synthetic HybridQkv retirement regression, without a model.

Run serially outside a sandbox that hides Metal:
  .venv/bin/python tools/validation/runtime_ane_qkv_regression.py \
    --native-dir build/verify-20261001-native \
    --output outputs/runtime-ane-qkv-focused-20261001

Only a new output directory is accepted. Logs, hashes and the test executable
are retained; all test source/compiled Core ML graphs and private temporary
files belong to a TemporaryDirectory removed even on failure. CPU_AND_NE is
not evidence of physical ANE placement or a production speed benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import time

from runtime_ane_common import benchmark_environment
from runtime_ane_memory import run_owned

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory")
    args = parser.parse_args()
    native = args.native_dir.resolve(strict=True)
    library = native / "libturbocider.dylib"
    if not library.is_file():
        parser.error("--native-dir must contain an existing libturbocider.dylib")
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve(strict=True)
    source = ROOT / "tests/native/ane_qkv_test.cpp"
    binary = output / "ane-qkv-test"
    mlx = Path(os.environ.get("MLX_ROOT") or str(Path(sysconfig.get_paths()["purelib"]) / "mlx"))
    if not (mlx / "lib/libmlx.dylib").is_file():
        parser.error("use the installed .venv Python or supply MLX_ROOT")
    compiler = Path("/Library/Developer/CommandLineTools/usr/bin/clang++")
    sdk = os.environ.get("SDKROOT", "/Library/Developer/CommandLineTools/SDKs/MacOSX15.2.sdk")
    manifest = native / "runtime-build/runtime-build-manifest.json"
    identity = json.loads(manifest.read_text()) if manifest.is_file() else {}
    deployment = identity.get("inputs", {}).get("policy", {}).get("deployment_target", "26.2")
    environment = benchmark_environment()
    report = {"scope": "three synthetic QKV wrapper cases once; no checkpoint/model",
              "graph_shape": [2, 4096, 12288], "input_rows": 5, "chunks": 2,
              "raw_weight_slot_bytes": 96 << 20, "tile_k": 1024, "tile_n": 512,
              "library_sha256": digest(library), "library": str(library),
              "test_source_sha256": digest(source), "test_source": str(source),
              "runtime_build_id": identity.get("runtime_build_id"),
              "hardware_ane_residency_proven": False, "steps": []}

    def command(argv, name, timeout):
        row = {"argv": [str(arg) for arg in argv],
               "combined_stdout_stderr": str(output / (name + ".log"))}
        report["steps"].append(row)
        started = time.monotonic()
        with Path(row["combined_stdout_stderr"]).open("w") as stream:
            try:
                result = run_owned(row["argv"], cwd=ROOT, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
                row["returncode"] = result.returncode
            except BaseException as error:
                row["error"] = f"{type(error).__name__}: {error}"
                if getattr(error, "__notes__", None):
                    row["cleanup_notes"] = error.__notes__
                raise
            finally:
                row["seconds"] = time.monotonic() - started
        if result.returncode:
            raise RuntimeError(f"{name} exit {result.returncode}; see {row['combined_stdout_stderr']}")
        print(f"{name}: exit 0 ({row['seconds']:.3f}s)", flush=True)

    started = time.monotonic()
    fixture = None
    try:
        command([compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                 "-isysroot", sdk, f"-mmacosx-version-min={deployment}",
                 "-I", ROOT / "native/core", "-isystem", mlx / "include", source,
                 "-L" + str(native), "-lturbocider", "-L" + str(mlx / "lib"), "-lmlx", "-ljaccl",
                 "-Wl,-rpath," + str(native), "-Wl,-rpath," + str(mlx / "lib"),
                 "-o", binary], "compile", 120)
        report["binary_sha256"] = digest(binary)
        fixture = tempfile.TemporaryDirectory(prefix="tc-qkv-focused-")
        fixture_root = Path(fixture.name).resolve()
        report["temporary_fixture"] = str(fixture_root)
        graph = fixture_root / "graph"
        temporary = fixture_root / "private-tmp"
        temporary.mkdir(mode=0o700)
        environment["TMPDIR"] = str(temporary) + "/"
        command([sys.executable, ROOT / "tools/coreml/export_runtime_ane.py",
                 "--kind", "matmul", "--rows", "2", "--hidden", "4096", "--width", "12288",
                 "--tile-k", "1024", "--tile-n", "512", "--output", graph], "export", 180)
        try:
            command([binary, graph / "manifest.json", temporary], "qkv-test", 180)
        finally:
            # Observe residue BEFORE deleting this test's owned parent. The C++
            # assertions also check failure cleanup while each wrapper is alive.
            report["leases_before_fixture_cleanup"] = sorted(
                str(path) for path in temporary.glob("turbocider-runtime-ane-*"))
        if report["leases_before_fixture_cleanup"]:
            raise RuntimeError("QKV private lease survived wrapper destruction")
        required = ("PASS QKV staging failure:", "PASS QKV partial prediction failure:",
                    "PASS QKV cancellation:", "PASS runtime QKV wrapper regression")
        if not all(marker in (output / "qkv-test.log").read_text() for marker in required):
            raise RuntimeError("missing QKV assertion receipts")
        report["success"] = True
    except BaseException as error:
        report["success"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        print(report["error"], file=sys.stderr)
    finally:
        if fixture is not None:
            try:
                fixture.cleanup()
                report["temporary_fixture_removed"] = not Path(fixture.name).exists()
            except Exception as error:
                report["success"] = False
                report["cleanup_error"] = str(error)
                report["temporary_fixture_removed"] = False
        try:
            report["library_sha256_after"] = digest(library)
            report["test_source_sha256_after"] = digest(source)
            report["input_hashes_unchanged"] = (
                report["library_sha256"] == report["library_sha256_after"] and
                report["test_source_sha256"] == report["test_source_sha256_after"])
            if not report["input_hashes_unchanged"]:
                report["success"] = False
                report["input_hash_error"] = "library or test source changed during regression"
        except Exception as error:
            report["success"] = False
            report["input_hash_error"] = str(error)
        report["total_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("Evidence: " + str(output / "report.json"), flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
