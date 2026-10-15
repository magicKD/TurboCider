#!/usr/bin/env python3
"""Owned-process tiny Metal capture capability probe; not a model benchmark.

Sets MTL_CAPTURE_ENABLED only in this explicit diagnostic child before MLX
import. Inputs are synthetic, raw capture stays local, and no viewer or Xcode
automation is launched. Missing capture is not silently labelled successful.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import time


def sha256_file(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def capture_artifacts(capture, output):
    if not capture.exists():
        raise ValueError("capture API returned but no artifact was published")
    files = sorted(path for path in capture.rglob("*") if path.is_file()) if capture.is_dir() else [capture]
    records = [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                "sha256": sha256_file(path)} for path in files]
    if not any(record["bytes"] for record in records):
        raise ValueError("capture API returned but artifacts were empty")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error(str(error))
    source = Path(__file__).resolve()
    report = {
        "schema": "tc-mlx-capture-capability-v1",
        "status": "starting",
        "synthetic": True,
        "production_qualified": False,
        "raw_trace_local_only": True,
        "formal_performance_score": False,
        "runner_sha256": sha256_file(source),
        "samples": [],
        "errors": [],
        "recorded_at_singapore": time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() + 8 * 3600)
        ) + "+08:00",
    }
    shutil.copy2(source, output / source.name)

    def save():
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    def record_error(error):
        report["status"] = "capture-unavailable-or-failed"
        report["errors"].append({"type": type(error).__name__, "message": str(error)})

    save()
    started_capture = False
    mx = None
    try:
        if platform.system() != "Darwin" or shutil.disk_usage(output).free < 512 * 1024 * 1024:
            raise ValueError("requires macOS and bounded disk headroom")
        os.environ["MTL_CAPTURE_ENABLED"] = "1"
        import numpy as np
        import mlx.core as mx

        mx.set_default_device(mx.gpu)
        info = mx.device_info()
        report["environment"] = {
            "macos": platform.mac_ver()[0], "mlx": version("mlx"),
            "device": {key: info.get(key) for key in ("device_name", "architecture", "memory_size")},
        }
        rng = np.random.default_rng(29)
        a = rng.uniform(-.1, .1, (128, 256)).astype(np.float32)
        b = rng.uniform(-.1, .1, (256, 384)).astype(np.float32)
        x, w = mx.array(a), mx.array(b)
        mx.eval(x, w)
        function = mx.compile(lambda left, right: mx.tanh(left @ right))
        for _ in range(2):
            mx.eval(function(x, w))
        for _ in range(4):
            now = time.perf_counter()
            result = function(x, w)
            mx.eval(result)
            report["samples"].append({
                "scope": "unprofiled-host-until-output-ready", "source_kind": "measured",
                "wall_seconds": time.perf_counter() - now,
            })
        capture = output / "owned-tiny.gputrace"
        mx.metal.start_capture(str(capture))
        started_capture = True
        result = function(x, w)
        mx.eval(result)
        mx.metal.stop_capture()
        started_capture = False
        actual = np.array(result)
        reference = np.tanh(np.einsum(
            "mk,kn->mn", a.astype(np.float64), b.astype(np.float64), optimize=False
        ))
        if actual.shape != reference.shape or not np.isfinite(actual).all():
            raise ValueError("invalid captured output")
        error = float(np.max(np.abs(actual.astype(np.float64) - reference)))
        if error > 1e-5:
            raise ValueError("independent FP64 capture oracle failed")
        report.update(
            status="captured", capture_path=capture.name, max_abs_vs_fp64=error,
            capture_files=capture_artifacts(capture, output),
        )
    except Exception as error:
        record_error(error)
    finally:
        if started_capture and mx is not None:
            try:
                mx.metal.stop_capture()
            except Exception as error:
                record_error(error)
        if mx is not None:
            try:
                mx.synchronize()
            except Exception as error:
                record_error(error)
        save()
    print(json.dumps(report, indent=2))
    return int(report["status"] != "captured")


if __name__ == "__main__":
    raise SystemExit(main())
