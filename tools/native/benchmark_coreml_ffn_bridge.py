#!/usr/bin/env python3
"""Exercise TurboCider's low-level fixed-shape Core ML FFN C ABI.

This is intentionally separate from the LLaDA pipeline benchmark.  It proves
that a compiled, checkpoint-bound artifact can be opened through the public C
ABI and that contiguous FP16 buffers survive the Python/C++/Core ML boundary.
It does not claim end-to-end GPU/ANE acceleration by itself.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=Path("build/native/libturbocider.dylib"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=544)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-scale", type=float, default=0.25)
    parser.add_argument(
        "--oracle-package",
        type=Path,
        help="optional source .mlpackage for direct Core ML output parity",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


class CoreMLFFN:
    def __init__(self, library: Path, manifest: Path, checkpoint: Path, rows: int, warmups: int):
        self.library = ctypes.CDLL(str(library.resolve()))
        self.handle = ctypes.c_void_p()
        self._bind()
        error = ctypes.c_void_p()
        started = time.perf_counter()
        status = self.library.tc_coreml_ffn_create(
            str(manifest.resolve()).encode(),
            str(checkpoint.resolve()).encode(),
            rows,
            warmups,
            ctypes.byref(self.handle),
            ctypes.byref(error),
        )
        self.create_seconds = time.perf_counter() - started
        self._check(status, error)

    def _bind(self) -> None:
        pointer = ctypes.POINTER(ctypes.c_void_p)
        fp16_pointer = ctypes.POINTER(ctypes.c_uint16)
        self.library.tc_coreml_ffn_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            pointer,
            pointer,
        ]
        self.library.tc_coreml_ffn_create.restype = ctypes.c_int
        self.library.tc_coreml_ffn_predict.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            fp16_pointer,
            ctypes.c_int,
            fp16_pointer,
            pointer,
        ]
        self.library.tc_coreml_ffn_predict.restype = ctypes.c_int
        self.library.tc_coreml_ffn_metrics_json.argtypes = [ctypes.c_void_p, pointer, pointer]
        self.library.tc_coreml_ffn_metrics_json.restype = ctypes.c_int
        self.library.tc_coreml_ffn_free.argtypes = [ctypes.c_void_p]
        self.library.tc_string_free.argtypes = [ctypes.c_void_p]

    def _take_string(self, pointer: ctypes.c_void_p) -> str:
        if not pointer.value:
            return ""
        try:
            return ctypes.cast(pointer, ctypes.c_char_p).value.decode()
        finally:
            self.library.tc_string_free(pointer)

    def _check(self, status: int, error: ctypes.c_void_p) -> None:
        if status:
            detail = self._take_string(error) or f"native status {status}"
            raise RuntimeError(detail)
        if error.value:
            self.library.tc_string_free(error)

    def metrics(self) -> dict:
        result = ctypes.c_void_p()
        error = ctypes.c_void_p()
        status = self.library.tc_coreml_ffn_metrics_json(
            self.handle, ctypes.byref(result), ctypes.byref(error)
        )
        self._check(status, error)
        return json.loads(self._take_string(result))

    def predict(self, block: int, value: np.ndarray) -> tuple[np.ndarray, float]:
        if value.dtype != np.float16 or value.ndim != 3 or value.shape[0] != 1:
            raise ValueError("input must be contiguous FP16 [1, rows, hidden]")
        value = np.ascontiguousarray(value)
        output = np.empty_like(value)
        error = ctypes.c_void_p()
        started = time.perf_counter()
        status = self.library.tc_coreml_ffn_predict(
            self.handle,
            block,
            value.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            value.shape[1],
            output.view(np.uint16).ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.byref(error),
        )
        elapsed = time.perf_counter() - started
        self._check(status, error)
        return output, elapsed

    def close(self) -> None:
        if self.handle.value:
            self.library.tc_coreml_ffn_free(self.handle)
            self.handle = ctypes.c_void_p()


def main() -> int:
    args = arguments()
    if args.rows <= 0 or args.iterations <= 0:
        raise ValueError("rows and iterations must be positive")
    manifest = json.loads(args.manifest.read_text())
    hidden = int(manifest["shape"]["K"])
    artifacts = manifest.get("artifacts", {})
    selected = artifacts.get(str(args.block))
    if not isinstance(selected, dict) or not selected:
        raise ValueError(f"manifest does not contain block {args.block}")
    compiled = args.manifest.parent / next(iter(selected.values()))
    if compiled.suffix != ".mlmodelc" or not compiled.is_dir():
        raise ValueError("--manifest must be a compiled-cache manifest pointing to .mlmodelc")
    rng = np.random.default_rng(args.seed)
    value = (rng.standard_normal((1, args.rows, hidden), dtype=np.float32) * args.input_scale).astype(
        np.float16
    )
    bridge = CoreMLFFN(args.library, args.manifest, args.checkpoint, args.rows, args.warmups)
    try:
        samples = []
        output = None
        for _ in range(args.iterations):
            output, seconds = bridge.predict(args.block, value)
            samples.append(seconds)
        assert output is not None
        report = {
            "schema_version": 1,
            "manifest": str(args.manifest.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "block": args.block,
            "rows": args.rows,
            "hidden": hidden,
            "create_seconds": bridge.create_seconds,
            "prediction_seconds": samples,
            "prediction_median_seconds": float(np.median(samples)),
            "output": {
                "all_finite": bool(np.isfinite(output).all()),
                "max_abs": float(np.max(np.abs(output.astype(np.float32)))),
                "mean": float(np.mean(output.astype(np.float32))),
                "sha256": hashlib.sha256(output.tobytes()).hexdigest(),
            },
            "native_metrics": bridge.metrics(),
        }
        if args.oracle_package:
            import coremltools as ct

            oracle = ct.models.MLModel(
                str(args.oracle_package.resolve()), compute_units=ct.ComputeUnit.CPU_AND_NE
            )
            oracle_input = np.ascontiguousarray(value.transpose(0, 2, 1)[:, :, None, :])
            reference = oracle.predict({"x": oracle_input})["y"]
            reference = np.ascontiguousarray(reference[:, :, 0, :].transpose(0, 2, 1)).astype(
                np.float16
            )
            actual32 = output.astype(np.float32).reshape(-1)
            reference32 = reference.astype(np.float32).reshape(-1)
            difference = actual32 - reference32
            actual_centered = actual32 - actual32.mean()
            reference_centered = reference32 - reference32.mean()
            cosine_denominator = float(np.linalg.norm(actual32) * np.linalg.norm(reference32))
            correlation_denominator = float(
                np.linalg.norm(actual_centered) * np.linalg.norm(reference_centered)
            )
            report["oracle"] = {
                "package": str(args.oracle_package.resolve()),
                "bitwise_equal": bool(np.array_equal(output.view(np.uint16), reference.view(np.uint16))),
                "mae": float(np.mean(np.abs(difference))),
                "max_abs_error": float(np.max(np.abs(difference))),
                "cosine": float(np.dot(actual32, reference32) / cosine_denominator),
                "correlation": float(
                    np.dot(actual_centered, reference_centered) / correlation_denominator
                ),
            }
        encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_name(args.output.name + ".tmp")
            temporary.write_text(encoded)
            temporary.replace(args.output)
        print(encoded, end="")
    finally:
        bridge.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
