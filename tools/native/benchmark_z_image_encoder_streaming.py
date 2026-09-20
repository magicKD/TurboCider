#!/usr/bin/env python3
"""Exploratory full-image encoder split comparison on legacy Z-Image streaming.

Fresh worker per request, fixed GPU/ANE/ANE/GPU order. No qualification claims,
pressure injection, route override, or change to public streaming admission.
"""
from __future__ import annotations

import argparse
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROMPT = ("A red fox sitting in a snowy forest, soft winter light, detailed fur. "
          "Tall pine trees frame the scene, with gentle shadows across fresh snow and pale mist in the distance. "
          "The fox looks toward the camera with bright amber eyes, a fluffy tail, and a calm expression.")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def api(library: str):
    lib = c.CDLL(library)
    lib.tc_string_free.argtypes = [c.c_void_p]
    lib.tc_engine_free.argtypes = [c.c_void_p]
    lib.tc_engine_create_model_worker.argtypes = [c.c_char_p, c.c_char_p, c.POINTER(c.c_void_p), c.POINTER(c.c_void_p)]
    lib.tc_z_image_tokenize_json.argtypes = [c.c_char_p, c.c_char_p, c.POINTER(c.c_void_p), c.POINTER(c.c_void_p)]
    lib.tc_plan_json.argtypes = [c.c_char_p, c.POINTER(c.c_void_p), c.POINTER(c.c_void_p)]
    return lib


def consume(lib, pointer):
    if not pointer.value:
        return None
    try:
        return c.string_at(pointer).decode()
    finally:
        lib.tc_string_free(pointer)


def worker(path: Path) -> None:
    case = json.loads(path.read_text())
    lib = api(case["library"])
    engine, error = c.c_void_p(), c.c_void_p()
    started = time.perf_counter()
    status = lib.tc_engine_create_model_worker(b"z-image-turbo", case["model_path"].encode(), c.byref(engine), c.byref(error))
    failure = consume(lib, error)
    if status:
        write(path.parent / "result.json", {"status": status, "error": failure})
        raise RuntimeError(failure)
    creation = time.perf_counter() - started
    try:
        callback_type = c.CFUNCTYPE(None, c.c_char_p, c.c_void_p)
        lib.tc_engine_generate.argtypes = [c.c_void_p, c.c_char_p, callback_type, c.c_void_p,
                                         c.POINTER(c.c_void_p), c.POINTER(c.c_void_p)]
        with (path.parent / "events.jsonl").open("w") as events:
            @callback_type
            def callback(raw, _):
                events.write(raw.decode() + "\n")
                events.flush()
            result, error = c.c_void_p(), c.c_void_p()
            started = time.perf_counter()
            status = lib.tc_engine_generate(engine, json.dumps(case["request"]).encode(), callback, None,
                                            c.byref(result), c.byref(error))
            wall = time.perf_counter() - started
            value, failure = consume(lib, result), consume(lib, error)
        report = {"status": status, "error": failure, "engine_creation_seconds": creation,
                  "request_wall_seconds": wall, "result": json.loads(value) if value else None}
        if status == 0:
            report["image_sha256"] = digest(Path(case["request"]["outputs"][0]["path"]))
        write(path.parent / "result.json", report)
        if status:
            raise RuntimeError(failure)
    finally:
        lib.tc_engine_free(engine)


def run(args) -> None:
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    library, model, manifest = args.library.resolve(), args.model.resolve(), args.manifest.resolve()
    lib = api(str(library))
    result, error = c.c_void_p(), c.c_void_p()
    status = lib.tc_z_image_tokenize_json(str(model).encode(), PROMPT.encode(), c.byref(result), c.byref(error))
    tokens, failure = consume(lib, result), consume(lib, error)
    if status or json.loads(tokens)["valid"] != 64:
        raise RuntimeError(f"expected 64 native tokens: {tokens}, {failure}")
    request = {"schema_version": 2, "model": "z-image-turbo", "operation": "image.generate",
               "inputs": [{"kind": "text", "role": "prompt", "text": PROMPT}],
               "outputs": [{"kind": "image", "path": "unused.png", "width": 256, "height": 256}],
               "sampling": {"seed": 42, "steps": 9},
               "execution": {"policy": "gpu", "residency": "streamed", "memory_budget_bytes": 8 << 30,
                             "warmup_iterations": 0}}
    plan = {"schema_version": 1, "scope": "Exploratory legacy budget streaming; not framework/public hybrid qualification",
            "library": str(library), "library_sha256": digest(library), "driver_sha256": digest(Path(__file__)),
            "model_path": str(model), "encoder_manifest": str(manifest), "manifest_sha256": digest(manifest),
            "tokenizer_result": json.loads(tokens), "request": request, "order": ["gpu", "hybrid", "hybrid", "gpu"],
            "lifecycle": "Fresh worker/engine per request; OS and Core ML service caches not reset",
            "quality_thresholds": {"rgb_correlation_min": .99, "rgb_cosine_min": .995, "rgb_mae_max": 2.55,
                                   "final_latent_relative_l2_max": .05, "final_latent_cosine_min": .995},
            "diagnostics": "Both routes dump the same tensors; this is not an uninstrumented timing campaign",
            "timeout_seconds": 300, "interpretation": "Two pairs only; no statistical acceleration or memory qualification; CPU+NE configuration does not prove ANE residency"}
    write(root / "plan.json", plan)
    cases = []
    for index, route in enumerate(plan["order"]):
        directory = root / f"{index:02d}-{route}"
        directory.mkdir()
        actual = json.loads(json.dumps(request))
        actual["outputs"][0]["path"] = str(directory / "image.png")
        # Dumping tensors changes timing. Both routes use identical diagnostics.
        actual["dump_tensors"] = str(directory / "tensors")
        if route == "hybrid":
            actual["execution"].update(encoder_ane_manifest=str(manifest), allow_approximation=True)
        write(directory / "case.json", {"library": str(library), "model_path": str(model), "request": actual})
        with (directory / "worker.log").open("wb") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker-case", str(directory / "case.json")],
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            started = time.perf_counter()
            try:
                code = process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                write(directory / "timeout.json", {"timeout_seconds": 300, "returncode": process.returncode})
                raise RuntimeError("owned worker timed out; evidence preserved")
        row = {"index": index, "route": route, "returncode": code, "process_wall_seconds": time.perf_counter() - started}
        if (directory / "result.json").exists():
            row.update(json.loads((directory / "result.json").read_text()))
        cases.append(row)
        write(root / "results.json", {"scope": plan["scope"], "cases": cases})
        if code:
            raise RuntimeError(f"case {index} failed; evidence preserved")
        value = row["result"]
        if value.get("prompt_cache_hit"):
            raise RuntimeError("encoder comparison unexpectedly hit prompt cache")
        if route == "hybrid" and not value.get("encoder_hybrid"):
            raise RuntimeError("hybrid request fell back to GPU")
        if route == "hybrid":
            metrics = value["encoder_hybrid"]
            if metrics["runtime_failed"] or metrics["runtime_failures_session_total"] or metrics["runtime_calls_session_total"] != 35:
                raise RuntimeError("encoder did not complete all 35 Core ML branches successfully")
        if value.get("block_streaming", {}).get("streamed_blocks", 0) <= 0:
            raise RuntimeError("denoiser did not report actual streamed blocks")
        print(index, route, row["request_wall_seconds"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-case", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker_case:
        worker(args.worker_case)
    elif not all((args.library, args.model, args.manifest, args.output)):
        parser.error("--library, --model, --manifest, --output are required")
    else:
        run(args)


if __name__ == "__main__":
    main()
