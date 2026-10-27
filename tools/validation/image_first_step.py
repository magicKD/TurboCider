"""Measure cold/warm image startup via the public C ABI (development only).

Separates time-to-sampling, completed step 0->1, later steps, and full export.
Repeated and changed prompts distinguish reusable state from per-request work.
No phase is called compilation time without independent compiler evidence.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def summarize(events, wall):
    boundaries = {}
    phases = {}
    for event in events:
        phases.setdefault(event["phase"], [event["elapsed_seconds"], event["elapsed_seconds"]])[1] = event["elapsed_seconds"]
        if event["phase"] == "denoise":
            boundaries.setdefault(event["completed"], event["elapsed_seconds"])
    samples = [{"step": step, "seconds": boundaries[step] - boundaries[step - 1]}
               for step in sorted(boundaries) if step > 0 and step - 1 in boundaries]
    later = [item["seconds"] for item in samples if item["step"] > 1]
    first = samples[0]["seconds"] if samples and samples[0]["step"] == 1 else None
    median = statistics.median(later) if later else None
    return dict(request_wall_seconds=wall, time_to_denoise_seconds=boundaries.get(0),
                first_step_seconds=first, later_step_median_seconds=median,
                first_to_later_ratio=first / median if first is not None and median else None,
                step_times=samples, phase_spans=phases)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", type=Path, default=Path("build/native/libturbocider.dylib"))
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--model-id", choices=["qwen-image-2.1", "z-image-turbo"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--steps", type=int)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--residency", choices=["resident", "component_staged"], default="resident")
    p.add_argument("--runs", type=int, default=4)
    p.add_argument("--prepare", choices=["none", "load", "warmup"], default="none")
    p.add_argument("--lora", type=Path)
    p.add_argument("--sampler", choices=["auto", "dmd2"], default="auto")
    p.add_argument("--reference-size", type=int, choices=[512, 1024], default=1024)
    p.add_argument("--metal-trace", type=Path, help="fresh trace from the injected development-only Metal observer")
    p.add_argument("--sample-run", type=int, help="macOS development only: sample this run's first denoise second")
    a = p.parse_args()
    if a.runs < 1 or (a.steps is not None and a.steps < 1):
        p.error("--runs and --steps must be positive")
    if a.size < 64 or a.size % 32:
        p.error("--size must be at least 64 and divisible by 32")
    if a.reference and a.model_id != "qwen-image-2.1":
        p.error("Z-Image Turbo currently exposes text-to-image only; do not invent an edit timing")
    if a.sampler != "auto" and a.model_id != "qwen-image-2.1":
        p.error("--sampler is supported only for Qwen Image 2.1")
    if a.sample_run is not None and (sys.platform != "darwin" or not 0 <= a.sample_run < a.runs):
        p.error("--sample-run requires macOS and a valid zero-based run index")
    a.output.mkdir(parents=True, exist_ok=True)
    lib = C.CDLL(str(a.library.resolve()))
    pointer = C.POINTER(C.c_void_p)
    callback_type = C.CFUNCTYPE(None, C.c_char_p, C.c_void_p)
    lib.tc_string_free.argtypes = [C.c_void_p]
    lib.tc_runtime_build_identity.restype = C.c_void_p
    lib.tc_system_json.restype = C.c_void_p
    lib.tc_engine_create_model.argtypes = [C.c_char_p, C.c_char_p, pointer, pointer]
    lib.tc_engine_generate.argtypes = [C.c_void_p, C.c_char_p, callback_type, C.c_void_p, pointer, pointer]
    lib.tc_engine_prepare.argtypes = [C.c_void_p, C.c_char_p, C.c_int, callback_type, C.c_void_p, pointer, pointer]
    lib.tc_engine_free.argtypes = [C.c_void_p]

    def consume(value):
        address = value.value if isinstance(value, C.c_void_p) else value
        if not address:
            return None
        text = C.string_at(address).decode()
        lib.tc_string_free(address)
        return text

    report = dict(schema_version=1, preparation_mode=a.prepare,
                  instrumentation=dict(metal_trace=a.metal_trace is not None, host_sample_run=a.sample_run),
                  runtime=consume(lib.tc_runtime_build_identity()),
                  system=json.loads(consume(lib.tc_system_json())), runs=[], preparation=None)
    request = dict(model=a.model_id, operation="image.edit" if a.reference else "image.generate",
                   prompt="Change the teapot in <image1> to blue, preserve its shape and background." if a.reference else
                          "A red ceramic teapot on a wooden table, warm sunlight, detailed photography.",
                   width=a.size, height=a.size, frames=1, audio=False, seed=42,
                   steps=a.steps or (20 if a.model_id == "qwen-image-2.1" else 8),
                   execution="gpu", residency=a.residency, output=str((a.output / "prepare.png").resolve()))
    if a.reference:
        request["inputs"] = [dict(kind="image", role="reference", path=str(a.reference.resolve()))]
        request["qwen21_reference_size"] = a.reference_size
        if a.reference_size != 1024:
            request["allow_approximation"] = True
    if a.lora:
        request.update(loras=[dict(path=str(a.lora.resolve()), strength=1.0, role="transformer")],
                       lora_strategy="inference_time", allow_approximation=True)
    if a.sampler != "auto":
        request["qwen21_sampler"] = a.sampler
    base_prompt = request["prompt"]
    engine, error = C.c_void_p(), C.c_void_p()
    start = time.perf_counter()
    code = lib.tc_engine_create_model(a.model_id.encode(), str(a.model.resolve()).encode(), C.byref(engine), C.byref(error))
    message = consume(error)
    if code:
        raise RuntimeError(message)
    report["constructor_seconds"] = time.perf_counter() - start

    def invoke(value, prepare=None, run_index=None):
        events, callback_errors = [], []
        sampler = None
        sample_path = a.output / f"{run_index}.sample.txt"
        @callback_type
        def callback(raw, _):
            nonlocal sampler
            try:
                event = json.loads(raw)
                event["observed_monotonic_ns"] = time.perf_counter_ns()
                if a.sample_run is not None and run_index == a.sample_run and sampler is None and event["phase"] == "denoise" and event["completed"] == 0:
                    if sample_path.exists():
                        raise ValueError("sample output already exists")
                    sampler = subprocess.Popen(["/usr/bin/sample", str(os.getpid()), "1", "1", "-mayDie", "-file", str(sample_path.resolve())],
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if event["phase"] not in ("transformer_block", "z_image_denoise_block"):
                    events.append(event)
            except Exception as error:
                callback_errors.append(str(error))
        output, error = C.c_void_p(), C.c_void_p()
        observed_start = time.perf_counter_ns()
        if prepare is None:
            status = lib.tc_engine_generate(engine, json.dumps(value).encode(), callback, None,
                                            C.byref(output), C.byref(error))
        else:
            status = lib.tc_engine_prepare(engine, json.dumps(value).encode(), int(prepare == "warmup"),
                                           callback, None, C.byref(output), C.byref(error))
        observed_end = time.perf_counter_ns()
        wall = (observed_end - observed_start) / 1e9
        text, message = consume(output), consume(error)
        if sampler is not None and sampler.wait(timeout=10) != 0:
            raise RuntimeError("macOS sample failed; do not treat this as a valid profile")
        if status or callback_errors:
            raise RuntimeError(message or callback_errors)
        row = dict(**summarize(events, wall), metrics=json.loads(text), events=events,
                   observed_start_ns=observed_start, observed_end_ns=observed_end)
        if a.metal_trace:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "profiling/image_startup"))
            from metal_trace import summarize_trace
            row["metal_trace"] = summarize_trace(a.metal_trace, observed_start, observed_end, events, os.getpid())
        if sampler is not None:
            row["host_sample"] = str(sample_path.resolve())
        return row

    try:
        if a.prepare != "none":
            report["preparation"] = invoke(request, a.prepare)
        for index in range(a.runs):
            changed = (index // 2) % 2 == 1
            value = dict(request, prompt=base_prompt + (" Soft studio lighting." if changed else ""),
                         output=str((a.output / f"{index}.png").resolve()))
            print(f"START {a.model_id} {value['operation']} run={index} changed_prompt={changed}", flush=True)
            row = invoke(value, run_index=index)
            row.update(index=index, changed_prompt=changed, request=value,
                       png_sha256=hashlib.sha256(Path(value["output"]).read_bytes()).hexdigest())
            report["runs"].append(row)
            (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({key: row[key] for key in ["index", "first_step_seconds", "later_step_median_seconds",
                                                       "time_to_denoise_seconds", "request_wall_seconds"]}), flush=True)
    finally:
        lib.tc_engine_free(engine)


if __name__ == "__main__":
    main()
