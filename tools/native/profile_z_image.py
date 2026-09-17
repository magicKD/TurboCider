"""Compare uninstrumented, overlapping and serialized Z-Image streaming.

Stage traces use host completion times, not hardware kernel timestamps. Serial
and split modes intentionally change scheduling; controls quantify their limits.
The client installation and its settings are not changed.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
import time
from pathlib import Path

from benchmark_z_image_streaming import process_memory, vm_counters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-gib", type=int, default=10)
    parser.add_argument("--experiment", choices=("stages", "fusion", "fusion-smoke"), default="stages",
                        help="fusion alternates eager/fused hybrid segments in the same loaded session")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trace = output / "blocks.jsonl"
    if trace.exists() or (output / "report.json").exists():
        parser.error("Use a new output directory to keep trace/request identities unambiguous")
    lib = C.CDLL(str(args.library.resolve()))
    lib.tc_string_free.argtypes = [C.c_void_p]
    lib.tc_engine_create_model.argtypes = [C.c_char_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    lib.tc_engine_generate.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    lib.tc_engine_free.argtypes = [C.c_void_p]

    def take(value):
        if not value.value:
            return None
        data = C.string_at(value).decode()
        lib.tc_string_free(value)
        return data

    engine, error = C.c_void_p(), C.c_void_p()
    if lib.tc_engine_create_model(b"z-image-turbo", str(args.model.resolve()).encode(), C.byref(engine), C.byref(error)):
        raise RuntimeError(take(error))
    base = dict(model="z-image-turbo", prompt=args.prompt_file.read_text().strip(),
                width=args.size, height=args.size, steps=args.steps, seed=args.seed,
                audio=False, dynamic_text=True, residency="streamed",
                memory_budget_bytes=args.budget_gib << 30)
    report = dict(library=str(args.library.resolve()), runs=[],
                  notes=["Host wall-clock timing with explicit completion boundaries; not pure hardware timing.",
                         "System VM deltas are global; no per-process swap attribution.",
                         "Core ML metrics are cumulative within each hybrid session.",
                         "gpu_split changes the compiled graph and can change floating-point rounding."])
    # Stay in one session to reuse text and shapes. Bracket diagnostics with
    # controls; exclude only explicitly identified route/graph warmup runs.
    cases = [("gpu_control_start", "gpu", None, 1, 2),
             ("hybrid_control_start", "gpu_ane", None, 1, 2),
             ("hybrid_overlap", "gpu_ane", "overlap", 0, 2),
             ("hybrid_serial", "gpu_ane", "serial", 0, 2),
             ("hybrid_control_end", "gpu_ane", None, 0, 2),
             ("gpu_split", "gpu", "gpu_split", 1, 2),
             ("gpu_fused_profile", "gpu", "overlap", 0, 2),
             ("gpu_control_end", "gpu", None, 0, 2)]
    cases = [(*case, False) for case in cases]
    if args.experiment == "fusion-smoke":
        cases = [("hybrid_eager", "gpu_ane", None, 1, 0, True),
                 ("hybrid_fused", "gpu_ane", None, 1, 0, False),
                 ("hybrid_eager_retry", "gpu_ane", None, 0, 1, True)]
    elif args.experiment == "fusion":
        cases = [("gpu_control_start", "gpu", None, 1, 2, False),
                 ("hybrid_eager_warm", "gpu_ane", None, 1, 0, True),
                 ("hybrid_fused_warm", "gpu_ane", None, 1, 0, False),
                 ("hybrid_eager_a", "gpu_ane", None, 0, 2, True),
                 ("hybrid_fused_a", "gpu_ane", None, 0, 2, False),
                 ("hybrid_fused_b", "gpu_ane", None, 0, 2, False),
                 ("hybrid_eager_b", "gpu_ane", None, 0, 2, True),
                 ("hybrid_eager_profile", "gpu_ane", "overlap", 0, 2, True),
                 ("hybrid_fused_profile", "gpu_ane", "overlap", 0, 2, False),
                 ("gpu_control_end", "gpu", None, 1, 2, False)]
    callback = C.CFUNCTYPE(None, C.c_char_p, C.c_void_p)
    trace_request = 0
    try:
        for case, execution, mode, warmups, repeats, eager_segments in cases:
            for index in range(warmups + repeats):
                name = f"{len(report['runs']):02d}-{case}"
                if mode:
                    os.environ["TURBOCIDER_Z_PROFILE"] = str(trace)
                    os.environ["TURBOCIDER_Z_PROFILE_MODE"] = mode
                    trace_request += 1
                else:
                    os.environ.pop("TURBOCIDER_Z_PROFILE", None)
                    os.environ.pop("TURBOCIDER_Z_PROFILE_MODE", None)
                if eager_segments:
                    os.environ["TURBOCIDER_Z_HYBRID_EAGER_SEGMENTS"] = "1"
                else:
                    os.environ.pop("TURBOCIDER_Z_HYBRID_EAGER_SEGMENTS", None)
                request = dict(base, execution=execution, output=str(output / (name + ".png")))
                if execution == "gpu_ane":
                    request.update(ane_manifest=str(args.manifest.resolve()), allow_approximation=True)
                last_step, last_time = -1, None
                intervals = []
                phase_vm = {}

                @callback
                def event(raw, _):
                    nonlocal last_step, last_time
                    value = json.loads(raw)
                    phase = None
                    if value["phase"] == "denoise" and value["completed"] == 0:
                        phase = "denoise_begin"
                    elif value["phase"] == "denoise" and value["completed"] == args.steps:
                        phase = "denoise_end"
                    elif value["phase"] == "export" and value["completed"] == 0:
                        phase = "export_begin"
                    if phase and phase not in phase_vm:
                        phase_vm[phase] = dict(seconds=time.monotonic() - start,
                                              system_vm=vm_counters(), process_memory=process_memory())
                    if value["phase"] == "denoise" and value["completed"] > last_step:
                        now = time.monotonic()
                        if last_time is not None:
                            intervals.append(now - last_time)
                        last_step, last_time = value["completed"], now

                before_vm, before_process = vm_counters(), process_memory()
                start = time.monotonic()
                result, error = C.c_void_p(), C.c_void_p()
                status = lib.tc_engine_generate(engine, json.dumps(request).encode(), event, None,
                                                C.byref(result), C.byref(error))
                elapsed = time.monotonic() - start
                data, message = take(result), take(error)
                after_process, after_vm = process_memory(), vm_counters()
                image = Path(request["output"])
                row = dict(name=name, case=case, execution=execution, profile_mode=mode,
                           hybrid_eager_segments=eager_segments, phase_vm=phase_vm,
                           warmup=index < warmups, trace_request=trace_request if mode else None,
                           request=request, wall_seconds=elapsed, status=status, error=message,
                           metrics=json.loads(data) if data else None, step_intervals_seconds=intervals,
                           process_memory=after_process,
                           process_disk_delta_bytes={k: after_process[k + "_bytes"] - before_process[k + "_bytes"]
                                                     for k in ("diskio_bytesread", "diskio_byteswritten")},
                           system_vm_delta_bytes={k: after_vm[k] - before_vm[k] for k in before_vm},
                           png_sha256=hashlib.sha256(image.read_bytes()).hexdigest() if image.exists() else None)
                report["runs"].append(row)
                (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(name, "warmup" if row["warmup"] else "sample", f"{elapsed:.3f}s", message or "ok", flush=True)
                if status:
                    raise RuntimeError(message)
                if args.experiment != "stages":
                    hashes = {r["png_sha256"] for r in report["runs"] if r["execution"] == execution}
                    if len(hashes) != 1:
                        raise AssertionError(f"Fusion changed {execution} output; inspect the saved comparison")
        # Instrumentation must preserve both production routes' exact output.
        for execution in {r["execution"] for r in report["runs"]}:
            hashes = {r["png_sha256"] for r in report["runs"]
                      if r["execution"] == execution and r["profile_mode"] != "gpu_split"}
            if len(hashes) != 1:
                raise AssertionError(f"Instrumentation changed {execution} output")
        report["instrumented_output_parity"] = True
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        os.environ.pop("TURBOCIDER_Z_PROFILE", None)
        os.environ.pop("TURBOCIDER_Z_PROFILE_MODE", None)
        os.environ.pop("TURBOCIDER_Z_HYBRID_EAGER_SEGMENTS", None)
        lib.tc_engine_free(engine)


if __name__ == "__main__":
    main()
