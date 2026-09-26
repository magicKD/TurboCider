"""Paired native Z-Image residency measurements; no model downloads or app state writes."""
import argparse
import ctypes as C
import hashlib
import json
import os
import re
import resource
import subprocess
import time
from pathlib import Path


def vm_counters():
    text = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
    page = int(re.search(r"page size of (\d+) bytes", text)[1])
    return {key: int(re.search(rf"^{key}:\s*(\d+)", text, re.M)[1]) * page
            for key in ("Swapins", "Swapouts", "Pageins", "Compressions", "Decompressions")}


def process_memory():
    # Public Darwin rusage_info_v4 layout from <sys/resource.h>. Unlike maxrss,
    # phys_footprint includes memory charged to this task outside ordinary RSS.
    fields = """user_time system_time pkg_idle_wkups interrupt_wkups pageins
        wired_size resident_size phys_footprint proc_start_abstime proc_exit_abstime
        child_user_time child_system_time child_pkg_idle_wkups child_interrupt_wkups
        child_pageins child_elapsed_abstime diskio_bytesread diskio_byteswritten
        cpu_time_qos_default cpu_time_qos_maintenance cpu_time_qos_background
        cpu_time_qos_utility cpu_time_qos_legacy cpu_time_qos_user_initiated
        cpu_time_qos_user_interactive billed_system_time serviced_system_time
        logical_writes lifetime_max_phys_footprint instructions cycles billed_energy
        serviced_energy interval_max_phys_footprint runnable_time""".split()

    class Usage(C.Structure):
        _fields_ = [("uuid", C.c_uint8 * 16)] + [(key, C.c_uint64) for key in fields]

    libproc = C.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    libproc.proc_pid_rusage.argtypes = [C.c_int, C.c_int, C.c_void_p]
    info = Usage()
    if libproc.proc_pid_rusage(os.getpid(), 4, C.byref(info)) != 0:
        raise OSError(C.get_errno(), "proc_pid_rusage failed")
    return {key + "_bytes": getattr(info, key) for key in
            ("resident_size", "phys_footprint", "lifetime_max_phys_footprint",
             "diskio_bytesread", "diskio_byteswritten")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--library", default="build/native/libturbocider.dylib")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--modes", default="resident,streamed,streamed,resident")
    p.add_argument("--budget-gib", type=int, default=10)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default="A red fox walking through fresh snow in a pine forest, soft morning sunlight, detailed photography.")
    p.add_argument("--execution", choices=("gpu", "gpu_ane"), default="gpu",
                   help="hybrid requires a library that explicitly supports streaming + Core ML")
    p.add_argument("--ane-manifest", type=Path)
    p.add_argument("--lifecycle", action="store_true", help="also test prepare, cancellation and retry")
    p.add_argument("--check-cache-scope", action="store_true", help="verify streaming restores the existing MLX cache limit")
    a = p.parse_args()
    if (a.execution == "gpu_ane") != (a.ane_manifest is not None):
        p.error("--execution gpu_ane requires --ane-manifest; GPU comparisons must omit it")
    cache_probe = None
    if a.check_cache_scope:
        import mlx.core as mx
        previous_cache_limit = mx.set_cache_limit(512 << 20)
        mx.set_cache_limit(previous_cache_limit)

        def cache_probe():
            observed = mx.set_cache_limit(previous_cache_limit)
            return observed == previous_cache_limit
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    lib = C.CDLL(str(Path(a.library).resolve()))
    lib.tc_string_free.argtypes = [C.c_void_p]
    lib.tc_engine_create_model.argtypes = [C.c_char_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    lib.tc_engine_generate.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    lib.tc_engine_prepare.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.c_void_p, C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    lib.tc_engine_cancel.argtypes = [C.c_void_p]
    lib.tc_engine_free.argtypes = [C.c_void_p]

    def take(value):
        if not value.value:
            return None
        text = C.string_at(value).decode()
        lib.tc_string_free(value)
        return text

    engine, error = C.c_void_p(), C.c_void_p()
    status = lib.tc_engine_create_model(b"z-image-turbo", a.model.encode(), C.byref(engine), C.byref(error))
    if status:
        raise RuntimeError(take(error))
    report = {"library": str(Path(a.library).resolve()), "runs": [],
              "notes": "Swap deltas are system-wide, not attribution to this process. maxrss is the process lifetime peak."}
    base = dict(model="z-image-turbo", prompt=a.prompt,
                output="", width=a.size, height=a.size, steps=a.steps, seed=a.seed,
                execution=a.execution, audio=False, dynamic_text=True)
    if a.ane_manifest:
        base.update(ane_manifest=str(a.ane_manifest.resolve()), allow_approximation=True)
    callback = C.CFUNCTYPE(None, C.c_char_p, C.c_void_p)

    def run(mode, name, prepare=False, cancel_phase=None):
        request = dict(base, residency=mode, output=str(a.output / f"{name}.png"))
        if mode == "streamed":
            request["memory_budget_bytes"] = a.budget_gib << 30
        events = []
        step_intervals = []
        last_step, last_step_time = -1, None

        @callback
        def event(raw, _):
            nonlocal last_step, last_step_time
            value = json.loads(raw)
            if value["phase"] in ("denoise", "load_z_image_stream", "pack_z_image_suffix", "export"):
                events.append(value)
            if value["phase"] == "denoise" and value["completed"] > last_step:
                now = time.monotonic()
                if last_step_time is not None:
                    step_intervals.append(now - last_step_time)
                last_step, last_step_time = value["completed"], now
            cancel_at = 1 if cancel_phase == "pack_z_image_suffix" else 20
            if cancel_phase and value["phase"] == cancel_phase and value["completed"] == cancel_at:
                lib.tc_engine_cancel(engine)

        before = vm_counters()
        process_before = process_memory()
        started = time.monotonic()
        result, error = C.c_void_p(), C.c_void_p()
        if prepare:
            status = lib.tc_engine_prepare(engine, json.dumps(request).encode(), 0, event, None, C.byref(result), C.byref(error))
        else:
            status = lib.tc_engine_generate(engine, json.dumps(request).encode(), event, None, C.byref(result), C.byref(error))
        elapsed = time.monotonic() - started
        data, message = take(result), take(error)
        after = vm_counters()
        process_after = process_memory()
        output = Path(request["output"])
        row = dict(name=name, request=request, wall_seconds=elapsed, status=status, error=message,
                   metrics=json.loads(data) if data else None, events=events,
                   step_intervals_seconds=step_intervals,
                   process_lifetime_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   process_memory=process_after,
                   process_disk_delta_bytes={key: process_after[key + "_bytes"] - process_before[key + "_bytes"]
                                             for key in ("diskio_bytesread", "diskio_byteswritten")},
                   system_vm_delta_bytes={k: after[k] - before[k] for k in before},
                   png_sha256=hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None)
        if cache_probe:
            row["cache_limit_restored"] = cache_probe()
        report["runs"].append(row)
        (a.output / "report.json").write_text(json.dumps(report, indent=2))
        print(name, f"{elapsed:.2f}s", message or "ok", flush=True)
        if cache_probe:
            assert row["cache_limit_restored"], "streaming changed the process-wide cache limit"
        if cancel_phase:
            assert status != 0 and "cancel" in (message or "").lower() and not output.exists(), row
        else:
            assert status == 0, row
        if prepare:
            assert not output.exists() and row["metrics"]["block_residency"]["enabled"], row
        return row

    try:
        for i, mode in enumerate(a.modes.split(",")):
            run(mode, f"{i:02d}-{mode}")
        if a.lifecycle:
            run("streamed", "prepare", prepare=True)
            run("streamed", "cancel", cancel_phase="z_image_denoise_block")
            run("streamed", "retry")
            base["prompt"] += " A small wooden cabin in the background."
            run("streamed", "changed-prompt-streamed")
            run("resident", "changed-prompt-resident")
            if a.execution == "gpu_ane":
                run("streamed", "cancel-pack", cancel_phase="pack_z_image_suffix")
                run("streamed", "retry-pack")
                base.update(execution="gpu", allow_approximation=False)
                manifest = base.pop("ane_manifest")
                run("streamed", "switch-gpu-streamed")
                run("resident", "switch-gpu-resident")
                base.update(execution="gpu_ane", allow_approximation=True, ane_manifest=manifest)
                run("streamed", "switch-back-hybrid")
        groups = {}
        for row in report["runs"]:
            if row["png_sha256"]:
                key = (row["request"]["prompt"], row["request"]["execution"])
                groups.setdefault(key, set()).add(row["png_sha256"])
        report["all_images_byte_identical"] = all(len(hashes) == 1 for hashes in groups.values())
        (a.output / "report.json").write_text(json.dumps(report, indent=2))
        assert report["all_images_byte_identical"], "resident/streamed image parity failed"
    finally:
        lib.tc_engine_free(engine)


if __name__ == "__main__":
    main()
