"""Paired native Z-Image residency measurements; no model downloads or app state writes."""
import argparse
import ctypes as C
import hashlib
import json
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
    p.add_argument("--lifecycle", action="store_true", help="also test prepare, cancellation and retry")
    p.add_argument("--check-cache-scope", action="store_true", help="verify streaming restores the existing MLX cache limit")
    a = p.parse_args()
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
    base = dict(model="z-image-turbo", prompt="A red fox walking through fresh snow in a pine forest, soft morning sunlight, detailed photography.",
                output="", width=a.size, height=a.size, steps=a.steps, seed=a.seed,
                execution="gpu", audio=False, dynamic_text=True)
    callback = C.CFUNCTYPE(None, C.c_char_p, C.c_void_p)

    def run(mode, name, prepare=False, cancel=False):
        request = dict(base, residency=mode, output=str(a.output / f"{name}.png"))
        if mode == "streamed":
            request["memory_budget_bytes"] = a.budget_gib << 30
        events = []

        @callback
        def event(raw, _):
            value = json.loads(raw)
            if value["phase"] in ("denoise", "load_z_image_stream", "export"):
                events.append(value)
            if cancel and value["phase"] == "z_image_denoise_block" and value["completed"] == 20:
                lib.tc_engine_cancel(engine)

        before = vm_counters()
        started = time.monotonic()
        result, error = C.c_void_p(), C.c_void_p()
        if prepare:
            status = lib.tc_engine_prepare(engine, json.dumps(request).encode(), 0, event, None, C.byref(result), C.byref(error))
        else:
            status = lib.tc_engine_generate(engine, json.dumps(request).encode(), event, None, C.byref(result), C.byref(error))
        elapsed = time.monotonic() - started
        data, message = take(result), take(error)
        after = vm_counters()
        output = Path(request["output"])
        row = dict(name=name, request=request, wall_seconds=elapsed, status=status, error=message,
                   metrics=json.loads(data) if data else None, events=events,
                   process_lifetime_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   system_vm_delta_bytes={k: after[k] - before[k] for k in before},
                   png_sha256=hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None)
        if cache_probe:
            row["cache_limit_restored"] = cache_probe()
        report["runs"].append(row)
        (a.output / "report.json").write_text(json.dumps(report, indent=2))
        print(name, f"{elapsed:.2f}s", message or "ok", flush=True)
        if cache_probe:
            assert row["cache_limit_restored"], "streaming changed the process-wide cache limit"
        if cancel:
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
            run("streamed", "cancel", cancel=True)
            run("streamed", "retry")
            base["prompt"] += " A small wooden cabin in the background."
            run("streamed", "changed-prompt-streamed")
            run("resident", "changed-prompt-resident")
        groups = {}
        for row in report["runs"]:
            if row["png_sha256"]:
                groups.setdefault(row["request"]["prompt"], set()).add(row["png_sha256"])
        report["all_images_byte_identical"] = all(len(hashes) == 1 for hashes in groups.values())
        (a.output / "report.json").write_text(json.dumps(report, indent=2))
        assert report["all_images_byte_identical"], "resident/streamed image parity failed"
    finally:
        lib.tc_engine_free(engine)


if __name__ == "__main__":
    main()
