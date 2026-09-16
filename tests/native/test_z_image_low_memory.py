"""Real-weight regression probe for the qualified Z-Image M5 memory policy.

Checks resident cache scoping, prompt-stage memory, cancellation/retry and PNG
parity. Run explicitly with a local Comfy BF16 model; no downloads are performed.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, default=Path("build/native/libturbocider.dylib"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="also verify M5 automatic GPU fallback")
    args = parser.parse_args()
    import mlx.core as mx

    lib = C.CDLL(str(args.library.resolve()))
    lib.tc_string_free.argtypes = [C.c_void_p]
    lib.tc_system_json.restype = C.c_void_p
    lib.tc_engine_create_model.argtypes = [C.c_char_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    common = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    lib.tc_engine_generate.argtypes = common
    lib.tc_engine_prepare.argtypes = common[:2] + [C.c_int] + common[2:]
    lib.tc_engine_cancel.argtypes = [C.c_void_p]
    lib.tc_engine_free.argtypes = [C.c_void_p]

    def take(pointer):
        if not pointer:
            return None
        value = C.string_at(pointer).decode()
        lib.tc_string_free(pointer)
        return value

    system = json.loads(take(lib.tc_system_json()))
    assert system.get("optimization_profile", {}).get("z_image_memory_lifecycle"), \
        "this regression probe requires the measured M5 Pro 24 GiB device policy"
    assert 0 < system["physical_memory_bytes"] < 32 << 30, system
    if args.manifest:
        assert system["gpu"] == "Apple M5 Pro", "fallback check targets M5 Pro"
    args.output.mkdir(parents=True, exist_ok=False)
    base = dict(model="z-image-turbo", operation="image.generate", audio=False,
                prompt="A red fox in a snowy forest.", width=256, height=256,
                steps=2, seed=42, execution="gpu", residency="resident", dynamic_text=True)
    sentinel = 1 << 30
    original_cache = mx.set_cache_limit(sentinel)
    engine = C.c_void_p()
    reports = []
    callback = C.CFUNCTYPE(None, C.c_char_p, C.c_void_p)

    def create():
        error = C.c_void_p()
        status = lib.tc_engine_create_model(b"z-image-turbo", str(args.model.resolve()).encode(), C.byref(engine), C.byref(error))
        assert status == 0, take(error)

    def run(name, *, prompt=None, cancel_phase=None, prepare=False, **changes):
        output = (args.output / f"{name}.png").resolve()
        request = dict(base, output=str(output), **changes)
        if prompt:
            request["prompt"] = prompt
        observations = []
        phases = []

        @callback
        def event(raw, _):
            value = json.loads(raw)
            phase = value["phase"]
            phases.append(phase)
            if phase in ("route_gpu", "z_image_text_encode", "denoise") and value["completed"] <= 1:
                observed = mx.set_cache_limit(sentinel)
                mx.set_cache_limit(observed)
                observations.append(dict(phase=phase, cache_limit=observed, active=mx.get_active_memory()))
            if phase == cancel_phase and value["completed"] == 1:
                lib.tc_engine_cancel(engine)

        result, error = C.c_void_p(), C.c_void_p()
        parameters = [engine, json.dumps(request).encode()]
        if prepare:
            parameters.append(0)
        parameters += [event, None, C.byref(result), C.byref(error)]
        status = (lib.tc_engine_prepare if prepare else lib.tc_engine_generate)(*parameters)
        text, failure = take(result), take(error)
        if not cancel_phase:
            assert status == 0, failure
        assert mx.set_cache_limit(sentinel) == sentinel, "request changed the caller's cache limit"
        assert observations and all(o["cache_limit"] == 512 << 20 for o in observations), observations
        text_observations = [o for o in observations if o["phase"] == "z_image_text_encode"]
        assert all(o["active"] < 9 << 30 for o in text_observations), "old denoiser overlapped Qwen3"
        if cancel_phase:
            assert status != 0 and "cancel" in (failure or "").lower() and not output.exists(), failure
            if cancel_phase == "z_image_text_encode":
                assert mx.get_active_memory() < 2 << 30, "cancelled prompt retained Qwen3"
        else:
            assert status == 0, failure
            assert prepare != output.exists()
        row = dict(name=name, metrics=json.loads(text) if text else None, error=failure,
                   cache_observations=observations, phases=sorted(set(phases)),
                   png_sha256=hashlib.sha256(output.read_bytes()).hexdigest() if output.exists() else None)
        reports.append(row)
        (args.output / "report.json").write_text(json.dumps(dict(passed=False, system=system, runs=reports), indent=2) + "\n")
        print(name, "passed", flush=True)
        return row

    try:
        create()
        first = run("first")
        repeated = run("repeated")
        assert repeated["png_sha256"] == first["png_sha256"]
        assert "z_image_text_cache_hit" in repeated["phases"]
        assert "load_z_image_transformer" not in repeated["phases"]
        run("prepared", prepare=True)
        if args.manifest:
            automatic = run("automatic", execution="auto", allow_approximation=True,
                            ane_manifest=str(args.manifest.resolve()))
            assert automatic["metrics"]["plan"]["execution"] == "gpu"
            assert automatic["png_sha256"] == first["png_sha256"]
        changed_prompt = "A ceramic teapot on a wooden table."
        run("cancel-text", prompt=changed_prompt, cancel_phase="z_image_text_encode")
        assert run("retry-old-prompt")["png_sha256"] == first["png_sha256"]
        run("cancel-denoise", cancel_phase="z_image_denoise_block")
        assert run("retry-denoise")["png_sha256"] == first["png_sha256"]
        changed = run("changed-prompt", prompt=changed_prompt)
        lib.tc_engine_free(engine)
        engine = C.c_void_p()
        create()
        fresh = run("fresh-prompt", prompt=changed_prompt)
        assert changed["png_sha256"] == fresh["png_sha256"]
        (args.output / "report.json").write_text(json.dumps(dict(passed=True, system=system, runs=reports), indent=2) + "\n")
    finally:
        if engine:
            lib.tc_engine_free(engine)
        mx.set_cache_limit(original_cache)


if __name__ == "__main__":
    main()
