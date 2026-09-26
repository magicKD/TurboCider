"""Real-weight M5 Pro auto/explicit parity and fallback integration probe.

Run on the measured M5 Pro 24 GiB with locally compiled FLUX partitions.
This command is separate from the model-free unit tests.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path


def run(args: argparse.Namespace) -> dict:
    library = C.CDLL(str(args.library.resolve()))
    library.tc_string_free.argtypes = [C.c_void_p]
    library.tc_engine_free.argtypes = [C.c_void_p]
    library.tc_system_json.restype = C.c_void_p
    library.tc_engine_create_model.argtypes = [
        C.c_char_p, C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    common = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_void_p,
              C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    library.tc_engine_generate.argtypes = common
    library.tc_engine_prepare.argtypes = common[:2] + [C.c_int] + common[2:]

    def take(pointer):
        if not pointer:
            return ""
        value = C.string_at(pointer).decode()
        library.tc_string_free(pointer)
        return value

    system = json.loads(take(library.tc_system_json()))
    assert system["gpu"] == "Apple M5 Pro" and system["physical_memory_bytes"] == 24 << 30
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    engine, error = C.c_void_p(), C.c_void_p()
    status = library.tc_engine_create_model(b"flux2-klein-4b", str(args.model.resolve()).encode(),
                                          C.byref(engine), C.byref(error))
    failure = take(error)
    if status:
        raise RuntimeError(failure)

    def call(request, prepare=False):
        result, error = C.c_void_p(), C.c_void_p()
        parameters = [engine, json.dumps(request).encode()]
        if prepare:
            parameters.append(0)
        parameters += [None, None, C.byref(result), C.byref(error)]
        function = library.tc_engine_prepare if prepare else library.tc_engine_generate
        status = function(*parameters)
        value, failure = take(result), take(error)
        if status:
            raise RuntimeError(failure)
        return json.loads(value)

    request = dict(model="flux2-klein-4b", operation="image.generate",
                   prompt="A red fox in a snowy forest.", width=512, height=512,
                   steps=4, seed=42, dynamic_text=True, execution="auto",
                   residency="resident", allow_approximation=True,
                   ane_manifest=str(args.manifest.resolve()))
    reports = {}
    try:
        for mode in ("auto", "gpu_ane"):
            reports[mode] = call(dict(request, execution=mode, output=str(output / f"{mode}.png")))
        automatic = reports["auto"]
        assert automatic["acceleration_selection"] == "gpu_ane: measured case m5pro24-flux4b-t2i-512-a6144-v1"
        assert automatic["hybrid"]["ane_mlp_range"] == [0, 6144]
        assert automatic["hybrid"]["runtime_calls_session_total"] == 80
        assert reports["gpu_ane"]["hybrid"]["runtime_calls_session_total"] == 160
        assert reports["gpu_ane"]["hybrid"]["output_copy_bytes_session_total"] == 0
        assert hashlib.sha256((output / "auto.png").read_bytes()).digest() == hashlib.sha256(
            (output / "gpu_ane.png").read_bytes()).digest()
        # Exercise real runtime decisions, including rejection of a valid but
        # unqualified full-width manifest, without generating larger images.
        malformed = output / "malformed.json"
        malformed.write_text('{"schema_version":2,"shape":null}')
        fallbacks = {
            "missing": dict(ane_manifest=str(output / "missing.json")),
            "malformed": dict(ane_manifest=str(malformed)),
            "approximation_disabled": dict(allow_approximation=False),
            "small_image": dict(width=256, height=256),
            "different_steps": dict(steps=1),
            "staged": dict(residency="component_staged"),
            "hybrid_budget_unavailable": dict(memory_budget_bytes=17 << 30),
            "long_text_bucket": dict(dynamic_text=False),
            "unqualified_partition": dict(ane_manifest=str(args.other_manifest.resolve())),
        }
        for name, changes in fallbacks.items():
            reports[name] = call(dict(request, **changes), prepare=True)
            assert reports[name]["execution"] == "gpu", name
        assert "6144-channel" in reports["unqualified_partition"]["acceleration_selection"]
        # A failed automatic candidate must not poison the next valid selection.
        reports["recovered"] = call(request, prepare=True)
        assert reports["recovered"]["execution"] == "gpu_ane"
        try:
            call(dict(request, execution="gpu_ane", ane_manifest=str(output / "missing.json")), prepare=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError("explicit hybrid silently fell back")
        report = dict(passed=True, system=system, reports=reports,
                      checks=["auto_explicit_png_identical", "session_reused", "zero_output_copy",
                              *fallbacks, "recovery", "explicit_failure_is_strict"])
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        library.tc_engine_free(engine)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path("build/native/libturbocider.dylib"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--other-manifest", type=Path, required=True,
                        help="valid local FLUX 1088-row partition with a different ANE width")
    parser.add_argument("--output", type=Path, required=True)
    report = run(parser.parse_args())
    print(json.dumps({key: report[key] for key in ("passed", "checks")}))


if __name__ == "__main__":
    main()
