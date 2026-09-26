"""Real-weight automatic selection, fallback and explicit-route parity.

This is a command-line integration probe rather than a pytest test module. It
keeps the historical invocation contract while avoiding work during pytest
collection.
"""

from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-1024", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    root = Path(__file__).resolve().parents[2]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    library = C.CDLL(str(root / "build/native/libturbocider.dylib"))
    library.tc_string_free.argtypes = [C.c_void_p]
    library.tc_engine_free.argtypes = [C.c_void_p]
    library.tc_engine_create.argtypes = [
        C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)
    ]
    common = [
        C.c_void_p, C.c_char_p, C.c_void_p, C.c_void_p,
        C.POINTER(C.c_void_p), C.POINTER(C.c_void_p),
    ]
    library.tc_engine_generate.argtypes = common
    library.tc_engine_prepare.argtypes = common[:2] + [C.c_int] + common[2:]

    def take(pointer: C.c_void_p) -> str:
        if not pointer.value:
            return ""
        value = C.string_at(pointer).decode()
        library.tc_string_free(pointer)
        return value

    engine, error = C.c_void_p(), C.c_void_p()
    status = library.tc_engine_create(
        str(args.model).encode(), C.byref(engine), C.byref(error)
    )
    if status:
        raise RuntimeError(take(error))

    def call(request: dict, prepare: bool = False) -> dict:
        result, error = C.c_void_p(), C.c_void_p()
        call_args = [engine, json.dumps(request).encode()]
        if prepare:
            call_args.append(0)
        call_args.extend([None, None, C.byref(result), C.byref(error)])
        function = library.tc_engine_prepare if prepare else library.tc_engine_generate
        status = function(*call_args)
        value, failure = take(result), take(error)
        if status:
            raise RuntimeError(failure)
        return json.loads(value)

    request = {
        "model": "flux2-klein-4b",
        "prompt": "A red fox in a snowy forest.",
        "width": 512,
        "height": 512,
        "steps": 4,
        "seed": 42,
        "execution": "auto",
        "allow_approximation": True,
        "ane_manifest": args.manifest,
    }
    reports, checks = {}, []
    try:
        reports["prepare"] = call(request, True)
        assert reports["prepare"]["execution"] == "gpu_ane"
        checks.append("hardware_checkpoint_and_bucket_match")

        request["output"] = str(output / "auto.png")
        reports["auto"] = call(request)
        assert reports["auto"]["hybrid"]["calls_session_total"] == 80

        request.update(execution="gpu_ane", output=str(output / "explicit.png"))
        reports["explicit"] = call(request)
        assert reports["explicit"]["hybrid"]["calls_session_total"] == 160

        def sha(name: str) -> str:
            return hashlib.sha256((output / name).read_bytes()).hexdigest()

        assert sha("auto.png") == sha("explicit.png")
        checks.extend(["hybrid_session_reused", "auto_explicit_png_identical"])

        request.update(execution="auto", width=768, height=768)
        reports["oversize"] = call(request, True)
        assert reports["oversize"]["execution"] == "gpu"
        assert "bucket" in reports["oversize"]["acceleration_selection"]
        checks.append("oversize_bucket_falls_back")

        request.update(width=512, height=512, ane_manifest=str(output / "missing.json"))
        reports["missing"] = call(request, True)
        assert reports["missing"]["execution"] == "gpu"
        checks.append("missing_manifest_falls_back")

        malformed = output / "malformed.json"
        malformed.write_text('{"schema_version":2,"shape":null}')
        request["ane_manifest"] = str(malformed)
        reports["malformed"] = call(request, True)
        assert reports["malformed"]["execution"] == "gpu"
        checks.append("malformed_manifest_falls_back")

        request.update(ane_manifest=args.manifest, allow_approximation=False)
        reports["exact_only"] = call(request, True)
        assert reports["exact_only"]["execution"] == "gpu"
        checks.append("approximation_opt_in_required")

        request.update(width=1024, height=1024, allow_approximation=True)
        reports["1024"] = call(request, True)
        assert reports["1024"]["execution"] == "gpu"
        assert "bucket" in reports["1024"]["acceleration_selection"]
        checks.append("1024_does_not_inherit_512_route")

        request.update(
            ane_manifest=args.manifest_1024,
            output=str(output / "auto-1024.png"),
        )
        reports["auto1024"] = call(request)
        assert reports["auto1024"]["plan"]["execution"].startswith("gpu_ane")
        assert reports["auto1024"]["hybrid"]["bucket"] == 4160

        request.update(execution="gpu_ane", output=str(output / "explicit-1024.png"))
        reports["explicit1024"] = call(request)
        assert sha("auto-1024.png") == sha("explicit-1024.png")
        checks.append("1024_independent_bucket_and_explicit_parity")

        request.update(execution="auto", width=256, height=256, ane_manifest=args.manifest)
        reports["padding"] = call(request, True)
        assert reports["padding"]["execution"] == "gpu"
        checks.append("small_image_avoids_unmeasured_padding")

        request.update(width=512, height=512, steps=1)
        reports["steps"] = call(request, True)
        assert reports["steps"]["execution"] == "gpu"
        checks.append("different_steps_require_measurement")

        request.update(
            steps=4,
            ane_manifest=str(output / "missing.json"),
            execution="gpu_ane",
            allow_approximation=True,
        )
        try:
            call(request, True)
            raise AssertionError("explicit hybrid silently fell back")
        except RuntimeError:
            checks.append("explicit_hybrid_still_strict")

        reports["png_sha256"] = sha("auto.png")
        report = {"passed": True, "checks": checks, "reports": reports}
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"passed": True, "checks": checks}))
        return report
    finally:
        library.tc_engine_free(engine)


def main() -> None:
    run(arguments())


if __name__ == "__main__":
    main()
