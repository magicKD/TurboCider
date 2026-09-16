#!/usr/bin/env python3
"""Opt-in real-model lifecycle test for candidate-scoped LTX exact streaming.

This test never applies artificial memory pressure. It intentionally retains one
candidate engine across successes, shape changes, cancellation and a media
failure so request-owned exact state cannot hide behind process teardown.
"""

from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import shutil
from pathlib import Path


EVENT = C.CFUNCTYPE(None, C.c_char_p, C.c_void_p)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--prompt", default="A red fox running through snow")
    args = parser.parse_args()
    if not args.library.is_file():
        parser.error(f"missing library: {args.library}")
    if not args.model.is_dir():
        parser.error(f"missing model: {args.model}")
    return args


def load_library(path: Path):
    library = C.CDLL(str(path.resolve()))
    library.tc_engine_create_model_candidate.argtypes = [
        C.c_char_p,
        C.c_char_p,
        C.POINTER(C.c_void_p),
        C.POINTER(C.c_void_p),
    ]
    library.tc_engine_create_model_candidate.restype = C.c_int
    library.tc_engine_generate.argtypes = [
        C.c_void_p,
        C.c_char_p,
        EVENT,
        C.c_void_p,
        C.POINTER(C.c_void_p),
        C.POINTER(C.c_void_p),
    ]
    library.tc_engine_generate.restype = C.c_int
    library.tc_engine_cancel.argtypes = [C.c_void_p]
    library.tc_engine_free.argtypes = [C.c_void_p]
    library.tc_string_free.argtypes = [C.c_void_p]
    return library


def consume(library, pointer: C.c_void_p) -> str:
    if not pointer.value:
        return ""
    value = C.string_at(pointer).decode()
    library.tc_string_free(pointer)
    return value


def request_value(
    args: argparse.Namespace,
    name: str,
    width: int,
    height: int,
    *,
    missing_parent: bool = False,
) -> tuple[dict, Path, Path]:
    directory = args.output / name
    if missing_parent:
        blocker = directory / "blocked-parent"
        dump = directory / "tensors"
        directory.mkdir(parents=True, exist_ok=True)
        if blocker.is_dir():
            shutil.rmtree(blocker)
        elif blocker.exists():
            blocker.unlink()
        blocker.write_bytes(b"not a directory\n")
        destination = blocker / "video.mp4"
    else:
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "video.mp4"
        dump = directory / "tensors"
    request = {
        "schema_version": 2,
        "model": "ltx-2.5-distilled",
        "operation": "video.generate",
        "inputs": [{"kind": "text", "role": "prompt", "text": args.prompt}],
        "outputs": [{
            "kind": "video",
            "path": str(destination.resolve()),
            "width": width,
            "height": height,
            "frames": 9,
            "fps": 24,
            "audio": False,
        }],
        "sampling": {"seed": 42, "steps": 11},
        "execution": {
            "policy": "gpu",
            "ltx_backend": "c_metal",
            "ltx_fast_av": True,
            "streaming": {
                "schema_version": 1,
                "enabled": True,
                "selection": "manual",
                "retention": "request",
                "stages": {"denoiser": {
                    "residency": "streamed",
                    "block_group_size": 1,
                    "slot_count": 3,
                    "resident_prefix_blocks": 8,
                    "prefetch_distance": 2,
                    "io_workers": 3,
                }},
            },
        },
        "dump_tensors": str(dump.resolve()),
    }
    return request, destination, dump / "stage2_video.bf16"


def generate(library, engine, request: dict, cancel_phase: str | None = None):
    state = {"cancelled": False, "upsampled": False}

    @EVENT
    def callback(raw, _context):
        event = json.loads(raw.decode())
        phase = event.get("phase")
        if phase == "latent_upsample" and event.get("completed") == event.get("total"):
            state["upsampled"] = True
        should_cancel = (
            cancel_phase == "stage1" and phase == "ltx_block"
        ) or (
            cancel_phase == "stage2" and state["upsampled"] and phase == "ltx_block"
        ) or (
            cancel_phase == "video_vae" and phase == "video_vae" and
            event.get("completed") == 0
        )
        if should_cancel and not state["cancelled"]:
            state["cancelled"] = True
            library.tc_engine_cancel(engine)

    result, error = C.c_void_p(), C.c_void_p()
    status = library.tc_engine_generate(
        engine,
        json.dumps(request).encode(),
        callback,
        None,
        C.byref(result),
        C.byref(error),
    )
    value = consume(library, result)
    failure = consume(library, error)
    return status, json.loads(value) if value else None, failure, state


def successful(library, engine, request: dict, destination: Path, latent: Path):
    status, result, failure, _ = generate(library, engine, request)
    assert status == 0 and result is not None, failure
    assert destination.is_file() and destination.stat().st_size > 0
    assert latent.is_file() and latent.stat().st_size > 0
    streaming = result["plan"]["streaming"]
    actual = streaming["actual_layout"]
    assert streaming["execution_supported"] is True
    assert streaming["eligibility"] == "experimental_candidate"
    assert streaming["resolution_state"] == "executed_exact_v2"
    assert streaming["authority"] == "private_candidate_constructor"
    assert actual == streaming["resolved_layout"]
    assert actual["resident_prefix_blocks"] == 8
    assert actual["block_group_size"] == 1
    assert actual["slot_count"] == 3
    assert actual["prefetch_distance"] == 2
    assert actual["io_workers"] == 3
    assert actual["group_count"] == 40
    assert actual["pass_count"] == 11
    block = result["block_streaming"]
    assert block["implementation"] == "c_metal_exact_v2"
    assert block["layout_digest"] == actual["digest"]
    assert block["request_slot_allocations"] == 3
    assert block["request_slot_refills"] == 440
    digest = hashlib.sha256(latent.read_bytes()).hexdigest()
    return result, actual["digest"], digest


def cancelled(library, engine, request: dict, phase: str):
    status, result, failure, state = generate(
        library, engine, request, cancel_phase=phase
    )
    assert status != 0 and result is None
    assert state["cancelled"], f"{phase} cancellation callback did not fire"
    assert "cancel" in failure.lower(), failure


def main() -> int:
    args = arguments()
    args.output = args.output.resolve()
    args.cache = args.cache.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    # The session reads this cache through the existing explicit benchmark
    # contract. It does not change the layout or lifecycle authority.
    import os
    os.environ["TURBOCIDER_LTX_CONDITIONING_CACHE_DIR"] = str(args.cache)

    library = load_library(args.library)
    engine, error = C.c_void_p(), C.c_void_p()
    status = library.tc_engine_create_model_candidate(
        b"ltx-2.5-distilled",
        str(args.model.resolve()).encode(),
        C.byref(engine),
        C.byref(error),
    )
    failure = consume(library, error)
    assert status == 0 and engine.value, failure

    records = []
    try:
        cancel1, _, _ = request_value(args, "cancel-stage1", 64, 64)
        cancelled(library, engine, cancel1, "stage1")

        request_a0, output_a0, latent_a0 = request_value(args, "a0", 64, 64)
        result_a0, layout_a0, latent_hash = successful(
            library, engine, request_a0, output_a0, latent_a0
        )
        records.append({"case": "a0", "layout": layout_a0,
                        "wall": result_a0["timings_seconds"]["request_wall"]})

        request_a1, output_a1, latent_a1 = request_value(args, "a1", 64, 64)
        result_a1, layout_a1, hash_a1 = successful(
            library, engine, request_a1, output_a1, latent_a1
        )
        assert layout_a1 == layout_a0 and hash_a1 == latent_hash
        records.append({"case": "a1", "layout": layout_a1,
                        "wall": result_a1["timings_seconds"]["request_wall"]})

        request_b, output_b, latent_b = request_value(args, "b", 128, 64)
        result_b, layout_b, _ = successful(
            library, engine, request_b, output_b, latent_b
        )
        assert layout_b != layout_a0
        records.append({"case": "b", "layout": layout_b,
                        "wall": result_b["timings_seconds"]["request_wall"]})

        request_a2, output_a2, latent_a2 = request_value(args, "a2", 64, 64)
        result_a2, layout_a2, hash_a2 = successful(
            library, engine, request_a2, output_a2, latent_a2
        )
        assert layout_a2 == layout_a0 and hash_a2 == latent_hash
        records.append({"case": "a2", "layout": layout_a2,
                        "wall": result_a2["timings_seconds"]["request_wall"]})

        cancel2, _, _ = request_value(args, "cancel-stage2", 64, 64)
        cancelled(library, engine, cancel2, "stage2")
        recovery2, output_r2, latent_r2 = request_value(
            args, "recover-stage2", 64, 64
        )
        result_r2, layout_r2, hash_r2 = successful(
            library, engine, recovery2, output_r2, latent_r2
        )
        assert layout_r2 == layout_a0 and hash_r2 == latent_hash
        records.append({"case": "recover-stage2", "layout": layout_r2,
                        "wall": result_r2["timings_seconds"]["request_wall"]})

        cancel_vae, _, _ = request_value(args, "cancel-video-vae", 64, 64)
        cancelled(library, engine, cancel_vae, "video_vae")
        recovery_vae, output_rv, latent_rv = request_value(
            args, "recover-video-vae", 64, 64
        )
        result_rv, layout_rv, hash_rv = successful(
            library, engine, recovery_vae, output_rv, latent_rv
        )
        assert layout_rv == layout_a0 and hash_rv == latent_hash
        records.append({"case": "recover-video-vae", "layout": layout_rv,
                        "wall": result_rv["timings_seconds"]["request_wall"]})

        export_failure, _, _ = request_value(
            args, "export-failure", 64, 64, missing_parent=True
        )
        status, result, failure, _ = generate(library, engine, export_failure)
        assert status != 0 and result is None, result
        assert failure, "export failure returned no diagnostic"

        recovery_export, output_re, latent_re = request_value(
            args, "recover-export", 64, 64
        )
        result_re, layout_re, hash_re = successful(
            library, engine, recovery_export, output_re, latent_re
        )
        assert layout_re == layout_a0 and hash_re == latent_hash
        records.append({"case": "recover-export", "layout": layout_re,
                        "wall": result_re["timings_seconds"]["request_wall"]})
    finally:
        library.tc_engine_free(engine)

    summary = {
        "format": "turbocider-ltx-exact-lifecycle-v1",
        "successes": records,
        "stage2_latent_sha256": latent_hash,
        "layout_a": layout_a0,
        "layout_b": layout_b,
        "checks": [
            "cancel-stage1-then-success",
            "success-then-success",
            "shape-a-b-a",
            "cancel-stage2-then-success",
            "cancel-video-vae-then-success",
            "export-failure-then-success",
            "actual-layout-and-counters",
        ],
    }
    (args.output / "lifecycle-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    print("PASS LTX exact candidate session lifecycle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
