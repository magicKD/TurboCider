"""ABBA check of sigma-independent context-refiner reuse on one loaded model."""
import argparse
import ctypes as C
import hashlib
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true",
                        help="run cache/control/control/cache after warmup")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    library = C.CDLL(str(args.library.resolve()))
    library.tc_string_free.argtypes = [C.c_void_p]
    library.tc_engine_create_model.argtypes = [C.c_char_p, C.c_char_p,
                                                C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    library.tc_engine_generate.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p,
                                           C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    library.tc_engine_free.argtypes = [C.c_void_p]

    def take(pointer):
        if not pointer.value:
            return None
        value = C.string_at(pointer).decode()
        library.tc_string_free(pointer)
        return value

    engine, error = C.c_void_p(), C.c_void_p()
    status = library.tc_engine_create_model(b"z-image-turbo", str(args.model.resolve()).encode(),
                                            C.byref(engine), C.byref(error))
    if status:
        raise RuntimeError(take(error))
    base = dict(model="z-image-turbo", prompt="两个人看海的照片", width=512, height=512,
                steps=9, seed=42, execution="gpu", audio=False, dynamic_text=True,
                residency="streamed", memory_budget_bytes=8 << 30)
    report = dict(library=str(args.library.resolve()), model=str(args.model.resolve()),
                  reverse=args.reverse, runs=[])
    callback = C.CFUNCTYPE(None, C.c_char_p, C.c_void_p)
    try:
        sequence = (True, True, False, False, True) if args.reverse else (
            False, False, True, True, False)
        for index, enabled in enumerate(sequence):
            if enabled:
                os.environ.pop("TURBOCIDER_Z_DISABLE_CACHE_CONTEXT", None)
                os.environ["TURBOCIDER_Z_CACHE_CONTEXT"] = "1"
            else:
                os.environ["TURBOCIDER_Z_DISABLE_CACHE_CONTEXT"] = "1"
                os.environ.pop("TURBOCIDER_Z_CACHE_CONTEXT", None)
            image = out / f"{index:02d}-{'cache' if enabled else 'control'}.png"
            request = dict(base, output=str(image))
            denoise_events = []

            @callback
            def event(raw, _):
                value = json.loads(raw)
                if value["phase"] == "denoise" and (
                    not denoise_events or value["completed"] > denoise_events[-1][0]
                ):
                    denoise_events.append((value["completed"], time.monotonic()))

            result, error = C.c_void_p(), C.c_void_p()
            start = time.monotonic()
            status = library.tc_engine_generate(engine, json.dumps(request).encode(), event, None,
                                                C.byref(result), C.byref(error))
            row = dict(index=index, cache_context=enabled, warmup=index == 0,
                       wall_seconds=time.monotonic() - start, status=status,
                       error=take(error), metrics=json.loads(take(result)) if result.value else None,
                       png_sha256=hashlib.sha256(image.read_bytes()).hexdigest() if image.exists() else None,
                       denoise_intervals_seconds=[denoise_events[i][1] - denoise_events[i - 1][1]
                                                  for i in range(1, len(denoise_events))])
            report["runs"].append(row)
            (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(index, "cache" if enabled else "control", f"{row['wall_seconds']:.3f}s",
                  row["error"] or "ok", flush=True)
            if status:
                raise RuntimeError(row["error"])
        if len({r["png_sha256"] for r in report["runs"]}) != 1:
            raise AssertionError("context cache changed PNG output")
    finally:
        os.environ.pop("TURBOCIDER_Z_DISABLE_CACHE_CONTEXT", None)
        os.environ.pop("TURBOCIDER_Z_CACHE_CONTEXT", None)
        library.tc_engine_free(engine)


if __name__ == "__main__":
    main()
