"""Measure the non-public LTX resident candidate through one native engine."""

import argparse
import ctypes as c
import json
import os
import time
from pathlib import Path


def consume(library, pointer):
    if not pointer.value:
        return None
    value = c.string_at(pointer).decode()
    library.tc_string_free(pointer)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be positive")

    cache = Path(args.cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["TURBOCIDER_LTX_CONDITIONING_CACHE_DIR"] = str(cache)
    library = c.CDLL(str(Path(args.library).resolve()))
    library.tc_string_free.argtypes = [c.c_void_p]
    library.tc_engine_create_model_candidate.argtypes = [
        c.c_char_p, c.c_char_p, c.POINTER(c.c_void_p), c.POINTER(c.c_void_p)
    ]
    library.tc_engine_generate.argtypes = [
        c.c_void_p, c.c_char_p, c.c_void_p, c.c_void_p,
        c.POINTER(c.c_void_p), c.POINTER(c.c_void_p)
    ]
    library.tc_engine_free.argtypes = [c.c_void_p]

    model = Path(args.model).resolve()
    output = Path(args.output).resolve()
    report = Path(args.report).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    engine = c.c_void_p()
    error = c.c_void_p()
    status = library.tc_engine_create_model_candidate(
        b"ltx-2.5-distilled", str(model).encode(), c.byref(engine),
        c.byref(error))
    failure = consume(library, error)
    if status:
        raise RuntimeError(failure)
    rows = []
    try:
        for index in range(args.runs):
            destination = output / f"ltx-resident-{index}.mp4"
            request = {
                "schema_version": 1,
                "model": "ltx-2.5-distilled",
                "operation": "video.generate",
                "prompt": "A cinematic red fox running through a snowy forest",
                "width": 704,
                "height": 448,
                "frames": 97,
                "fps": 24,
                "steps": 11,
                "seed": 42,
                "execution": "gpu",
                "residency": "resident",
                "audio": False,
                "output": str(destination),
            }
            result = c.c_void_p()
            error = c.c_void_p()
            started = time.perf_counter()
            status = library.tc_engine_generate(
                engine, json.dumps(request).encode(), None, None,
                c.byref(result), c.byref(error))
            wall = time.perf_counter() - started
            value = consume(library, result)
            failure = consume(library, error)
            if status:
                raise RuntimeError(failure)
            row = {
                "run": index,
                "client_wall_seconds": wall,
                "result": json.loads(value),
                "output_bytes": destination.stat().st_size,
            }
            rows.append(row)
            report.write_text(json.dumps({
                "format": "turbocider-ltx-resident-c-abi-benchmark-v1",
                "runs": rows,
            }, indent=2))
            print(json.dumps(row), flush=True)
    finally:
        library.tc_engine_free(engine)


if __name__ == "__main__":
    main()
