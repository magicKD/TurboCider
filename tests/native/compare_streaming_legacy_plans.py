#!/usr/bin/env python3
"""Compare legacy plan output in isolated processes; NOT a GPU benchmark."""
import argparse
import ctypes as C
import json
import subprocess
import sys
from pathlib import Path


def snapshot(path):
    lib = C.CDLL(str(path.resolve()))
    lib.tc_plan_json.argtypes = [C.c_char_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)]
    lib.tc_string_free.argtypes = [C.c_void_p]
    output = []
    models = ["ltx-2.5-distilled", "minimax-h3-turbo", "flux2-klein-4b", "flux2-klein-9b",
              "z-image-turbo", "z-image-turbo-gguf"]
    for model in models:
        for mode in ("default", "resident", "streamed"):
            request = {"model": model}
            if mode != "default":
                request["residency"] = mode
            out, err = C.c_void_p(), C.c_void_p()
            status = lib.tc_plan_json(json.dumps(request).encode(), C.byref(out), C.byref(err))
            value = json.loads(C.string_at(out).decode()) if out.value else None
            error = C.string_at(err).decode() if err.value else None
            output.append({"request": request, "status": status, "plan": value, "error": error})
            lib.tc_string_free(out)
            lib.tc_string_free(err)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    if args.snapshot:
        print(json.dumps(snapshot(args.snapshot), sort_keys=True))
        return
    if not args.baseline or not args.candidate:
        parser.error("--baseline and --candidate are required")
    values = []
    for path in (args.baseline, args.candidate):
        raw = subprocess.check_output([sys.executable, "-B", __file__, "--snapshot", str(path)], text=True)
        values.append(json.loads(raw))
    if values[0] != values[1]:
        for before, after in zip(values[0], values[1]):
            if before != after:
                print(json.dumps({"before": before, "after": after}, ensure_ascii=False, indent=2))
        raise SystemExit("FAIL: legacy plan changed")
    print(f"PASS: {len(values[0])} legacy plan/error results identical in isolated processes; not performance evidence")


if __name__ == "__main__":
    main()
