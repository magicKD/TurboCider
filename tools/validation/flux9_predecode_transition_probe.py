#!/usr/bin/env python3
"""Actual packed/predecoded/packed/predecoded session; not latency ranking."""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path

from PIL import Image
from flux9_bf16_framework_screen import requests, write_json
from runtime_ane_common import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("library","model","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh diagnostic output required")
    args.output.mkdir(parents=True)
    lib = C.CDLL(str(args.library.resolve(strict=True)))
    pointer = C.c_void_p
    lib.tc_engine_create_model.argtypes = [C.c_char_p,C.c_char_p,C.POINTER(pointer),C.POINTER(pointer)]
    lib.tc_engine_generate.argtypes = [pointer,C.c_char_p,pointer,pointer,C.POINTER(pointer),C.POINTER(pointer)]
    lib.tc_engine_free.argtypes = [pointer]
    lib.tc_string_free.argtypes = [pointer]

    def take(value):
        text = C.string_at(value).decode() if value.value else ""
        if value.value:
            lib.tc_string_free(value)
        return text

    counts = {}
    callback_type = C.CFUNCTYPE(None,C.c_char_p,pointer)

    def observed(payload,_):
        name = json.loads(payload.decode())["phase"]
        counts[name] = counts.get(name,0)+1

    callback = callback_type(observed)
    engine,error = pointer(),pointer()
    if lib.tc_engine_create_model(b"flux2-klein-9b",str(args.model.resolve(strict=True)).encode(),C.byref(engine),C.byref(error)):
        raise RuntimeError(take(error))
    key = "TURBOCIDER_FLUX9_PREDECODE_AFFINE"
    previous = os.environ.get(key)
    data = dict(schema="tc-flux9-predecode-transition-v1",status="running",qualification_passed=False,
        library_sha256=sha256_file(args.library),order=["packed","predecoded","packed","predecoded"],runs=[],
        scope="one actual same-source session, same prompt/seed, state restoration and RGB; untimed diagnostic, not full-framework latency or memory qualification")
    try:
        for index,decoded in enumerate((False,True,False,True)):
            if decoded:
                os.environ[key] = "1"
            else:
                os.environ.pop(key,None)
            task = requests(args.output.absolute(),decoded)[2]
            task.update(allow_approximation=True,output=str(args.output.absolute() / f"{index}.png"))
            write_json(args.output / f"request-{index}.json",task)
            counts.clear()
            result,error = pointer(),pointer()
            status = lib.tc_engine_generate(engine,json.dumps(task).encode(),C.cast(callback,pointer),None,C.byref(result),C.byref(error))
            output,failure = take(result),take(error)
            if status:
                raise RuntimeError(failure)
            if (counts.get("compiled_single_gpu_block",0) != (96 if decoded else 0) or
                    counts.get("affine_predecode_projection",0) != (153 if decoded else 0)):
                raise ValueError("actual decode and restored packed branch differ")
            with Image.open(task["output"]) as image:
                image.load()
                if image.size != (512,512):
                    raise ValueError("complete unresized output required")
                pixels = hashlib.sha256(image.tobytes()).hexdigest()
            data["runs"].append(dict(index=index,decoded=decoded,events=counts.copy(),metrics=json.loads(output),
                png_sha256=sha256_file(Path(task["output"])),pixels_sha256=pixels))
        if data["runs"][0]["pixels_sha256"] != data["runs"][2]["pixels_sha256"] or data["runs"][1]["pixels_sha256"] != data["runs"][3]["pixels_sha256"]:
            raise ValueError("packed and decoded source restoration must each preserve own pixels")
        data["status"] = "complete_diagnostic"
    except BaseException as failure:
        data.update(status="failed",error=str(failure))
        raise
    finally:
        if previous is None:
            os.environ.pop(key,None)
        else:
            os.environ[key] = previous
        lib.tc_engine_free(engine)
        write_json(args.output / "summary.json",data)
    print(json.dumps(dict(status=data["status"],order=data["order"])))


if __name__ == "__main__":
    main()
