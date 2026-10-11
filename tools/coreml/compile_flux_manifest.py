"""Compile original Klein4B/9B partitions using the managed engine-cache API.

Creating the engine selects its actual model id; this never requests inference,
weakens a hybrid guard, or labels9B partitions as4B to bypass validation.
"""
import argparse
import ctypes as C
import json
from pathlib import Path


def validate_manifest(data, model):
    count, hidden, width = (20,3072,9216) if model == "flux2-klein-4b" else (24,4096,12288)
    artifacts, shape, source = data.get("artifacts"), data.get("shape", {}), data.get("source", {})
    if (data.get("schema_version") != 2 or not isinstance(artifacts, dict) or
            sorted(artifacts) != sorted(map(str, range(count))) or
            (shape.get("K"), shape.get("N"), shape.get("mlp_width")) != (hidden,hidden,width) or
            source.get("blocks") != list(range(count))):
        raise ValueError("complete actual selected Klein geometry/partitions required")
    if model == "flux2-klein-9b" and (not str(source.get("checkpoint", "")).endswith(".safetensors.index.json") or
            not isinstance(source.get("checkpoint_shards"), dict) or not source["checkpoint_shards"]):
        raise ValueError("original9B index and complete shard identity map required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("library", "source", "cache", "model-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--model", choices=("flux2-klein-4b","flux2-klein-9b"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    validate_manifest(json.loads(source.read_text()), args.model)
    if args.output.exists() or args.cache.is_symlink() or args.cache.resolve() == Path("/"):
        parser.error("fresh result and dedicated non-symlink cache required")
    lib = C.CDLL(str(args.library.resolve(strict=True)))
    pointer = C.c_void_p
    lib.tc_string_free.argtypes = [pointer]
    lib.tc_engine_free.argtypes = [pointer]
    lib.tc_engine_create_model.argtypes = [C.c_char_p,C.c_char_p,C.POINTER(pointer),C.POINTER(pointer)]
    lib.tc_engine_cache.argtypes = [pointer,C.c_char_p,pointer,pointer,C.POINTER(pointer),C.POINTER(pointer)]
    engine, error, result = pointer(), pointer(), pointer()
    if lib.tc_engine_create_model(args.model.encode(), str(args.model_root.resolve(strict=True)).encode(), C.byref(engine), C.byref(error)):
        message = C.string_at(error).decode() if error.value else "engine creation failed"
        if error.value:lib.tc_string_free(error)
        raise RuntimeError(message)
    try:
        request = dict(action="compile_manifest", source=str(source), cache=str(args.cache.absolute()))
        status = lib.tc_engine_cache(engine,json.dumps(request).encode(),None,None,C.byref(result),C.byref(error))
        if status:
            raise RuntimeError(C.string_at(error).decode() if error.value else "managed manifest compilation failed")
        value = json.loads(C.string_at(result).decode())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as out:
            json.dump(value,out,indent=2,allow_nan=False);out.write("\n")
        print(json.dumps(value))
    finally:
        if result.value:lib.tc_string_free(result)
        if error.value:lib.tc_string_free(error)
        lib.tc_engine_free(engine)


if __name__ == "__main__":main()
