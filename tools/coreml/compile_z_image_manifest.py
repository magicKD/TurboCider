"""Compile a Z-Image or Qwen21 source manifest into an isolated managed cache.

The native resource API checks source paths, hashes, cache ownership and
publishes an immutable compiled manifest; this wrapper only exposes its
return value to the offline experimental workflow.
"""

import argparse
import ctypes as C
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", choices=["z-image-turbo", "qwen-image-2.1"],
                        default="z-image-turbo")
    args = parser.parse_args()
    manifest = json.loads(args.source.read_text())
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or any(not isinstance(key, str) or
                                              not key.isdecimal() for key in artifacts):
        parser.error("source manifest needs decimal-indexed artifacts")
    count = len(artifacts)
    if (manifest.get("schema_version") != 2 or not 1 <= count <= 32 or
            sorted(map(int, artifacts)) != list(range(count)) or
            (args.model == "z-image-turbo" and count != 32) or
            (args.model == "qwen-image-2.1" and
             manifest.get("export_identity", {}).get("tensor_layout") != "qwen21")):
        parser.error("expected contiguous partitions for the selected model")
    if args.cache.is_symlink() or args.cache.resolve() == Path("/"):
        parser.error("cache must be a dedicated, non-symlink directory")
    library = C.CDLL(str(args.library.resolve()))
    pointer = C.c_void_p
    library.tc_coreml_resources_json.argtypes = [C.c_char_p, pointer, pointer,
                                                   C.POINTER(pointer), C.POINTER(pointer)]
    library.tc_coreml_resources_json.restype = C.c_int
    library.tc_string_free.argtypes = [pointer]
    result, error = pointer(), pointer()
    request = {"action": "compile", "model": args.model,
               "source_manifest": str(args.source.resolve()),
               "cache": str(args.cache.absolute())}
    status = library.tc_coreml_resources_json(
        json.dumps(request).encode(), None, None, C.byref(result), C.byref(error))
    try:
        if status:
            raise RuntimeError(C.string_at(error).decode() if error.value else "Core ML compile failed")
        print(C.string_at(result).decode())
    finally:
        if result.value:
            library.tc_string_free(result)
        if error.value:
            library.tc_string_free(error)


if __name__ == "__main__":
    main()
