"""Compile a complete Z-Image source manifest into an isolated managed cache.

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
    args = parser.parse_args()
    manifest = json.loads(args.source.read_text())
    if manifest.get("schema_version") != 2 or sorted(map(int, manifest["artifacts"])) != list(range(32)):
        parser.error("expected a complete 32-block Z-Image source manifest")
    if args.cache.is_symlink() or args.cache.resolve() == Path("/"):
        parser.error("cache must be a dedicated, non-symlink directory")
    library = C.CDLL(str(args.library.resolve()))
    pointer = C.c_void_p
    library.tc_coreml_resources_json.argtypes = [C.c_char_p, pointer, pointer,
                                                   C.POINTER(pointer), C.POINTER(pointer)]
    library.tc_coreml_resources_json.restype = C.c_int
    library.tc_string_free.argtypes = [pointer]
    result, error = pointer(), pointer()
    request = {"action": "compile", "model": "z-image-turbo",
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
