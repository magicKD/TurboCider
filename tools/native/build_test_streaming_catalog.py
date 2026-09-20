#!/usr/bin/env python3
"""Build a test-only public-semantics streaming catalog from a real adapter.

The native library must be built with TURBOCIDER_BUILD_TEST_HOOKS=1.  The
returned catalog deliberately contains template calibration/review fields and
is suitable only for public-semantics calibration campaigns, never production.
"""

from __future__ import annotations

import argparse
import ctypes as c
import json
import os
import tempfile
from pathlib import Path


GIB = 1 << 30
PUBLIC_TARGETS_GIB = (8, 10, 12, 16, 20)


class CatalogBuildError(RuntimeError):
    pass


def consume(library: c.CDLL, pointer: c.c_void_p) -> str:
    if not pointer.value:
        return ""
    value = c.string_at(pointer).decode("utf-8")
    library.tc_string_free(pointer)
    return value


def load_library(path: Path) -> c.CDLL:
    try:
        library = c.CDLL(str(path))
    except OSError as exc:
        raise CatalogBuildError(f"cannot load native library: {exc}") from exc
    required = (
        "tc_string_free",
        "tc_engine_create_model_worker",
        "tc_engine_free",
        "tc_engine_test_build_streaming_catalog_json",
    )
    for symbol in required:
        if not hasattr(library, symbol):
            raise CatalogBuildError(
                f"{path} does not export required test-only symbol {symbol}"
            )
    library.tc_string_free.argtypes = [c.c_void_p]
    library.tc_engine_create_model_worker.argtypes = [
        c.c_char_p, c.c_char_p, c.POINTER(c.c_void_p), c.POINTER(c.c_void_p)
    ]
    library.tc_engine_create_model_worker.restype = c.c_int
    library.tc_engine_free.argtypes = [c.c_void_p]
    library.tc_engine_test_build_streaming_catalog_json.argtypes = [
        c.c_void_p, c.c_char_p, c.c_char_p, c.c_uint64, c.c_char_p,
        c.POINTER(c.c_void_p), c.POINTER(c.c_void_p)
    ]
    library.tc_engine_test_build_streaming_catalog_json.restype = c.c_int
    return library


def read_json(path: Path, label: str) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise CatalogBuildError(f"cannot read {label} {path}: {exc}") from exc
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogBuildError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CatalogBuildError(f"{label} must contain a JSON object")
    return payload


def build_catalog(args: argparse.Namespace) -> dict:
    if args.target_gib not in PUBLIC_TARGETS_GIB:
        raise CatalogBuildError("target must be one of 8, 10, 12, 16, 20 GiB")
    if not args.model_id or not args.catalog_revision:
        raise CatalogBuildError("model id and catalog revision are required")
    library_path = args.library.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    if not library_path.is_file():
        raise CatalogBuildError(f"native library is missing: {library_path}")
    if not model_path.is_dir():
        raise CatalogBuildError(f"model path is missing: {model_path}")
    request = read_json(args.request.expanduser().resolve(), "request")
    plan = read_json(args.plan.expanduser().resolve(), "plan")
    library = load_library(library_path)
    engine = c.c_void_p()
    error = c.c_void_p()
    status = library.tc_engine_create_model_worker(
        args.model_id.encode(), str(model_path).encode(),
        c.byref(engine), c.byref(error)
    )
    failure = consume(library, error)
    if status or not engine.value:
        raise CatalogBuildError(failure or "native engine creation failed")
    try:
        output = c.c_void_p()
        error = c.c_void_p()
        status = library.tc_engine_test_build_streaming_catalog_json(
            engine, request, plan, args.target_gib * GIB,
            args.catalog_revision.encode(), c.byref(output), c.byref(error)
        )
        value = consume(library, output)
        failure = consume(library, error)
        if status or not value:
            raise CatalogBuildError(
                failure or "native test catalog construction failed"
            )
        try:
            catalog = json.loads(value)
        except json.JSONDecodeError as exc:
            raise CatalogBuildError(f"native catalog JSON is invalid: {exc}") from exc
        if not isinstance(catalog, dict):
            raise CatalogBuildError("native runtime returned a non-object catalog")
        return catalog
    finally:
        library.tc_engine_free(engine)


def write_catalog(path: Path, catalog: dict, force: bool) -> None:
    if path.exists() and not force:
        raise CatalogBuildError(f"output already exists: {path}; use --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(catalog, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--target-gib", required=True, type=int, choices=PUBLIC_TARGETS_GIB)
    parser.add_argument("--catalog-revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        catalog = build_catalog(args)
        output = args.output.expanduser().resolve()
        write_catalog(output, catalog, args.force)
    except CatalogBuildError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
