#!/usr/bin/env python3
"""Capture independent streaming audit counters from an audit-only build.

This tool deliberately runs outside the performance campaign.  Audit builds
contain relaxed atomic counters and are evidence for route isolation, not
latency baselines.  Production/release builds must not export the two private
audit symbols used here.
"""

from __future__ import annotations

import argparse
import ctypes as c
import hashlib
import json
from pathlib import Path
from typing import Any


COUNTERS = (
    "new_framework_hooks",
    "new_memory_probes",
    "new_worker_threads",
    "new_pool_allocations",
    "new_cache_clear_or_unload_calls",
    "steady_framework_allocations",
    "steady_framework_thread_creates",
)


class AuditError(RuntimeError):
    pass


def consume(library: Any, pointer: c.c_void_p) -> str | None:
    if not pointer.value:
        return None
    value = c.string_at(pointer).decode()
    library.tc_string_free(pointer)
    return value


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"{label} must be a JSON object")
    return value


def validate_snapshot(snapshot: dict[str, Any]) -> dict[str, int]:
    if snapshot.get("format") != "turbocider-streaming-audit-snapshot-v1":
        raise AuditError("audit snapshot has an unknown format")
    result: dict[str, int] = {}
    for name in COUNTERS:
        value = snapshot.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AuditError(f"audit counter {name} must be a non-negative integer")
        result[name] = value
    return result


def evaluate_snapshot(
    snapshot: dict[str, Any], expectation: str, request_succeeded: bool,
) -> dict[str, Any]:
    counters = validate_snapshot(snapshot)
    if expectation == "default-zero":
        passed = request_succeeded and all(value == 0 for value in counters.values())
        reason = (
            "default request did not enter the new streaming/memory framework"
            if passed else
            "default request failed or observed a new framework counter"
        )
    elif expectation == "framework-active":
        passed = any(value > 0 for value in counters.values())
        reason = (
            "opt-in request entered at least one audited framework callsite"
            if passed else
            "opt-in request did not enter an audited framework callsite"
        )
    elif expectation == "observe":
        passed = None
        reason = "counters captured without a pass/fail expectation"
    else:
        raise AuditError(f"unknown audit expectation: {expectation}")
    return {
        "format": "turbocider-streaming-audit-v1",
        "status": "passed" if passed is True else "failed" if passed is False else "partial",
        "passed": passed,
        "reason": reason,
        **counters,
    }


def capture(
    library_path: Path,
    model_id: str,
    model_path: Path,
    request: dict[str, Any],
    constructor_kind: str,
    expectation: str,
    allow_request_failure: bool,
    test_streaming_catalog: Path | None = None,
) -> dict[str, Any]:
    library_path = library_path.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    if not library_path.is_file():
        raise AuditError(f"audit library is missing: {library_path}")
    if not model_path.exists():
        raise AuditError(f"model path is missing: {model_path}")
    catalog_path = (
        test_streaming_catalog.expanduser().resolve()
        if test_streaming_catalog is not None else None
    )
    if catalog_path is not None:
        if constructor_kind != "public":
            raise AuditError(
                "test streaming catalog requires the public constructor"
            )
        if not catalog_path.is_file():
            raise AuditError(f"test streaming catalog is missing: {catalog_path}")
    library = c.CDLL(str(library_path))
    library.tc_string_free.argtypes = [c.c_void_p]
    try:
        reset = library.tc_streaming_audit_reset
        snapshot_call = library.tc_streaming_audit_snapshot_json
    except AttributeError as exc:
        raise AuditError(
            "library is not an audit build; rebuild with "
            "TURBOCIDER_BUILD_AUDIT_COUNTERS=1"
        ) from exc
    reset.argtypes = []
    reset.restype = None
    snapshot_call.argtypes = [c.POINTER(c.c_void_p), c.POINTER(c.c_void_p)]
    snapshot_call.restype = c.c_int
    constructor_name = (
        "tc_engine_create_model_candidate"
        if constructor_kind == "candidate" else "tc_engine_create_model"
    )
    try:
        constructor = getattr(library, constructor_name)
    except AttributeError as exc:
        raise AuditError(f"library does not export {constructor_name}") from exc
    constructor.argtypes = [
        c.c_char_p, c.c_char_p, c.POINTER(c.c_void_p), c.POINTER(c.c_void_p)
    ]
    constructor.restype = c.c_int
    library.tc_engine_generate.argtypes = [
        c.c_void_p, c.c_char_p, c.c_void_p, c.c_void_p,
        c.POINTER(c.c_void_p), c.POINTER(c.c_void_p),
    ]
    library.tc_engine_generate.restype = c.c_int
    library.tc_engine_free.argtypes = [c.c_void_p]
    install_catalog = None
    catalog_bytes = None
    if catalog_path is not None:
        try:
            install_catalog = library.tc_engine_test_set_streaming_catalog_json
        except AttributeError as exc:
            raise AuditError(
                "library does not export the test-only streaming catalog "
                "installer; rebuild with TURBOCIDER_BUILD_TEST_HOOKS=1"
            ) from exc
        install_catalog.argtypes = [
            c.c_void_p, c.c_char_p, c.POINTER(c.c_void_p)
        ]
        install_catalog.restype = c.c_int
        catalog_bytes = catalog_path.read_bytes()
    engine = c.c_void_p()
    error = c.c_void_p()
    status = constructor(
        model_id.encode(), str(model_path).encode(),
        c.byref(engine), c.byref(error),
    )
    failure = consume(library, error)
    if status or not engine.value:
        raise AuditError(failure or "native engine creation failed")
    result_pointer = c.c_void_p()
    request_error = c.c_void_p()
    snapshot_pointer = c.c_void_p()
    snapshot_error = c.c_void_p()
    try:
        if install_catalog is not None and catalog_bytes is not None:
            catalog_error = c.c_void_p()
            catalog_status = install_catalog(
                engine, catalog_bytes, c.byref(catalog_error)
            )
            catalog_failure = consume(library, catalog_error)
            if catalog_status:
                raise AuditError(
                    catalog_failure or "test streaming catalog install failed"
                )
        reset()
        request_status = library.tc_engine_generate(
            engine,
            json.dumps(request, sort_keys=True, separators=(",", ":")).encode(),
            None, None, c.byref(result_pointer), c.byref(request_error),
        )
        result_text = consume(library, result_pointer)
        request_failure = consume(library, request_error)
        snapshot_status = snapshot_call(
            c.byref(snapshot_pointer), c.byref(snapshot_error)
        )
        snapshot_text = consume(library, snapshot_pointer)
        snapshot_failure = consume(library, snapshot_error)
        if snapshot_status or not snapshot_text:
            raise AuditError(snapshot_failure or "audit snapshot failed")
        snapshot = json.loads(snapshot_text)
        if not isinstance(snapshot, dict):
            raise AuditError("audit snapshot is not a JSON object")
        audit = evaluate_snapshot(snapshot, expectation, request_status == 0)
        audit.update({
            "library": str(library_path),
            "constructor": constructor_kind,
            "model_id": model_id,
            "request_status": int(request_status),
            "request_succeeded": request_status == 0,
            "request_error": request_failure,
            "result_present": bool(result_text),
        })
        if catalog_path is not None and catalog_bytes is not None:
            audit.update({
                "test_streaming_catalog_path": str(catalog_path),
                "test_streaming_catalog_sha256": hashlib.sha256(
                    catalog_bytes
                ).hexdigest(),
                "test_streaming_catalog_size_bytes": len(catalog_bytes),
            })
        if request_status and not allow_request_failure:
            audit["status"] = "failed"
            audit["passed"] = False
            audit["reason"] = request_failure or "audited request failed"
        return audit
    finally:
        library.tc_engine_free(engine)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--constructor", choices=("public", "candidate"), default="public")
    parser.add_argument(
        "--expect",
        choices=("default-zero", "framework-active", "observe"),
        default="default-zero",
    )
    parser.add_argument("--allow-request-failure", action="store_true")
    parser.add_argument("--test-streaming-catalog", type=Path)
    arguments = parser.parse_args()
    try:
        request = read_object(arguments.request, "request")
        result = capture(
            arguments.library, arguments.model_id, arguments.model, request,
            arguments.constructor, arguments.expect,
            arguments.allow_request_failure,
            arguments.test_streaming_catalog,
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0 if result.get("passed") is not False else 1
    except (AuditError, OSError, json.JSONDecodeError) as exc:
        print(f"streaming audit failed: {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
