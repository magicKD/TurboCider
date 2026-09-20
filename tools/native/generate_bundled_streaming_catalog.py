#!/usr/bin/env python3
"""Compile independently versioned catalog data after revalidating its evidence.

Inventory entries name the original builder inputs, not self-certified records.
All paths are relative to the inventory. Generated C++ is data assignments only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from build_streaming_catalog import (
    CatalogBuildError, build_record, validate_record_shape, canonical_record_digest,
)
from verify_streaming_campaign import EvidenceError

SCHEMA = "tc-bundled-streaming-catalog-input-v1"
INPUT_KEYS = {"bundle", "record_input", "review", "performance_bundle",
              "default_bundle", "swap_bundle", "expected_record_digest"}
U64_FIELDS = {"minimum_physical_memory_bytes", "maximum_physical_memory_bytes",
              "calibrated_request_bytes", "logical_read_bytes",
              "confirmation_sample_count", "maximum_sample_gap_ns"}


def cpp_string(value: str) -> str:
    if "\0" in value:
        raise ValueError("catalog strings must not contain NUL")
    return '"' + ''.join(chr(byte) if 32 <= byte < 127 and byte not in (34, 92)
                         else f"\\{byte:03o}" for byte in value.encode("utf-8")) + '"'


def assignments(path: str, value: dict) -> list[str]:
    result = []
    for key, item in value.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key):
            raise ValueError("invalid catalog member")
        member = path + "." + key
        if key == "stages":
            for name, stage in item.items():
                result.extend(assignments(member + "[" + cpp_string(name) + "]", stage))
        elif key == "token_shapes":
            for token in item:
                result.append(member + ".emplace_back();")
                result.extend(assignments(member + ".back()", token))
        elif isinstance(item, dict):
            result.extend(assignments(member, item))
        elif type(item) is bool:
            result.append(member + " = " + str(item).lower() + ";")
        elif type(item) is int:
            maximum = (1 << (64 if key in U64_FIELDS else 32)) - 1
            if not 0 <= item <= maximum:
                raise ValueError(f"{member} is outside native integer range")
            result.append(f"{member} = {item}ULL;")
        elif isinstance(item, str):
            result.append(member + " = " + cpp_string(item) + ";")
        else:
            raise ValueError(f"unsupported catalog field: {member}")
    return result


def render_catalog(revision: str, records: list[dict], runtime_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,256}", revision):
        raise ValueError("invalid bundled catalog revision")
    lines = ["#pragma once", "// Generated data; do not edit.",
             "static StreamingPresetCatalog make_bundled_streaming_catalog() {",
             "StreamingPresetCatalog catalog;", "catalog.revision = " + cpp_string(revision) + ";"]
    ids = set()
    for record in records:
        validate_record_shape(record)
        if record["catalog_revision"] != revision:
            raise ValueError("bundled catalog revision mismatch")
        if record["runtime"]["turbocider_build_id"] != runtime_id:
            raise ValueError("bundled record belongs to a different native build")
        if record["release"]["channel"] not in ("public-stable", "public-experimental"):
            raise ValueError("only public channels may enter the bundled catalog")
        if record.get("canonical_record_digest") != canonical_record_digest(record):
            raise ValueError("bundled record digest mismatch")
        if record["id"] in ids:
            raise ValueError("duplicate bundled preset id")
        ids.add(record["id"])
        lines.extend(["{", "catalog.records.emplace_back();", "auto &record = catalog.records.back();"])
        lines.extend(assignments("record", record))
        lines.extend(["validate_streaming_preset_record(record, catalog.revision);", "}"])
    return "\n".join(lines + ["return catalog;", "}", ""])


def generate(inventory: Path, runtime_manifest: Path) -> tuple[str, dict]:
    raw = inventory.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"schema", "revision", "records"} or value["schema"] != SCHEMA:
        raise ValueError("unsupported bundled catalog input schema")
    if not isinstance(value["records"], list) or len(value["records"]) > 64:
        raise ValueError("bundled catalog must contain 0...64 records")
    runtime_id = json.loads(runtime_manifest.read_text())["runtime_build_id"]
    records = []
    for entry in value["records"]:
        if not isinstance(entry, dict) or set(entry) != INPUT_KEYS:
            raise ValueError("bundled entries must name all original builder inputs and expected digest")
        def source(key: str) -> Path:
            if not isinstance(entry[key], str) or not entry[key]:
                raise ValueError("missing evidence input: " + key)
            return (inventory.parent / entry[key]).resolve()
        # Re-run the verifier and review checks. A JSON `status: verified` or
        # TEMPLATE calibration alone cannot authorize a production entry.
        verified = build_record(source("bundle"), source("record_input"), source("review"),
                                source("performance_bundle"), source("default_bundle"),
                                source("swap_bundle"))
        record = verified["record"]
        if record["canonical_record_digest"] != entry["expected_record_digest"]:
            raise ValueError("reviewed bundled record changed")
        records.append(record)
    header = render_catalog(value["revision"], records, runtime_id)
    return header, {"schema": "tc-bundled-streaming-catalog-manifest-v1",
                    "revision": value["revision"], "runtime_build_id": runtime_id,
                    "input_sha256": hashlib.sha256(raw).hexdigest(),
                    "header_sha256": hashlib.sha256(header.encode()).hexdigest(),
                    "record_digests": [r["canonical_record_digest"] for r in records]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    try:
        header, manifest = generate(args.inventory, args.runtime_manifest)
        header_path = args.output / "turbocider_bundled_catalog_generated.hpp"
        manifest_path = args.output / "bundled-catalog-manifest.json"
        if args.verify:
            if header_path.read_text() != header or json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("bundled catalog inputs changed during compilation")
        else:
            args.output.mkdir(parents=True, exist_ok=True)
            header_path.write_text(header)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    except (OSError, ValueError, CatalogBuildError, EvidenceError) as error:
        parser.exit(1, f"bundled catalog rejected: {error}\n")


if __name__ == "__main__":
    main()
