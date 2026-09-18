#!/usr/bin/env python3
"""Bind one reviewed streaming record draft to deterministic P0-P3 policies.

The catalog builder requires every campaign to carry the exact source,
workload, runtime, device, plan and performance-profile identity of the record
it is intended to qualify.  This tool applies that binding to four frozen
operator-authored templates without inventing model parameters or changing the
selected layout.  It also freezes the P2 public target and validates every
result with the same campaign-policy validator used by the runner.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from build_streaming_catalog import (
    CatalogBuildError,
    PUBLIC_TARGETS,
    catalog_binding,
    read_object,
    validate_record_shape,
)
from run_streaming_campaign import CampaignError, validate_policy


SCHEMA = "turbocider-streaming-release-policy-set-v1"
GENERATOR_REVISION = "tc-streaming-release-policy-generator-v1"
KINDS = ("P0", "P1", "P2", "P3")


class PolicyPreparationError(ValueError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    value = read_object(path, "record input")
    record = copy.deepcopy(value.get("record", value))
    if not isinstance(record, dict):
        raise PolicyPreparationError("record input must contain an object")
    record.pop("canonical_record_digest", None)
    try:
        validate_record_shape(record)
    except CatalogBuildError as exc:
        raise PolicyPreparationError(f"record input is invalid: {exc}") from exc
    return record


def bind_policy(
    template: dict[str, Any], kind: str, record: dict[str, Any], target_bytes: int
) -> dict[str, Any]:
    if kind not in KINDS:
        raise PolicyPreparationError(f"unsupported gate {kind}")
    policy = copy.deepcopy(template)
    if policy.get("comparison_kind") != kind:
        raise PolicyPreparationError(
            f"{kind} template comparison_kind is {policy.get('comparison_kind')!r}"
        )
    if policy.get("status") != "frozen":
        raise PolicyPreparationError(f"{kind} template must be frozen")
    if "catalog_binding" in policy:
        raise PolicyPreparationError(
            f"{kind} template already contains catalog_binding"
        )
    policy["catalog_binding"] = catalog_binding(record)
    if kind == "P2":
        memory = policy.get("memory_sampling")
        if not isinstance(memory, dict):
            raise PolicyPreparationError("P2 template lacks memory_sampling")
        memory["target_bytes"] = target_bytes
        memory["headroom_policy_revision"] = "tc-public-headroom-v1"
        memory["required_variants"] = ["candidate"]
        memory["allow_swap_out"] = False
    try:
        validate_policy(policy)
    except CampaignError as exc:
        raise PolicyPreparationError(f"{kind} policy is invalid: {exc}") from exc
    return policy


def prepare_policy_set(
    record_path: Path,
    template_paths: dict[str, Path],
    target_bytes: int,
    output: Path,
) -> dict[str, Any]:
    if target_bytes not in PUBLIC_TARGETS:
        raise PolicyPreparationError(
            "target must be one of 8/10/12/16/20 GiB"
        )
    if set(template_paths) != set(KINDS):
        raise PolicyPreparationError("templates must contain P0, P1, P2 and P3")
    if output.exists():
        raise PolicyPreparationError(f"output must not already exist: {output}")
    record = _record(record_path.resolve())
    policies: dict[str, dict[str, Any]] = {}
    for kind in KINDS:
        template = read_object(template_paths[kind].resolve(), f"{kind} template")
        policies[kind] = bind_policy(template, kind, record, target_bytes)

    output.mkdir(parents=True)
    files: dict[str, dict[str, Any]] = {}
    for kind in KINDS:
        name = f"{kind.lower()}-campaign-policy.json"
        path = output / name
        path.write_text(
            json.dumps(policies[kind], indent=2, ensure_ascii=False) + "\n"
        )
        files[name] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "comparison_kind": kind,
        }
    binding = catalog_binding(record)
    manifest = {
        "schema": SCHEMA,
        "generator_revision": GENERATOR_REVISION,
        "status": "frozen",
        "target_bytes": target_bytes,
        "catalog_binding_sha256": sha256_bytes(canonical_json(binding)),
        "record_input_sha256": sha256_file(record_path.resolve()),
        "template_sha256": {
            kind: sha256_file(template_paths[kind].resolve()) for kind in KINDS
        },
        "files": files,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-input", required=True, type=Path)
    parser.add_argument("--p0-template", required=True, type=Path)
    parser.add_argument("--p1-template", required=True, type=Path)
    parser.add_argument("--p2-template", required=True, type=Path)
    parser.add_argument("--p3-template", required=True, type=Path)
    parser.add_argument(
        "--target-gib", required=True, type=int, choices=(8, 10, 12, 16, 20)
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = prepare_policy_set(
            args.record_input,
            {
                "P0": args.p0_template,
                "P1": args.p1_template,
                "P2": args.p2_template,
                "P3": args.p3_template,
            },
            args.target_gib << 30,
            args.output.resolve(),
        )
    except (OSError, json.JSONDecodeError, PolicyPreparationError) as exc:
        print(f"policy preparation rejected input: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
