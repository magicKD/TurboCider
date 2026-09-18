#!/usr/bin/env python3
"""Run a frozen TurboCider streaming performance campaign.

The coordinator never loads a native library.  Baseline and candidate run in
two independent, persistent worker processes.  A frozen campaign explicitly
chooses whether each worker retains one engine or creates one engine per
request.  Measured requests are serialized in alternating ABBA/BAAB blocks and
appended to raw-samples.jsonl before the next request is dispatched.

This tool intentionally does not create memory pressure, clear OS caches, or
change swap settings.  Those operations require a separate, explicitly
authorized P3 protocol.
"""

from __future__ import annotations

import argparse
import ctypes as c
import hashlib
import json
import os
import platform
import resource
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from verify_streaming_campaign import EvidenceError, verify
from capture_streaming_source_identity import IdentityError, capture
from process_tree_sampler import (
    DarwinBackend,
    HashChainJsonlWriter,
    ProcessTreeSampler,
    RealClock,
    RoleRule,
)
from verify_process_tree_samples import EvidenceError as MemoryEvidenceError
from verify_process_tree_samples import verify as verify_memory_evidence


SCHEMA_VERSION = 1
VARIANTS = ("baseline", "candidate")
ENGINE_LIFECYCLES = ("persistent", "per_request")
SEMANTIC_LAYOUT_FIELDS = (
    "stage",
    "resident_prefix_blocks",
    "block_group_size",
    "slot_count",
    "prefetch_distance",
    "io_workers",
    "group_count",
    "pass_count",
    "startup_policy",
    "pass_transition",
    "retention",
    "reader_revision",
    "weight_format",
    "kernel_revision",
    "conditioning_recipe",
    "upsample_boundary",
)
P1_EXPECTED_FIELDS = (
    *SEMANTIC_LAYOUT_FIELDS,
    "engine_lifecycle",
    "total_fills",
)
SEQUENCES = {
    "ABBA": ("baseline", "candidate", "candidate", "baseline"),
    "BAAB": ("candidate", "baseline", "baseline", "candidate"),
}


class CampaignError(RuntimeError):
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


def read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CampaignError(f"{label} must be a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n"
    with path.open("a") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def deep_merge(base: Any, patch: Any) -> Any:
    if not isinstance(base, dict) or not isinstance(patch, dict):
        return deepcopy(patch)
    result = deepcopy(base)
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        elif key in result:
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def set_dotted(value: dict[str, Any], dotted: str, replacement: Any) -> None:
    parts = dotted.split(".")
    current: Any = value
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise CampaignError(f"seed path does not exist: {dotted}")
        current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current:
        raise CampaignError(f"seed path does not exist: {dotted}")
    current[parts[-1]] = replacement


def expand(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        for key, replacement in replacements.items():
            value = value.replace("${" + key + "}", replacement)
        return value
    if isinstance(value, list):
        return [expand(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: expand(item, replacements) for key, item in value.items()}
    return value


def validate_policy(policy: dict[str, Any]) -> None:
    if policy.get("schema_version") != SCHEMA_VERSION:
        raise CampaignError("campaign policy schema_version must be 1")
    if policy.get("status") != "frozen":
        raise CampaignError("campaign policy must be frozen before execution")
    if policy.get("comparison_kind") not in ("P0", "P1", "P2", "P3", "P4"):
        raise CampaignError("comparison_kind must be P0, P1, P2, P3 or P4")
    lifecycle = policy.get("engine_lifecycle", "persistent")
    if lifecycle not in ENGINE_LIFECYCLES:
        raise CampaignError(
            "engine_lifecycle must be persistent or per_request"
        )
    if policy.get("comparison_kind") == "P1":
        declaration = policy.get("semantic_equivalence")
        if (
            not isinstance(declaration, dict) or
            declaration.get("declared_equivalent") is not True
        ):
            raise CampaignError(
                "P1 requires a declared semantic equivalence"
            )
        expected = declaration.get("expected_actual")
        if not isinstance(expected, dict):
            raise CampaignError(
                "P1 semantic_equivalence.expected_actual must be an object"
            )
        missing = [
            field for field in P1_EXPECTED_FIELDS
            if field not in expected
        ]
        if missing:
            raise CampaignError(
                "P1 expected_actual is incomplete: " + ", ".join(missing)
            )
    variants = policy.get("variants")
    if not isinstance(variants, dict) or set(variants) != set(VARIANTS):
        raise CampaignError("variants must contain exactly baseline and candidate")
    expected_implementations = policy.get("expected_implementations")
    if expected_implementations is not None and (
        not isinstance(expected_implementations, dict) or
        set(expected_implementations) != set(VARIANTS) or
        any(
            not isinstance(expected_implementations[name], str) or
            not expected_implementations[name]
            for name in VARIANTS
        )
    ):
        raise CampaignError(
            "expected_implementations must contain non-empty baseline and "
            "candidate strings"
        )
    workload = policy.get("workload")
    if not isinstance(workload, dict) or not isinstance(workload.get("request"), dict):
        raise CampaignError("workload.request must be a JSON object")
    protocol = policy.get("protocol")
    if not isinstance(protocol, dict):
        raise CampaignError("protocol must be a JSON object")
    blocks = protocol.get("measured_blocks")
    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks <= 0:
        raise CampaignError("protocol.measured_blocks must be positive")
    warmups = protocol.get("warmup_requests_per_variant", 1)
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 0:
        raise CampaignError("warmup_requests_per_variant must be non-negative")
    timeout = protocol.get("request_timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise CampaignError("request_timeout_seconds must be positive")
    start_timeout = protocol.get("worker_start_timeout_seconds", 30)
    if (
        isinstance(start_timeout, bool) or
        not isinstance(start_timeout, (int, float)) or start_timeout <= 0
    ):
        raise CampaignError("worker_start_timeout_seconds must be positive")
    launch_order = protocol.get("worker_launch_order", list(VARIANTS))
    if (
        not isinstance(launch_order, list) or
        len(launch_order) != len(VARIANTS) or
        set(launch_order) != set(VARIANTS)
    ):
        raise CampaignError(
            "worker_launch_order must list baseline and candidate exactly once"
        )
    if protocol.get("launch_pressure"):
        raise CampaignError("this runner never launches memory pressure")
    memory_sampling = policy.get("memory_sampling")
    if policy.get("comparison_kind") in ("P2", "P3") and not isinstance(
        memory_sampling, dict
    ):
        raise CampaignError("P2/P3 campaigns require memory_sampling")
    if memory_sampling is not None:
        if not isinstance(memory_sampling, dict):
            raise CampaignError("memory_sampling must be an object")
        if memory_sampling.get("enabled") is not True:
            raise CampaignError(
                "memory_sampling, when present, must set enabled=true"
            )
        interval = memory_sampling.get("interval_ms", 20)
        max_gap = memory_sampling.get("max_gap_ms", 100)
        if (
            isinstance(interval, bool) or not isinstance(interval, int) or
            interval <= 0
        ):
            raise CampaignError("memory_sampling.interval_ms must be positive")
        if (
            isinstance(max_gap, bool) or not isinstance(max_gap, int) or
            max_gap < interval
        ):
            raise CampaignError(
                "memory_sampling.max_gap_ms must be >= interval_ms"
            )
        roles = memory_sampling.get("roles", [])
        if not isinstance(roles, list) or any(
            not isinstance(value, str) or "=" not in value or
            not value.split("=", 1)[0] or not value.split("=", 1)[1]
            for value in roles
        ):
            raise CampaignError(
                "memory_sampling.roles must contain ROLE=GLOB strings"
            )
        root_role = memory_sampling.get("root_role", "streaming-worker")
        if not isinstance(root_role, str) or not root_role:
            raise CampaignError("memory_sampling.root_role must be non-empty")
        include_warmups = memory_sampling.get("include_warmups", False)
        if not isinstance(include_warmups, bool):
            raise CampaignError("memory_sampling.include_warmups must be boolean")
        required_variants = memory_sampling.get(
            "required_variants", ["candidate"]
        )
        if (
            not isinstance(required_variants, list) or
            not required_variants or
            len(set(required_variants)) != len(required_variants) or
            any(value not in VARIANTS for value in required_variants)
        ):
            raise CampaignError(
                "memory_sampling.required_variants must be a unique, non-empty "
                "subset of baseline/candidate"
            )
        if policy.get("comparison_kind") == "P2":
            target = memory_sampling.get("target_bytes")
            public_targets = {value << 30 for value in (8, 10, 12, 16, 20)}
            if (
                isinstance(target, bool) or not isinstance(target, int) or
                target not in public_targets
            ):
                raise CampaignError(
                    "P2 memory_sampling.target_bytes must be a public 8/10/12/16/20 GiB tier"
                )
            if memory_sampling.get("headroom_policy_revision") != (
                "tc-public-headroom-v1"
            ):
                raise CampaignError(
                    "P2 requires headroom_policy_revision=tc-public-headroom-v1"
                )
            if "candidate" not in required_variants:
                raise CampaignError("P2 must require candidate memory evidence")
    quality = policy.get("quality")
    if not isinstance(quality, dict) or quality.get("mode") != "artifact_sha256_equal":
        raise CampaignError("quality.mode must be artifact_sha256_equal")
    artifacts = quality.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise CampaignError("quality.artifacts must be a non-empty list")
    names: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise CampaignError("each quality artifact must be an object")
        name = artifact.get("name")
        path = artifact.get("path")
        if not isinstance(name, str) or not name or name in names:
            raise CampaignError("quality artifact names must be unique strings")
        if not isinstance(path, str) or not path:
            raise CampaignError(f"quality artifact {name} needs a path template")
        names.add(name)
    for variant in VARIANTS:
        config = variants[variant]
        if not isinstance(config, dict):
            raise CampaignError(f"variant {variant} must be an object")
        backend = config.get("backend", "native")
        if backend not in ("native", "probe", "synthetic"):
            raise CampaignError(f"variant {variant} has unsupported backend")
        if backend == "native":
            for key in ("library", "model_id", "model_path"):
                if not isinstance(config.get(key), str) or not config[key]:
                    raise CampaignError(f"native variant {variant} requires {key}")
            if config.get("constructor", "public") not in ("public", "candidate"):
                raise CampaignError(f"variant {variant} has invalid constructor")
            environment = config.get("environment", {})
            if not isinstance(environment, dict):
                raise CampaignError(f"variant {variant} environment must be an object")
            for name, value in environment.items():
                if (
                    not isinstance(name, str) or
                    not name.startswith("TURBOCIDER_") or
                    not isinstance(value, str)
                ):
                    raise CampaignError(
                        f"variant {variant} environment only accepts "
                        "string TURBOCIDER_* variables"
                    )
        elif backend == "probe":
            if lifecycle != "per_request":
                raise CampaignError(
                    "probe variants require engine_lifecycle=per_request"
                )
            for key in ("executable", "model_path", "output_artifact"):
                if not isinstance(config.get(key), str) or not config[key]:
                    raise CampaignError(
                        f"probe variant {variant} requires {key}"
                    )
            arguments = config.get("arguments")
            if (
                not isinstance(arguments, list) or not arguments or
                any(not isinstance(value, str) for value in arguments)
            ):
                raise CampaignError(
                    f"probe variant {variant} requires string arguments"
                )
            environment = config.get("environment", {})
            if not isinstance(environment, dict):
                raise CampaignError(
                    f"probe variant {variant} environment must be an object"
                )
            for name, value in environment.items():
                if (
                    not isinstance(name, str) or
                    not name.startswith("TURBOCIDER_") or
                    not isinstance(value, str)
                ):
                    raise CampaignError(
                        f"probe variant {variant} environment only accepts "
                        "string TURBOCIDER_* variables"
                    )
        elif not isinstance(config.get("synthetic"), dict):
            raise CampaignError(f"synthetic variant {variant} needs synthetic config")


class RequestMemorySampler:
    """Attach a process-tree sampler to one persistent campaign worker.

    The sampler is deliberately opt-in.  It runs outside the worker and does
    not touch the model/runtime, so disabled campaigns retain the old hot path.
    """

    def __init__(
        self,
        worker_pid: int,
        evidence_path: Path,
        correlation_id: str,
        config: dict[str, Any],
    ) -> None:
        self.worker_pid = worker_pid
        self.evidence_path = evidence_path
        self.correlation_id = correlation_id
        self.config = config
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.writer: HashChainJsonlWriter | None = None
        self.sampler: ProcessTreeSampler | None = None
        self.result: dict[str, Any] | None = None
        self.error: BaseException | None = None

    def _run(self) -> None:
        try:
            backend = DarwinBackend()
            self.writer = HashChainJsonlWriter(self.evidence_path)
            role_rules = [
                RoleRule(value.split("=", 1)[0], value.split("=", 1)[1])
                for value in self.config.get("roles", [])
            ]
            self.sampler = ProcessTreeSampler(
                backend, RealClock(), self.writer, self.worker_pid,
                root_role=self.config.get("root_role", "streaming-worker"),
                role_rules=role_rules,
                interval_ns=int(self.config.get("interval_ms", 20)) * 1_000_000,
                max_gap_ns=int(self.config.get("max_gap_ms", 100)) * 1_000_000,
                correlation_id=self.correlation_id,
            )
            self.result = self.sampler.run(
                stop_event=self.stop_event,
                ready_event=self.ready_event,
            )
        except BaseException as exc:  # preserve a diagnosable sampler failure
            self.error = exc
            self.ready_event.set()
        finally:
            if self.writer is not None:
                self.writer.close()

    def start(self) -> None:
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(
            target=self._run,
            name=f"tc-memory-sampler-{self.correlation_id}",
            daemon=True,
        )
        self.thread.start()
        wait_seconds = max(5.0, float(self.config.get("max_gap_ms", 100)) / 1000.0 * 10)
        if not self.ready_event.wait(wait_seconds):
            self.stop_event.set()
            self.thread.join(wait_seconds)
            raise CampaignError("process-tree sampler did not become ready")
        if self.error is not None:
            self.thread.join(wait_seconds)
            raise CampaignError(f"process-tree sampler failed: {self.error}")

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(
                max(5.0, float(self.config.get("max_gap_ms", 100)) / 1000.0 * 10)
            )
            if self.thread.is_alive():
                raise CampaignError("process-tree sampler did not stop")
        if self.error is not None:
            raise CampaignError(f"process-tree sampler failed: {self.error}")
        if self.result is None:
            raise CampaignError("process-tree sampler produced no terminal record")
        try:
            verified = verify_memory_evidence(self.evidence_path)
        except MemoryEvidenceError as exc:
            raise CampaignError(f"invalid process-tree evidence: {exc}") from exc
        if verified["correlation_id"] != self.correlation_id:
            raise CampaignError("process-tree evidence correlation differs")
        return {
            "schema": "turbocider-streaming-memory-summary-v1",
            "sampler_revision": verified["sampler_revision"],
            "correlation_id": self.correlation_id,
            "status": verified["status"],
            "complete": verified["complete"],
            "reasons": verified["reasons"],
            "sample_count": verified["sample_count"],
            "max_gap_ns": verified["max_gap_ns"],
            "allowed_max_gap_ns": verified["allowed_max_gap_ns"],
            "tree_peak_rss_bytes": verified["tree_peak_rss_bytes"],
            "tree_peak_phys_footprint_bytes": verified[
                "tree_peak_phys_footprint_bytes"
            ],
            "swap_in_bytes": verified["swap_in_bytes"],
            "swap_out_bytes": verified["swap_out_bytes"],
            "compression_bytes": verified["compression_bytes"],
            "decompression_bytes": verified["decompression_bytes"],
            "command_exit_code": verified["command_exit_code"],
            "evidence_digest": verified["final_evidence_digest"],
            "evidence_path": str(self.evidence_path),
        }


def run_worker_request(
    worker: "Worker",
    command: dict[str, Any],
    output: Path,
    policy: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    memory_config = policy.get("memory_sampling")
    sampler: RequestMemorySampler | None = None
    should_sample = (
        isinstance(memory_config, dict) and
        memory_config.get("enabled") is True and
        (phase == "measured" or memory_config.get("include_warmups") is True)
    )
    evidence_path: Path | None = None
    if should_sample:
        evidence_path = output / "memory" / f"{command['run_id']}.jsonl"
        sampler = RequestMemorySampler(
            worker.pid, evidence_path, command["run_id"], memory_config
        )
        sampler.start()
    response: dict[str, Any] | None = None
    request_error: BaseException | None = None
    try:
        response = worker.run(command)
    except BaseException as exc:
        request_error = exc
    finally:
        if sampler is not None:
            summary = sampler.stop()
            assert evidence_path is not None
            summary["phase"] = phase
            summary["variant"] = command["variant"]
            summary["run_id"] = command["run_id"]
            summary["evidence_path"] = str(evidence_path.relative_to(output))
            summary_path = output / "memory" / f"{command['run_id']}.summary.json"
            summary["summary_path"] = str(summary_path.relative_to(output))
            with summary_path.open("x", encoding="utf-8") as stream:
                json.dump(summary, stream, indent=2)
                stream.write("\n")
            if response is not None:
                response["memory_summary"] = summary
    if request_error is not None:
        raise request_error
    if response is None:
        raise CampaignError("worker returned no response")
    return response


def build_identity(policy: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in VARIANTS:
        config = policy["variants"][variant]
        backend = config.get("backend", "native")
        if backend == "native":
            library = Path(config["library"]).expanduser().resolve()
            if not library.is_file():
                raise CampaignError(f"missing {variant} library: {library}")
            identity = {
                "backend": "native",
                "binary_path": str(library),
                "binary_sha256": sha256_file(library),
                "binary_size_bytes": library.stat().st_size,
                "constructor": config.get("constructor", "public"),
            }
        elif backend == "probe":
            executable = Path(config["executable"]).expanduser().resolve()
            if not executable.is_file():
                raise CampaignError(
                    f"missing {variant} probe executable: {executable}"
                )
            identity = {
                "backend": "probe",
                "binary_path": str(executable),
                "binary_sha256": sha256_file(executable),
                "binary_size_bytes": executable.stat().st_size,
                "constructor": "probe_process",
            }
            linked_library = config.get("linked_library")
            if isinstance(linked_library, str) and linked_library:
                library = Path(linked_library).expanduser().resolve()
                if not library.is_file():
                    raise CampaignError(
                        f"missing {variant} probe library: {library}"
                    )
                identity["linked_library_path"] = str(library)
                identity["linked_library_sha256"] = sha256_file(library)
                identity["linked_library_size_bytes"] = library.stat().st_size
        else:
            identity = {
                "backend": "synthetic",
                "binary_path": None,
                "binary_sha256": sha256_bytes(canonical_json(config["synthetic"])),
                "binary_size_bytes": 0,
                "constructor": "synthetic",
            }
        if isinstance(config.get("source_root"), str):
            try:
                identity["source_identity"] = capture(
                    Path(config["source_root"]), config.get("source_commit")
                )
            except IdentityError as exc:
                raise CampaignError(
                    f"cannot capture {variant} source identity: {exc}"
                ) from exc
        elif isinstance(config.get("source_identity"), dict):
            identity["source_identity"] = deepcopy(config["source_identity"])
        identity["variant_config_sha256"] = sha256_bytes(canonical_json(config))
        result[variant] = identity
    return result


def reject_output_inside_sources(policy: dict[str, Any], output: Path) -> None:
    output = output.resolve()
    for variant in VARIANTS:
        raw_root = policy["variants"][variant].get("source_root")
        if not isinstance(raw_root, str):
            continue
        source_root = Path(raw_root).expanduser().resolve()
        try:
            common = Path(os.path.commonpath((str(output), str(source_root))))
        except ValueError:
            continue
        if common == source_root:
            raise CampaignError(
                f"output cannot be inside {variant} source_root: {output}"
            )


def send_message(connection: socket.socket, value: dict[str, Any]) -> None:
    payload = canonical_json(value)
    connection.sendall(struct.pack("!Q", len(payload)) + payload)


def receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("worker connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_message(connection: socket.socket, timeout: float) -> dict[str, Any]:
    connection.settimeout(timeout)
    header = receive_exact(connection, 8)
    length = struct.unpack("!Q", header)[0]
    if length > 128 * (1 << 20):
        raise CampaignError("worker response exceeds protocol limit")
    value = json.loads(receive_exact(connection, length))
    if not isinstance(value, dict):
        raise CampaignError("worker response is not an object")
    return value


def consume(library: Any, pointer: c.c_void_p) -> str | None:
    if not pointer.value:
        return None
    value = c.string_at(pointer).decode()
    library.tc_string_free(pointer)
    return value


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def nested(value: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def extract_layouts(result: dict[str, Any]) -> tuple[str | None, str | None]:
    resolved = nested(result, "plan", "streaming", "resolved_layout", "digest")
    actual = nested(result, "plan", "streaming", "actual_layout", "digest")
    block = result.get("block_streaming") or result.get("block_residency") or {}
    block_digest = block.get("layout_digest") if isinstance(block, dict) else None
    if not resolved:
        resolved = block_digest
    if not actual:
        actual = block_digest
    return resolved or None, actual or None


def extract_streaming_implementation(result: dict[str, Any]) -> str | None:
    block = result.get("block_streaming") or result.get("block_residency")
    implementation = (
        block.get("implementation") if isinstance(block, dict) else None
    )
    return implementation if isinstance(implementation, str) else None


def actual_semantic_layout(
    result: dict[str, Any], engine_lifecycle: str
) -> tuple[dict[str, Any] | None, list[str]]:
    """Normalize observed layout/runtime semantics for a P1 comparison.

    A generic plan digest is an implementation identity.  The legacy LTX
    implementation intentionally has no generic digest, so P1 compares this
    normalized semantic identity instead.
    """
    block = result.get("block_streaming") or result.get("block_residency")
    raw = nested(result, "plan", "streaming", "actual_layout")
    if not isinstance(raw, dict) and isinstance(block, dict):
        raw = block.get("actual_layout")
    if not isinstance(raw, dict):
        return None, [*SEMANTIC_LAYOUT_FIELDS, "engine_lifecycle", "total_fills"]

    semantic = {
        field: deepcopy(raw[field])
        for field in SEMANTIC_LAYOUT_FIELDS
        if field in raw and raw[field] is not None
    }
    semantic["engine_lifecycle"] = engine_lifecycle
    # The native result calls this retention.  P1's request lifecycle must be
    # explicit so a retained engine cannot be compared with a request owner.
    if engine_lifecycle == "per_request":
        # Legacy adapters may report their engine-owned cache retention even
        # when the campaign recreates the engine per request; normalize that
        # case to the request lifecycle.  Preserve an explicit request-scoped
        # multi-pool qualifier (for example Flux retain_all), since it is part
        # of the layout identity and must be compared by P1.
        retention = semantic.get("retention")
        if not (
            isinstance(retention, str) and
            (retention == "request" or retention.startswith("request;"))
        ):
            semantic["retention"] = "request"
    if isinstance(block, dict):
        fills = block.get("request_slot_fills")
        allocations = block.get("request_slot_allocations")
        refills = block.get("request_slot_refills")
        if isinstance(fills, int) and not isinstance(fills, bool) and fills >= 0:
            semantic["total_fills"] = fills
        elif (
            isinstance(allocations, int) and not isinstance(allocations, bool)
            and allocations >= 0
            and isinstance(refills, int) and not isinstance(refills, bool)
            and refills >= 0
        ):
            semantic["total_fills"] = allocations + refills
    for key in (
        "conditioning_mode",
        "conditioning_cache_hit",
        "denoiser_cache_hit",
        "video_vae_isolation",
    ):
        if key in result and result[key] is not None:
            semantic[key] = deepcopy(result[key])
    required = [*SEMANTIC_LAYOUT_FIELDS, "engine_lifecycle", "total_fills"]
    return semantic, [field for field in required if field not in semantic]


def semantic_digest(value: dict[str, Any] | None) -> str | None:
    return sha256_bytes(canonical_json(value)) if value is not None else None


def request_semantic_identity(request: dict[str, Any]) -> dict[str, Any]:
    """Normalize schema-v1/v2 workload fields while excluding output paths.

    Executor-specific residency/layout fields are deliberately excluded: P1
    compares those from the observed semantic layout.  Everything that can
    change model work or numerical policy remains part of this identity.
    """
    schema = request.get("schema_version", 1)
    if schema == 2:
        inputs = request.get("inputs") or []
        prompts = [
            item.get("text")
            for item in inputs
            if isinstance(item, dict) and item.get("kind") == "text"
            and item.get("role") == "prompt"
        ]
        nontext_inputs = [
            {
                key: item.get(key)
                for key in ("kind", "role", "strength")
                if item.get(key) is not None
            }
            for item in inputs
            if isinstance(item, dict) and item.get("kind") != "text"
        ]
        outputs = request.get("outputs") or []
        output = next(
            (
                item for item in outputs
                if isinstance(item, dict) and
                item.get("kind") in ("video", "image")
            ),
            {},
        )
        sampling = request.get("sampling") or {}
        execution = request.get("execution") or {}
        return {
            "model": request.get("model"),
            "operation": request.get("operation"),
            "prompts": prompts,
            "nontext_inputs": nontext_inputs,
            "width": output.get("width"),
            "height": output.get("height"),
            "frames": output.get("frames", 1),
            "fps": output.get("fps"),
            "audio": output.get("audio", False),
            "seed": sampling.get("seed"),
            "steps": sampling.get("steps"),
            "execution": execution.get("policy"),
            "ltx_backend": execution.get("ltx_backend"),
            "ltx_fast_av": execution.get("ltx_fast_av"),
            "ltx_video_attention_batch": execution.get(
                "ltx_video_attention_batch", False
            ),
            "allow_approximation": request.get(
                "allow_approximation", False
            ),
            "ltx_sol_stage2": request.get("ltx_sol_stage2", False),
            "ltx_sol_tau": request.get("ltx_sol_tau"),
            "ltx_sol_dense_edge_blocks": request.get(
                "ltx_sol_dense_edge_blocks"
            ),
            "ltx_sol_dense_edge_steps": request.get(
                "ltx_sol_dense_edge_steps"
            ),
        }
    return {
        "model": request.get("model"),
        "operation": request.get("operation"),
        "prompts": [request.get("prompt")],
        "nontext_inputs": [],
        "width": request.get("width"),
        "height": request.get("height"),
        "frames": request.get("frames", 1),
        "fps": request.get("fps"),
        "audio": request.get("audio", False),
        "seed": request.get("seed"),
        "steps": request.get("steps"),
        "execution": request.get("execution"),
        "ltx_backend": request.get("ltx_backend"),
        "ltx_fast_av": request.get("ltx_fast_av"),
        "ltx_video_attention_batch": request.get(
            "ltx_video_attention_batch", False
        ),
        "allow_approximation": request.get("allow_approximation", False),
        "ltx_sol_stage2": request.get("ltx_sol_stage2", False),
        "ltx_sol_tau": request.get("ltx_sol_tau"),
        "ltx_sol_dense_edge_blocks": request.get(
            "ltx_sol_dense_edge_blocks"
        ),
        "ltx_sol_dense_edge_steps": request.get(
            "ltx_sol_dense_edge_steps"
        ),
    }


def load_native_library(config: dict[str, Any]) -> Any:
    for name, value in config.get("environment", {}).items():
        os.environ[name] = value
    library_path = Path(config["library"]).expanduser().resolve()
    library = c.CDLL(str(library_path))
    library.tc_string_free.argtypes = [c.c_void_p]
    constructor_name = (
        "tc_engine_create_model_candidate"
        if config.get("constructor", "public") == "candidate"
        else "tc_engine_create_model"
    )
    try:
        constructor = getattr(library, constructor_name)
    except AttributeError as exc:
        raise CampaignError(
            f"{library_path} does not export {constructor_name}"
        ) from exc
    constructor.argtypes = [
        c.c_char_p, c.c_char_p, c.POINTER(c.c_void_p), c.POINTER(c.c_void_p)
    ]
    library.tc_engine_generate.argtypes = [
        c.c_void_p, c.c_char_p, c.c_void_p, c.c_void_p,
        c.POINTER(c.c_void_p), c.POINTER(c.c_void_p),
    ]
    library.tc_engine_free.argtypes = [c.c_void_p]
    # Audit symbols are private and exist only in an audit build.  Keep the
    # function pointers on the CDLL object so the worker can reset/snapshot
    # counters around each request without changing the public ABI.
    try:
        audit_reset = library.tc_streaming_audit_reset
        audit_snapshot = library.tc_streaming_audit_snapshot_json
    except AttributeError:
        library._tc_streaming_audit_reset = None
        library._tc_streaming_audit_snapshot = None
    else:
        audit_reset.argtypes = []
        audit_reset.restype = None
        audit_snapshot.argtypes = [
            c.POINTER(c.c_void_p), c.POINTER(c.c_void_p)
        ]
        audit_snapshot.restype = c.c_int
        library._tc_streaming_audit_reset = audit_reset
        library._tc_streaming_audit_snapshot = audit_snapshot
    return library


def create_native_engine(library: Any, config: dict[str, Any]) -> c.c_void_p:
    model_path = Path(config["model_path"]).expanduser().resolve()
    constructor_name = (
        "tc_engine_create_model_candidate"
        if config.get("constructor", "public") == "candidate"
        else "tc_engine_create_model"
    )
    constructor = getattr(library, constructor_name)
    engine = c.c_void_p()
    error = c.c_void_p()
    status = constructor(
        config["model_id"].encode(), str(model_path).encode(),
        c.byref(engine), c.byref(error),
    )
    failure = consume(library, error)
    if status or not engine.value:
        raise CampaignError(failure or "native engine creation failed")
    return engine


def load_native_worker(config: dict[str, Any]) -> tuple[Any, c.c_void_p]:
    library = load_native_library(config)
    return library, create_native_engine(library, config)


def reset_native_audit(library: Any) -> bool:
    reset = getattr(library, "_tc_streaming_audit_reset", None)
    if reset is None:
        return False
    reset()
    return True


def snapshot_native_audit(library: Any) -> dict[str, Any] | None:
    snapshot_call = getattr(library, "_tc_streaming_audit_snapshot", None)
    if snapshot_call is None:
        return None
    result_pointer = c.c_void_p()
    error_pointer = c.c_void_p()
    status = snapshot_call(c.byref(result_pointer), c.byref(error_pointer))
    result_text = consume(library, result_pointer)
    failure = consume(library, error_pointer)
    if status or not result_text:
        raise CampaignError(failure or "native audit snapshot failed")
    try:
        value = json.loads(result_text)
    except json.JSONDecodeError as exc:
        raise CampaignError(f"native audit snapshot is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CampaignError("native audit snapshot must be a JSON object")
    return value


def hash_artifacts(paths: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, raw_path in paths.items():
        path = Path(raw_path)
        if path.is_file():
            result[name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        else:
            result[name] = {
                "path": str(path), "bytes": None, "sha256": None,
                "error": "artifact is missing or not a regular file",
            }
    return result


def run_native_sample(
    library: Any, engine: c.c_void_p, command: dict[str, Any],
    engine_lifecycle: str,
) -> dict[str, Any]:
    audit_available = reset_native_audit(library)
    result_pointer = c.c_void_p()
    error_pointer = c.c_void_p()
    started = time.perf_counter()
    status = library.tc_engine_generate(
        engine, canonical_json(command["request"]), None, None,
        c.byref(result_pointer), c.byref(error_pointer),
    )
    client_wall = time.perf_counter() - started
    result_text = consume(library, result_pointer)
    failure = consume(library, error_pointer)
    runtime_audit = snapshot_native_audit(library) if audit_available else None
    if status:
        row = {
            "status": "failure",
            "native_status": int(status),
            "error": failure or "native request failed without a diagnostic",
            "client_wall_seconds": client_wall,
            "process_peak_rss_bytes": peak_rss_bytes(),
        }
        if runtime_audit is not None:
            row["runtime_audit"] = runtime_audit
        return row
    if not result_text:
        row = {
            "status": "failure", "native_status": 0,
            "error": "native request succeeded without a result",
            "client_wall_seconds": client_wall,
            "process_peak_rss_bytes": peak_rss_bytes(),
        }
        if runtime_audit is not None:
            row["runtime_audit"] = runtime_audit
        return row
    result = json.loads(result_text)
    timings = result.get("timings_seconds") or {}
    stage1 = float(timings.get("stage1", 0.0))
    stage2 = float(timings.get("stage2", 0.0))
    denoise = timings.get("denoise")
    if denoise is None:
        denoise = stage1 + stage2
    resolved, actual = extract_layouts(result)
    block = result.get("block_streaming") or result.get("block_residency")
    semantic, semantic_missing = actual_semantic_layout(
        result, engine_lifecycle
    )
    if semantic is not None:
        semantic["request"] = request_semantic_identity(command["request"])
    row = {
        "status": "success",
        "native_status": 0,
        "client_wall_seconds": client_wall,
        "request_wall_seconds": float(timings.get("request_wall", client_wall)),
        "denoise_seconds": float(denoise),
        "stage_timings_seconds": timings,
        "resolved_layout_digest": resolved,
        "actual_layout_digest": actual,
        "actual_semantic_layout": semantic,
        "actual_semantic_layout_digest": semantic_digest(semantic),
        "actual_semantic_layout_missing_fields": semantic_missing,
        "streaming_implementation": extract_streaming_implementation(result),
        "engine_lifecycle": engine_lifecycle,
        "block_counters": block,
        "artifacts": hash_artifacts(command["artifact_paths"]),
        "process_peak_rss_bytes": peak_rss_bytes(),
        "runtime_audit": {
            "audit_available": audit_available,
            "block_streaming_enabled": (
                bool(block.get("enabled")) if isinstance(block, dict) else False
            ),
            "request_slot_allocations": (
                block.get("request_slot_allocations")
                if isinstance(block, dict) else None
            ),
            "request_slot_refills": (
                block.get("request_slot_refills")
                if isinstance(block, dict) else None
            ),
        },
    }
    if runtime_audit is not None:
        row["runtime_audit"].update(runtime_audit)
    return row


def run_synthetic_sample(
    config: dict[str, Any], command: dict[str, Any], engine_lifecycle: str
) -> dict[str, Any]:
    synthetic = config["synthetic"]
    identity = command["run_id"]
    sleep_seconds = float(synthetic.get("sleep_seconds", 0.0))
    if sleep_seconds:
        time.sleep(sleep_seconds)
    if identity in synthetic.get("fail_runs", []):
        return {
            "status": "failure",
            "native_status": 1,
            "error": f"injected synthetic failure for {identity}",
            "client_wall_seconds": sleep_seconds,
            "process_peak_rss_bytes": peak_rss_bytes(),
        }
    payloads = synthetic.get("artifact_payloads", {})
    for name, raw_path in command["artifact_paths"].items():
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = str(payloads.get(name, "shared-quality"))
        path.write_bytes((payload + ":" + command["pair_id"]).encode())
    wall = float(synthetic.get("request_wall_seconds", 1.0))
    denoise = float(synthetic.get("denoise_seconds", wall * 0.8))
    layout = synthetic.get("layout_digest")
    semantic = deepcopy(synthetic.get("actual_semantic_layout"))
    if isinstance(semantic, dict):
        semantic["engine_lifecycle"] = engine_lifecycle
        semantic["request"] = request_semantic_identity(command["request"])
        semantic_missing = [
            field for field in
            [*SEMANTIC_LAYOUT_FIELDS, "engine_lifecycle", "total_fills"]
            if field not in semantic
        ]
    else:
        semantic = None
        semantic_missing = [
            *SEMANTIC_LAYOUT_FIELDS, "engine_lifecycle", "total_fills"
        ]
    return {
        "status": "success",
        "native_status": 0,
        "client_wall_seconds": wall,
        "request_wall_seconds": wall,
        "denoise_seconds": denoise,
        "stage_timings_seconds": {"request_wall": wall, "denoise": denoise},
        "resolved_layout_digest": layout,
        "actual_layout_digest": layout,
        "actual_semantic_layout": semantic,
        "actual_semantic_layout_digest": semantic_digest(semantic),
        "actual_semantic_layout_missing_fields": semantic_missing,
        "streaming_implementation": synthetic.get(
            "streaming_implementation"
        ),
        "engine_lifecycle": engine_lifecycle,
        "block_counters": synthetic.get("block_counters", {"enabled": False}),
        "artifacts": hash_artifacts(command["artifact_paths"]),
        "process_peak_rss_bytes": peak_rss_bytes(),
        "runtime_audit": deepcopy(synthetic.get("runtime_audit", {
            "audit_available": True,
            "block_streaming_enabled": False,
            "request_slot_allocations": 0,
            "request_slot_refills": 0,
        })),
    }


def run_probe_sample(
    config: dict[str, Any], command: dict[str, Any], engine_lifecycle: str
) -> dict[str, Any]:
    executable = Path(config["executable"]).expanduser().resolve()
    model = Path(config["model_path"]).expanduser().resolve()
    artifact_name = config["output_artifact"]
    output = command["artifact_paths"].get(artifact_name)
    if not isinstance(output, str) or not output:
        raise CampaignError(
            f"probe output artifact is absent: {artifact_name}"
        )
    for raw_path in command["artifact_paths"].values():
        Path(raw_path).parent.mkdir(parents=True, exist_ok=True)
    replacements = {
        "MODEL": str(model),
        "OUTPUT": output,
        "PAIR_ID": command["pair_id"],
        "BLOCK_ID": command["block_id"],
        "VARIANT": command["variant"],
        "RUN_ID": command["run_id"],
    }
    arguments = expand(config["arguments"], replacements)
    environment = os.environ.copy()
    environment.update(config.get("environment", {}))
    working_directory = Path(
        config.get("working_directory", executable.parent)
    ).expanduser().resolve()
    timeout = float(config.get("request_timeout_seconds", 3600.0))
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(executable), *arguments], cwd=working_directory,
            env=environment, capture_output=True, text=True,
            timeout=max(1.0, timeout * 0.95), check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "native_status": None,
            "error": f"probe exceeded {timeout * 0.95:.3f} seconds",
            "client_wall_seconds": time.perf_counter() - started,
            "probe_stdout": (exc.stdout or "")[-4096:],
            "probe_stderr": (exc.stderr or "")[-4096:],
            "process_peak_rss_bytes": peak_rss_bytes(),
        }
    client_wall = time.perf_counter() - started
    if completed.returncode:
        return {
            "status": "failure",
            "native_status": int(completed.returncode),
            "error": completed.stderr.strip() or
                completed.stdout.strip() or "probe failed without a diagnostic",
            "client_wall_seconds": client_wall,
            "probe_stdout": completed.stdout[-4096:],
            "probe_stderr": completed.stderr[-4096:],
            "process_peak_rss_bytes": peak_rss_bytes(),
        }
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise CampaignError("probe succeeded without a JSON result")
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise CampaignError(f"probe result is invalid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise CampaignError("probe result must be a JSON object")
    expected_schema = config.get("result_schema")
    if expected_schema and result.get("schema") != expected_schema:
        raise CampaignError(
            f"probe result schema differs: {result.get('schema')!r}"
        )
    timings = result.get("timings_seconds") or {}
    block = result.get("block_streaming") or result.get("block_residency")
    semantic, semantic_missing = actual_semantic_layout(
        result, engine_lifecycle
    )
    if semantic is not None:
        semantic["request"] = request_semantic_identity(command["request"])
    audit_snapshot = result.get("audit_snapshot")
    runtime_audit = {
        "audit_available": isinstance(audit_snapshot, dict),
        "block_streaming_enabled": (
            bool(block.get("enabled")) if isinstance(block, dict) else False
        ),
        "request_slot_allocations": (
            block.get("request_slot_allocations")
            if isinstance(block, dict) else None
        ),
        "request_slot_refills": (
            block.get("request_slot_refills")
            if isinstance(block, dict) else None
        ),
    }
    if isinstance(audit_snapshot, dict):
        runtime_audit.update(audit_snapshot)
    request_wall = float(timings.get("request_wall", client_wall))
    denoise = float(timings.get("denoise", result.get("denoise_seconds", 0.0)))
    resolved, actual = extract_layouts(result)
    return {
        "status": "success",
        "native_status": 0,
        "client_wall_seconds": client_wall,
        "request_wall_seconds": request_wall,
        "denoise_seconds": denoise,
        "stage_timings_seconds": timings,
        "resolved_layout_digest": resolved,
        "actual_layout_digest": actual,
        "actual_semantic_layout": semantic,
        "actual_semantic_layout_digest": semantic_digest(semantic),
        "actual_semantic_layout_missing_fields": semantic_missing,
        "streaming_implementation": extract_streaming_implementation(result),
        "engine_lifecycle": engine_lifecycle,
        "block_counters": block,
        "artifacts": hash_artifacts(command["artifact_paths"]),
        "process_peak_rss_bytes": peak_rss_bytes(),
        "runtime_audit": runtime_audit,
        "probe_result": result,
    }


def worker_main(config_path: Path, descriptor: int) -> int:
    config = read_object(config_path, "worker config")
    engine_lifecycle = config.get("engine_lifecycle", "persistent")
    if engine_lifecycle not in ENGINE_LIFECYCLES:
        raise CampaignError("worker received an invalid engine lifecycle")
    connection = socket.socket(fileno=descriptor)
    library = None
    engine = None
    engine_generation = 0
    try:
        backend = config.get("backend", "native")
        if backend == "native":
            library = load_native_library(config)
            if engine_lifecycle == "persistent":
                engine = create_native_engine(library, config)
                engine_generation = 1
        elif engine_lifecycle == "persistent":
            engine_generation = 1
        send_message(connection, {
            "type": "ready", "pid": os.getpid(),
            "backend": config.get("backend", "native"),
            "engine_lifecycle": engine_lifecycle,
        })
        while True:
            command = receive_message(connection, 365 * 24 * 60 * 60)
            if command.get("type") == "stop":
                send_message(connection, {"type": "stopped", "pid": os.getpid()})
                return 0
            if command.get("type") != "run":
                raise CampaignError("unknown worker command")
            try:
                if backend == "native":
                    if engine_lifecycle == "persistent":
                        response = run_native_sample(
                            library, engine, command, engine_lifecycle
                        )
                    else:
                        lifecycle_started = time.perf_counter()
                        request_engine = create_native_engine(library, config)
                        engine_generation += 1
                        create_finished = time.perf_counter()
                        try:
                            response = run_native_sample(
                                library, request_engine, command,
                                engine_lifecycle,
                            )
                        finally:
                            destroy_started = time.perf_counter()
                            library.tc_engine_free(request_engine)
                            destroy_finished = time.perf_counter()
                        lifecycle_wall = destroy_finished - lifecycle_started
                        response["native_request_wall_seconds"] = response.get(
                            "request_wall_seconds"
                        )
                        response["request_wall_seconds"] = lifecycle_wall
                        response["client_wall_seconds"] = lifecycle_wall
                        response["engine_create_seconds"] = (
                            create_finished - lifecycle_started
                        )
                        response["engine_destroy_seconds"] = (
                            destroy_finished - destroy_started
                        )
                elif backend == "probe":
                    engine_generation += 1
                    response = run_probe_sample(
                        config, command, engine_lifecycle
                    )
                else:
                    if engine_lifecycle == "per_request":
                        engine_generation += 1
                    response = run_synthetic_sample(
                        config, command, engine_lifecycle
                    )
            except Exception as exc:  # Preserve worker for the next sample.
                response = {
                    "status": "worker_error", "error": str(exc),
                    "client_wall_seconds": 0.0,
                    "process_peak_rss_bytes": peak_rss_bytes(),
                }
            response.update({
                "type": "result", "run_id": command["run_id"],
                "worker_pid": os.getpid(),
                "engine_lifecycle": engine_lifecycle,
                "engine_generation": engine_generation,
            })
            send_message(connection, response)
    except Exception as exc:
        try:
            send_message(connection, {
                "type": "fatal", "error": str(exc), "pid": os.getpid(),
            })
        except Exception:
            pass
        return 2
    finally:
        if library is not None and engine is not None:
            library.tc_engine_free(engine)
        connection.close()


class Worker:
    def __init__(
        self,
        variant: str,
        config_path: Path,
        log_path: Path,
        request_timeout: float,
        start_timeout: float,
    ) -> None:
        self.variant = variant
        self.timeout = request_timeout
        parent, child = socket.socketpair()
        child.set_inheritable(True)
        self.connection = parent
        self.log = log_path.open("wb")
        command = [
            sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
            "--worker-config", str(config_path), "--worker-fd", str(child.fileno()),
        ]
        self.process = subprocess.Popen(
            command, pass_fds=(child.fileno(),), stdout=self.log, stderr=self.log,
            close_fds=True,
        )
        child.close()
        try:
            ready = receive_message(self.connection, start_timeout)
        except Exception:
            self.terminate()
            raise
        if ready.get("type") != "ready":
            self.terminate()
            raise CampaignError(
                f"{variant} worker failed to start: {ready.get('error', ready)}"
            )
        self.pid = int(ready["pid"])

    def run(self, command: dict[str, Any]) -> dict[str, Any]:
        send_message(self.connection, command)
        response = receive_message(self.connection, self.timeout)
        if response.get("type") not in ("result", "fatal"):
            raise CampaignError(f"invalid {self.variant} worker response")
        if response.get("type") == "fatal":
            raise CampaignError(response.get("error", "worker failed"))
        return response

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                send_message(self.connection, {"type": "stop"})
                receive_message(self.connection, min(self.timeout, 10.0))
            except Exception:
                pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.terminate()
        self.connection.close()
        self.log.close()

    def terminate(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        try:
            self.connection.close()
        except OSError:
            pass
        if not self.log.closed:
            self.log.close()


def planned_samples(blocks: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for block_index in range(blocks):
        sequence_name = "ABBA" if block_index % 2 == 0 else "BAAB"
        sequence = SEQUENCES[sequence_name]
        block_id = f"block-{block_index:03d}"
        pair_ids = (f"{block_id}-pair-0", f"{block_id}-pair-1")
        pair_for_position = (pair_ids[0], pair_ids[0], pair_ids[1], pair_ids[1])
        for position, variant in enumerate(sequence):
            result.append({
                "block_id": block_id,
                "block_index": block_index,
                "sequence": sequence_name,
                "position": position,
                "variant": variant,
                "pair_id": pair_for_position[position],
                "pair_index": block_index * 2 + (position // 2),
                "run_id": f"{block_id}-position-{position}-{variant}",
            })
    return result


def request_for(
    policy: dict[str, Any], sample: dict[str, Any], output: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    variant = sample["variant"]
    config = policy["variants"][variant]
    base = config.get("request", policy["workload"]["request"])
    request = deep_merge(base, config.get("request_patch", {}))
    seed_path = policy["workload"].get("seed_path")
    seed_stride = int(policy["workload"].get("seed_stride", 0))
    if seed_path and seed_stride:
        original = request
        for part in seed_path.split("."):
            original = original[part]
        if isinstance(original, bool) or not isinstance(original, int):
            raise CampaignError("workload seed must be an integer")
        set_dotted(request, seed_path, original + sample["pair_index"] * seed_stride)
    run_directory = output / "runs" / sample["run_id"]
    output_suffix = policy["workload"].get("output_suffix", ".bin")
    if not isinstance(output_suffix, str) or not output_suffix.startswith("."):
        raise CampaignError("workload.output_suffix must start with a dot")
    destination = run_directory / f"output{output_suffix}"
    dump = run_directory / "tensors"
    replacements = {
        "OUTPUT": str(destination),
        "DUMP": str(dump),
        "PAIR_ID": sample["pair_id"],
        "BLOCK_ID": sample["block_id"],
        "VARIANT": variant,
        "RUN_ID": sample["run_id"],
    }
    request = expand(request, replacements)
    artifacts = {
        item["name"]: expand(item["path"], replacements)
        for item in policy["quality"]["artifacts"]
    }
    return request, artifacts


def make_quality(raw: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for row in raw:
        pairs.setdefault(row["pair_id"], {})[row["variant"]] = row
    samples = []
    for pair_id, variants in sorted(pairs.items()):
        statuses = {
            variant: variants.get(variant, {}).get("status", "missing")
            for variant in VARIANTS
        }
        artifacts: dict[str, Any] = {}
        measured = all(status == "success" for status in statuses.values())
        passed: bool | None = True if measured else None
        names: set[str] = set()
        for row in variants.values():
            names.update((row.get("artifacts") or {}).keys())
        for name in sorted(names):
            baseline = nested(variants.get("baseline", {}), "artifacts", name, "sha256")
            candidate = nested(variants.get("candidate", {}), "artifacts", name, "sha256")
            equal = bool(baseline and candidate and baseline == candidate)
            artifacts[name] = {
                "baseline_sha256": baseline,
                "candidate_sha256": candidate,
                "equal": equal,
            }
            if passed is not None:
                passed = passed and equal
        if not names:
            passed = False if measured else None
        samples.append({
            "pair_id": pair_id,
            "passed": passed,
            "measured": measured,
            "statuses": statuses,
            "artifacts": artifacts,
        })
    return {
        "format": "turbocider-streaming-quality-v1",
        "mode": "artifact_sha256_equal",
        "status": "complete",
        "samples": samples,
    }


def make_semantic_equivalence(
    policy: dict[str, Any], raw: list[dict[str, Any]]
) -> dict[str, Any]:
    declaration = policy.get("semantic_equivalence")
    if not isinstance(declaration, dict):
        declaration = {}
    successful = [row for row in raw if row.get("status") == "success"]
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in successful:
        by_pair.setdefault(row["pair_id"], {})[row["variant"]] = {
            "layout_digest": row.get("actual_layout_digest"),
            "semantic_digest": row.get("actual_semantic_layout_digest"),
            "semantic_layout": row.get("actual_semantic_layout"),
            "missing_fields": row.get(
                "actual_semantic_layout_missing_fields", []
            ),
            "engine_lifecycle": row.get("engine_lifecycle"),
            "streaming_implementation": row.get(
                "streaming_implementation"
            ),
        }
    complete = {
        pair_id: values for pair_id, values in by_pair.items()
        if set(values) == set(VARIANTS)
    }
    observed_same = bool(complete) and all(
        values["baseline"]["semantic_digest"]
        and not values["baseline"]["missing_fields"]
        and not values["candidate"]["missing_fields"]
        and values["baseline"]["semantic_digest"] ==
            values["candidate"]["semantic_digest"]
        for values in complete.values()
    )
    expected = declaration.get("expected_actual")
    observed_expected = bool(complete) and isinstance(expected, dict)
    if observed_expected:
        for values in complete.values():
            for variant in VARIANTS:
                layout = values[variant].get("semantic_layout")
                if (
                    not isinstance(layout, dict)
                    or any(
                        layout.get(field) != expected_value
                        for field, expected_value in expected.items()
                    )
                ):
                    observed_expected = False
                    break
            if not observed_expected:
                break
    declared = declaration.get("declared_equivalent") is True
    return {
        "format": "turbocider-streaming-semantic-equivalence-v2",
        "equivalent": bool(
            declared and observed_same and observed_expected
        ),
        "declared_equivalent": declared,
        "declaration": declaration,
        "observed_same_actual_layout": observed_same,
        "observed_same_semantic_layout": observed_same,
        "observed_matches_expected": observed_expected,
        "expected_actual": expected,
        "pair_layouts": by_pair,
        "comparison": "normalized_actual_semantic_layout",
    }


def default_audit(raw: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    candidate = [row for row in raw if row["variant"] == "candidate"]
    successful = [row for row in candidate if row.get("status") == "success"]
    enabled = [row.get("runtime_audit", {}).get("block_streaming_enabled")
               for row in successful]
    observations: dict[str, Any] = {
        "candidate_successful_requests": len(successful),
        "candidate_total_requests": len(candidate),
        "block_streaming_enabled_values": enabled,
    }
    required = (
        "new_framework_hooks", "new_memory_probes", "new_worker_threads",
        "new_pool_allocations", "new_cache_clear_or_unload_calls",
    )
    if policy.get("comparison_kind") == "P1":
        required += (
            "steady_framework_allocations",
            "steady_framework_thread_creates",
        )
    complete = bool(successful) and all(
        row.get("runtime_audit", {}).get("audit_available") is True and
        all(isinstance(row.get("runtime_audit", {}).get(name), int) and
            not isinstance(row.get("runtime_audit", {}).get(name), bool) and
            row.get("runtime_audit", {}).get(name) >= 0 for name in required)
        for row in successful
    )
    if complete:
        totals = {
            name: sum(row["runtime_audit"][name] for row in successful)
            for name in required
        }
        observations["per_request"] = [
            {"run_id": row.get("run_id"), **{
                name: row["runtime_audit"][name] for name in required
            }} for row in successful
        ]
        observations["counter_totals"] = totals
        observations["audit_build"] = True
        if policy.get("comparison_kind") == "P0":
            passed = not any(totals.values()) and len(successful) == len(candidate)
            return {
                "format": "turbocider-streaming-audit-v1",
                "status": "passed" if passed else "failed",
                "passed": passed,
                "reason": (
                    "default candidate requests did not enter the new framework"
                    if passed else
                    "default candidate requests failed or entered a new framework callsite"
                ),
                **totals,
                "runtime_observations": observations,
            }
        if policy.get("comparison_kind") == "P1":
            passed = (
                len(successful) == len(candidate) and
                all(enabled) and
                totals["steady_framework_allocations"] == 0 and
                totals["steady_framework_thread_creates"] == 0
            )
            return {
                "format": "turbocider-streaming-audit-v1",
                "status": "passed" if passed else "failed",
                "passed": passed,
                "reason": (
                    "opt-in requests used the framework without steady "
                    "allocation or thread creation"
                    if passed else
                    "opt-in requests failed, bypassed the framework, or "
                    "allocated/created threads in the steady path"
                ),
                **totals,
                "runtime_observations": observations,
            }
    observations["audit_build"] = False
    return {
        "format": "turbocider-streaming-audit-v1",
        "status": "partial",
        "passed": None,
        "reason": (
            "runtime observations are present, but hook/probe/thread/pool and "
            "cache-clear counters require an independent audit build"
        ),
        "runtime_observations": observations,
    }


def default_environment() -> dict[str, Any]:
    return {
        "format": "turbocider-streaming-environment-v1",
        "status": "partial",
        "reason": (
            "automatic capture records the process platform only; GPU, RAM, "
            "SSD, power, thermal, SDK and pressure protocol require an "
            "operator-supplied environment record"
        ),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
    }


def make_faults(
    raw: list[dict[str, Any]], warmups: list[dict[str, Any]]
) -> dict[str, Any]:
    failures = [
        {
            "phase": phase,
            "run_id": row.get("run_id"),
            "block_id": row.get("block_id"),
            "pair_id": row.get("pair_id"),
            "variant": row.get("variant"),
            "status": row.get("status"),
            "error": row.get("error"),
        }
        for phase, rows in (("warmup", warmups), ("measured", raw))
        for row in rows if row.get("status") != "success"
    ]
    return {
        "format": "turbocider-streaming-faults-v1",
        "status": "complete",
        "failure_count": len(failures),
        "samples": failures,
    }


def manifest_files(output: Path, names: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in names:
        path = output / name
        if path.is_file():
            result[name] = {
                "sha256": sha256_file(path), "bytes": path.stat().st_size,
            }
    return result


def run_campaign(
    policy_path: Path,
    output: Path,
    audit_path: Path | None = None,
    environment_path: Path | None = None,
) -> dict[str, Any]:
    policy_path = policy_path.resolve()
    policy_bytes = policy_path.read_bytes()
    policy = read_object(policy_path, "campaign policy")
    validate_policy(policy)
    reject_output_inside_sources(policy, output)
    if output.exists():
        raise CampaignError(f"output must not already exist: {output}")
    output.mkdir(parents=True)
    (output / "workers").mkdir()
    shutil.copyfile(policy_path, output / "campaign-policy.json")
    raw_path = output / "raw-samples.jsonl"
    warmup_path = output / "warmups.jsonl"
    raw_path.touch()
    warmup_path.touch()
    memory_enabled = (
        isinstance(policy.get("memory_sampling"), dict) and
        policy["memory_sampling"].get("enabled") is True
    )
    memory_summary_path = output / "memory-summaries.jsonl"
    if memory_enabled:
        memory_summary_path.touch()
    identities = build_identity(policy)
    write_json(output / "build-identity.json", identities)
    timeout = float(policy["protocol"]["request_timeout_seconds"])
    start_timeout = float(
        policy["protocol"].get("worker_start_timeout_seconds", 30)
    )
    worker_configs: dict[str, Path] = {}
    engine_lifecycle = policy.get("engine_lifecycle", "persistent")
    for variant in VARIANTS:
        path = output / "workers" / f"{variant}.json"
        worker_config = deepcopy(policy["variants"][variant])
        worker_config["engine_lifecycle"] = engine_lifecycle
        worker_config["request_timeout_seconds"] = timeout
        write_json(path, worker_config)
        worker_configs[variant] = path
    manifest = {
        "format": "turbocider-streaming-campaign-manifest-v1",
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "policy_sha256": sha256_bytes(policy_bytes),
        "pressure_launched_by_runner": False,
        "engine_lifecycle": engine_lifecycle,
        "worker_launch_order": policy["protocol"].get(
            "worker_launch_order", list(VARIANTS)
        ),
    }
    write_json(output / "manifest.json", manifest)
    workers: dict[str, Worker] = {}
    raw: list[dict[str, Any]] = []
    warmups: list[dict[str, Any]] = []
    aborted = False
    abort_reason = None
    plan = planned_samples(int(policy["protocol"]["measured_blocks"]))
    try:
        launch_order = tuple(
            policy["protocol"].get("worker_launch_order", list(VARIANTS))
        )
        for variant in launch_order:
            workers[variant] = Worker(
                variant, worker_configs[variant],
                output / "workers" / f"{variant}.log", timeout, start_timeout,
            )
        warmup_count = int(policy["protocol"].get("warmup_requests_per_variant", 1))
        warmup_failed = False
        for warmup_index in range(warmup_count):
            order = VARIANTS if warmup_index % 2 == 0 else tuple(reversed(VARIANTS))
            for variant in order:
                sample = {
                    "block_id": f"warmup-{warmup_index:03d}",
                    "block_index": -1,
                    "pair_id": f"warmup-{warmup_index:03d}",
                    "pair_index": warmup_index,
                    "position": 0,
                    "variant": variant,
                    "run_id": f"warmup-{warmup_index:03d}-{variant}",
                }
                request, artifacts = request_for(policy, sample, output)
                command = {
                    "type": "run", **sample, "request": request,
                    "artifact_paths": artifacts,
                }
                try:
                    response = run_worker_request(
                        workers[variant], command, output, policy, "warmup"
                    )
                except socket.timeout:
                    response = {
                        "status": "timeout", "error": "warmup timed out",
                        "worker_pid": workers[variant].pid,
                    }
                    workers[variant].terminate()
                row = {**sample, **response}
                warmups.append(row)
                append_jsonl(warmup_path, row)
                if row.get("memory_summary") is not None:
                    append_jsonl(memory_summary_path, row["memory_summary"])
                if row.get("status") != "success":
                    warmup_failed = True
                    aborted = True
                    abort_reason = (
                        f"{variant} warmup failed: "
                        f"{row.get('error', row['status'])}"
                    )
                    break
            if warmup_failed:
                break
        for sample_index, sample in enumerate(plan):
            if aborted:
                row = {
                    **sample,
                    "sample_index": sample_index,
                    "status": "not_run_after_abort",
                    "error": abort_reason,
                    "worker_pid": workers.get(sample["variant"], None).pid
                    if sample["variant"] in workers else None,
                }
            else:
                request, artifacts = request_for(policy, sample, output)
                command = {
                    "type": "run", **sample, "sample_index": sample_index,
                    "request": request, "artifact_paths": artifacts,
                }
                try:
                    response = run_worker_request(
                        workers[sample["variant"]], command, output, policy,
                        "measured",
                    )
                    row = {**sample, "sample_index": sample_index, **response}
                    if row.get("run_id") != sample["run_id"]:
                        raise CampaignError("worker returned a different run_id")
                    if row.get("status") == "worker_error":
                        aborted = True
                        abort_reason = row.get("error", "worker error")
                except socket.timeout:
                    row = {
                        **sample,
                        "sample_index": sample_index,
                        "status": "timeout",
                        "error": f"request exceeded {timeout} seconds",
                        "worker_pid": workers[sample["variant"]].pid,
                    }
                    workers[sample["variant"]].terminate()
                    aborted = True
                    abort_reason = row["error"]
                except (EOFError, OSError, CampaignError) as exc:
                    row = {
                        **sample,
                        "sample_index": sample_index,
                        "status": "worker_error",
                        "error": str(exc),
                        "worker_pid": workers[sample["variant"]].pid,
                    }
                    aborted = True
                    abort_reason = str(exc)
            raw.append(row)
            append_jsonl(raw_path, row)
            if row.get("memory_summary") is not None:
                append_jsonl(memory_summary_path, row["memory_summary"])
    finally:
        for worker in workers.values():
            try:
                worker.close()
            except Exception:
                pass
    quality = make_quality(raw)
    write_json(output / "quality.json", quality)
    if audit_path:
        audit = read_object(audit_path.resolve(), "audit evidence")
    else:
        audit = default_audit(raw, policy)
    write_json(output / "audit.json", audit)
    if environment_path:
        environment = read_object(
            environment_path.resolve(), "environment evidence"
        )
    else:
        environment = default_environment()
    write_json(output / "environment.json", environment)
    write_json(output / "faults.json", make_faults(raw, warmups))
    if policy["comparison_kind"] == "P1":
        write_json(
            output / "semantic-equivalence.json",
            make_semantic_equivalence(policy, raw),
        )
    manifest.update({
        "status": "aborted" if aborted else "complete",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "planned_requests": len(plan),
        "recorded_requests": len(raw),
        "planned_pairs": len(plan) // 2,
        "aborted": aborted,
        "abort_reason": abort_reason,
    })
    names = [
        "campaign-policy.json", "build-identity.json", "raw-samples.jsonl",
        "warmups.jsonl", "quality.json", "audit.json", "environment.json",
        "faults.json",
    ]
    if memory_enabled:
        names.append("memory-summaries.jsonl")
        memory_directory = output / "memory"
        if memory_directory.is_dir():
            names.extend(
                str(path.relative_to(output))
                for path in sorted(memory_directory.iterdir())
                if path.is_file()
            )
    if policy["comparison_kind"] == "P1":
        names.append("semantic-equivalence.json")
    manifest["files"] = manifest_files(output, names)
    write_json(output / "manifest.json", manifest)
    try:
        summary = verify(output)
    except EvidenceError as exc:
        summary = {
            "format": "turbocider-streaming-campaign-verification-v1",
            "overall": "INVALID",
            "error": str(exc),
        }
    write_json(output / "summary.json", summary)
    return summary


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.worker_config is None or args.worker_fd is None:
            parser.error("worker mode requires config and descriptor")
    elif args.policy is None or args.output is None:
        parser.error("--policy and --output are required")
    return args


def main() -> int:
    args = arguments()
    if args.worker:
        return worker_main(args.worker_config.resolve(), args.worker_fd)
    try:
        summary = run_campaign(
            args.policy.resolve(), args.output.resolve(),
            args.audit.resolve() if args.audit else None,
            args.environment.resolve() if args.environment else None,
        )
    except (CampaignError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"overall": "INVALID", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(summary, indent=2))
    if summary.get("overall") == "PASS":
        return 0
    return 2 if summary.get("overall") == "INVALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
