#!/usr/bin/env python3
"""Append-only process-tree memory sampler for streaming calibration.

The sampler is deliberately independent from the TurboCider runtime.  It can
either attach to an existing root PID or launch a command, recursively tracks
descendants (including children that are later re-parented), and writes a
hash-chained JSONL evidence stream.  On Darwin it uses libproc for per-process
RSS/physical-footprint counters and host_statistics64 for system compression
and swap counters.

This tool does not create memory pressure, alter swap policy, or infer missing
driver/GPU bytes.  Missing counters and sampling gaps remain explicit so the
catalog verifier can fail closed.
"""

from __future__ import annotations

import argparse
import ctypes as c
import errno
import fnmatch
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence


SCHEMA = "turbocider-process-tree-sampler-v1"
REVISION = "tc-process-tree-sampler-darwin-v2"
DEFAULT_INTERVAL_MS = 20
DEFAULT_MAX_GAP_MS = 100
EXIT_INCONCLUSIVE = 3


class SamplerError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


@dataclass(frozen=True, order=True)
class ProcessIdentity:
    pid: int
    start_seconds: int
    start_microseconds: int


@dataclass(frozen=True)
class ProcessSnapshot:
    identity: ProcessIdentity
    ppid: int
    name: str
    executable: str | None
    resident_bytes: int
    phys_footprint_bytes: int
    lifetime_max_phys_footprint_bytes: int
    wired_bytes: int
    pageins: int
    disk_read_bytes: int
    disk_write_bytes: int


@dataclass(frozen=True)
class SystemSnapshot:
    page_size: int
    pageins: int
    pageouts: int
    compressions: int
    decompressions: int
    swapins: int
    swapouts: int
    compressor_pages: int
    swapped_pages: int


class SamplingBackend(Protocol):
    revision: str

    def capabilities(self) -> dict[str, Any]: ...

    def process(self, pid: int) -> ProcessSnapshot | None: ...

    def children(self, pid: int) -> list[int]: ...

    def system(self) -> SystemSnapshot: ...


class Clock(Protocol):
    def monotonic_ns(self) -> int: ...

    def wall_time_ns(self) -> int: ...

    def sleep(self, seconds: float) -> None: ...


class RealClock:
    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def wall_time_ns(self) -> int:
        return time.time_ns()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class HashChainJsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.stream = path.open("x", encoding="utf-8")
        self.sequence = 0
        self.previous_digest = "0" * 64

    def write(self, value: dict[str, Any], *, sync: bool = False) -> dict[str, Any]:
        record = dict(value)
        record["schema"] = SCHEMA
        record["sequence"] = self.sequence
        record["previous_digest"] = self.previous_digest
        record["record_digest"] = hashlib.sha256(canonical_json(record)).hexdigest()
        self.stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        self.stream.flush()
        if sync:
            os.fsync(self.stream.fileno())
        self.previous_digest = record["record_digest"]
        self.sequence += 1
        return record

    def close(self) -> None:
        if self.stream.closed:
            return
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()

    def __enter__(self) -> "HashChainJsonlWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class _ProcBsdInfo(c.Structure):
    _fields_ = [
        ("pbi_flags", c.c_uint32),
        ("pbi_status", c.c_uint32),
        ("pbi_xstatus", c.c_uint32),
        ("pbi_pid", c.c_uint32),
        ("pbi_ppid", c.c_uint32),
        ("pbi_uid", c.c_uint32),
        ("pbi_gid", c.c_uint32),
        ("pbi_ruid", c.c_uint32),
        ("pbi_rgid", c.c_uint32),
        ("pbi_svuid", c.c_uint32),
        ("pbi_svgid", c.c_uint32),
        ("rfu_1", c.c_uint32),
        ("pbi_comm", c.c_char * 16),
        ("pbi_name", c.c_char * 32),
        ("pbi_nfiles", c.c_uint32),
        ("pbi_pgid", c.c_uint32),
        ("pbi_pjobc", c.c_uint32),
        ("e_tdev", c.c_uint32),
        ("e_tpgid", c.c_uint32),
        ("pbi_nice", c.c_int32),
        ("pbi_start_tvsec", c.c_uint64),
        ("pbi_start_tvusec", c.c_uint64),
    ]


class _RusageInfoV4(c.Structure):
    _fields_ = [("ri_uuid", c.c_uint8 * 16)] + [
        (name, c.c_uint64)
        for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups",
            "ri_interrupt_wkups", "ri_pageins", "ri_wired_size",
            "ri_resident_size", "ri_phys_footprint",
            "ri_proc_start_abstime", "ri_proc_exit_abstime",
            "ri_child_user_time", "ri_child_system_time",
            "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
            "ri_child_pageins", "ri_child_elapsed_abstime",
            "ri_diskio_bytesread", "ri_diskio_byteswritten",
            "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
            "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility",
            "ri_cpu_time_qos_legacy", "ri_cpu_time_qos_user_initiated",
            "ri_cpu_time_qos_user_interactive", "ri_billed_system_time",
            "ri_serviced_system_time", "ri_logical_writes",
            "ri_lifetime_max_phys_footprint", "ri_instructions",
            "ri_cycles", "ri_billed_energy", "ri_serviced_energy",
            "ri_interval_max_phys_footprint", "ri_runnable_time",
        )
    ]


class _VmStatistics64(c.Structure):
    _fields_ = [
        ("free_count", c.c_uint32),
        ("active_count", c.c_uint32),
        ("inactive_count", c.c_uint32),
        ("wire_count", c.c_uint32),
        ("zero_fill_count", c.c_uint64),
        ("reactivations", c.c_uint64),
        ("pageins", c.c_uint64),
        ("pageouts", c.c_uint64),
        ("faults", c.c_uint64),
        ("cow_faults", c.c_uint64),
        ("lookups", c.c_uint64),
        ("hits", c.c_uint64),
        ("purges", c.c_uint64),
        ("purgeable_count", c.c_uint32),
        ("speculative_count", c.c_uint32),
        ("decompressions", c.c_uint64),
        ("compressions", c.c_uint64),
        ("swapins", c.c_uint64),
        ("swapouts", c.c_uint64),
        ("compressor_page_count", c.c_uint32),
        ("throttled_count", c.c_uint32),
        ("external_page_count", c.c_uint32),
        ("internal_page_count", c.c_uint32),
        ("total_uncompressed_pages_in_compressor", c.c_uint64),
        ("swapped_count", c.c_uint64),
        ("total_tag_storage_pages", c.c_uint64),
        ("nontag_pageable_tag_storage_pages", c.c_uint64),
        ("nontag_wired_tag_storage_pages", c.c_uint64),
        ("free_tag_storage_pages", c.c_uint64),
        ("tag_storing_tag_storage_pages", c.c_uint64),
        ("total_tagged_pages", c.c_uint64),
        ("resident_tagged_pages", c.c_uint64),
        ("compressed_tagged_pages", c.c_uint64),
        ("tagged_compressions", c.c_uint64),
        ("tagged_decompressions", c.c_uint64),
        ("compressed_tag_storage_bytes", c.c_uint64),
    ]


def _decode_c_string(value: bytes) -> str:
    return value.split(b"\0", 1)[0].decode("utf-8", "replace")


class DarwinBackend:
    revision = REVISION
    _PROC_PIDTBSDINFO = 3
    _RUSAGE_INFO_V4 = 4
    _HOST_VM_INFO64 = 4
    _PROC_PIDPATHINFO_MAXSIZE = 4096

    def __init__(self):
        if sys.platform != "darwin":
            raise SamplerError("Darwin process sampling is only available on macOS")
        self.libproc = c.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        self.libsystem = c.CDLL(None, use_errno=True)
        self.libproc.proc_listchildpids.argtypes = [
            c.c_int, c.c_void_p, c.c_int,
        ]
        self.libproc.proc_listchildpids.restype = c.c_int
        self.libproc.proc_pidinfo.argtypes = [
            c.c_int, c.c_int, c.c_uint64, c.c_void_p, c.c_int,
        ]
        self.libproc.proc_pidinfo.restype = c.c_int
        self.libproc.proc_pid_rusage.argtypes = [
            c.c_int, c.c_int, c.c_void_p,
        ]
        self.libproc.proc_pid_rusage.restype = c.c_int
        self.libproc.proc_pidpath.argtypes = [c.c_int, c.c_void_p, c.c_uint32]
        self.libproc.proc_pidpath.restype = c.c_int
        self.libsystem.mach_host_self.argtypes = []
        self.libsystem.mach_host_self.restype = c.c_uint32
        self.libsystem.host_statistics64.argtypes = [
            c.c_uint32, c.c_int, c.c_void_p, c.POINTER(c.c_uint32),
        ]
        self.libsystem.host_statistics64.restype = c.c_int
        self.page_size = int(os.sysconf("SC_PAGE_SIZE"))

    def capabilities(self) -> dict[str, Any]:
        return {
            "process_resident_bytes": True,
            "process_phys_footprint_bytes": True,
            "process_compressed_bytes": False,
            "system_compressor_bytes": True,
            "system_swap_counters": True,
            "system_swap_used_bytes": True,
            "gpu_driver_bytes": False,
            "pressure_state": False,
            "child_discovery": "recursive-libproc-polling",
        }

    def _bsd(self, pid: int) -> _ProcBsdInfo | None:
        value = _ProcBsdInfo()
        size = self.libproc.proc_pidinfo(
            pid, self._PROC_PIDTBSDINFO, 0, c.byref(value), c.sizeof(value)
        )
        if size == c.sizeof(value):
            return value
        if size <= 0 and c.get_errno() in (errno.ESRCH, errno.EPERM):
            return None
        return None

    def process(self, pid: int) -> ProcessSnapshot | None:
        bsd = self._bsd(pid)
        if bsd is None:
            return None
        usage = _RusageInfoV4()
        if self.libproc.proc_pid_rusage(
            pid, self._RUSAGE_INFO_V4, c.byref(usage)
        ) != 0:
            if c.get_errno() in (errno.ESRCH, errno.EPERM):
                return None
            return None
        path_buffer = c.create_string_buffer(self._PROC_PIDPATHINFO_MAXSIZE)
        path_size = self.libproc.proc_pidpath(
            pid, path_buffer, self._PROC_PIDPATHINFO_MAXSIZE
        )
        executable = (
            path_buffer.value.decode("utf-8", "replace")
            if path_size > 0 else None
        )
        name = _decode_c_string(bytes(bsd.pbi_name)) or _decode_c_string(
            bytes(bsd.pbi_comm)
        )
        return ProcessSnapshot(
            ProcessIdentity(
                pid, int(bsd.pbi_start_tvsec), int(bsd.pbi_start_tvusec)
            ),
            int(bsd.pbi_ppid), name, executable,
            int(usage.ri_resident_size), int(usage.ri_phys_footprint),
            int(usage.ri_lifetime_max_phys_footprint),
            int(usage.ri_wired_size), int(usage.ri_pageins),
            int(usage.ri_diskio_bytesread), int(usage.ri_diskio_byteswritten),
        )

    def children(self, pid: int) -> list[int]:
        required = self.libproc.proc_listchildpids(pid, None, 0)
        if required <= 0:
            return []
        # Darwin's proc_listchildpids has returned both a byte-sized required
        # buffer and an element count from different SDK/runtime combinations;
        # over-allocate from the former and normalize the latter below.
        capacity = max(required // c.sizeof(c.c_int), required) + 8
        values = (c.c_int * capacity)()
        used = self.libproc.proc_listchildpids(
            pid, values, c.sizeof(values)
        )
        if used <= 0:
            return []
        count = used if used <= capacity else used // c.sizeof(c.c_int)
        return sorted({int(values[index]) for index in range(count) if values[index] > 0})

    def system(self) -> SystemSnapshot:
        value = _VmStatistics64()
        count = c.c_uint32(c.sizeof(value) // c.sizeof(c.c_int32))
        result = self.libsystem.host_statistics64(
            self.libsystem.mach_host_self(), self._HOST_VM_INFO64,
            c.byref(value), c.byref(count)
        )
        if result != 0:
            raise SamplerError(f"host_statistics64 failed with kern_return_t={result}")
        return SystemSnapshot(
            self.page_size, int(value.pageins), int(value.pageouts),
            int(value.compressions), int(value.decompressions),
            int(value.swapins), int(value.swapouts),
            int(value.compressor_page_count), int(value.swapped_count),
        )


@dataclass(frozen=True)
class RoleRule:
    role: str
    pattern: str

    def matches(self, snapshot: ProcessSnapshot) -> bool:
        candidates = [snapshot.name]
        if snapshot.executable:
            candidates.extend(
                [snapshot.executable, os.path.basename(snapshot.executable)]
            )
        return any(fnmatch.fnmatchcase(value, self.pattern) for value in candidates)


def parse_role_rule(value: str) -> RoleRule:
    if "=" not in value:
        raise argparse.ArgumentTypeError("role rules must use ROLE=GLOB")
    role, pattern = value.split("=", 1)
    if not role or not pattern:
        raise argparse.ArgumentTypeError("role rules must use non-empty ROLE=GLOB")
    return RoleRule(role, pattern)


def read_phase(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SamplerError(f"cannot read phase snapshot {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SamplerError("phase snapshot must be a JSON object")
    allowed = {
        "phase", "stage", "pass", "group", "pool", "slot", "fills",
        "fences_issued", "fences_completed", "mlx_active_bytes",
        "mlx_peak_bytes", "driver_bytes",
    }
    return {key: value[key] for key in allowed if key in value}


def system_delta(current: SystemSnapshot, baseline: SystemSnapshot) -> dict[str, int]:
    page_size = current.page_size
    if page_size != baseline.page_size:
        raise SamplerError("system page size changed during sampling")

    def delta(current_value: int, baseline_value: int, label: str) -> int:
        if current_value < baseline_value:
            raise SamplerError(f"system counter moved backwards: {label}")
        return current_value - baseline_value

    return {
        "pagein_bytes": delta(current.pageins, baseline.pageins, "pageins") * page_size,
        "pageout_bytes": delta(current.pageouts, baseline.pageouts, "pageouts") * page_size,
        "compression_bytes": delta(
            current.compressions, baseline.compressions, "compressions"
        ) * page_size,
        "decompression_bytes": delta(
            current.decompressions, baseline.decompressions, "decompressions"
        ) * page_size,
        "swap_in_bytes": delta(current.swapins, baseline.swapins, "swapins") * page_size,
        "swap_out_bytes": delta(current.swapouts, baseline.swapouts, "swapouts") * page_size,
    }


class ProcessTreeSampler:
    def __init__(
        self,
        backend: SamplingBackend,
        clock: Clock,
        writer: HashChainJsonlWriter,
        root_pid: int,
        *,
        root_role: str,
        role_rules: Sequence[RoleRule],
        interval_ns: int,
        max_gap_ns: int,
        correlation_id: str,
        phase_file: Path | None = None,
    ):
        if root_pid <= 0:
            raise SamplerError("root PID must be positive")
        if interval_ns <= 0 or max_gap_ns < interval_ns:
            raise SamplerError("sampling interval/max gap are invalid")
        if not correlation_id:
            raise SamplerError("correlation id must be non-empty")
        self.backend = backend
        self.clock = clock
        self.writer = writer
        self.root_pid = root_pid
        self.root_role = root_role
        self.role_rules = list(role_rules)
        self.interval_ns = interval_ns
        self.max_gap_ns = max_gap_ns
        self.correlation_id = correlation_id
        self.phase_file = phase_file
        self.known: dict[ProcessIdentity, ProcessSnapshot] = {}
        self.roles: dict[ProcessIdentity, str] = {}
        self.alive: set[ProcessIdentity] = set()
        self.unknown_children: set[ProcessIdentity] = set()
        self.errors: list[str] = []
        self.sample_count = 0
        self.last_sample_ns: int | None = None
        self.max_observed_gap_ns = 0
        self.tree_peak_rss_bytes = 0
        self.tree_peak_phys_footprint_bytes = 0
        self.tree_peak_rss_mono_ns = 0
        self.tree_peak_phys_footprint_mono_ns = 0
        self.root_identity: ProcessIdentity | None = None
        self.system_baseline: SystemSnapshot | None = None
        self.system_last: SystemSnapshot | None = None
        self.last_root_alive = False

    def _role(self, snapshot: ProcessSnapshot, is_root: bool) -> str:
        if is_root:
            return self.root_role
        for rule in self.role_rules:
            if rule.matches(snapshot):
                return rule.role
        self.unknown_children.add(snapshot.identity)
        return "unknown"

    def _discover(self) -> dict[ProcessIdentity, ProcessSnapshot]:
        expected: dict[int, ProcessIdentity] = {}
        if self.root_identity is not None:
            expected[self.root_pid] = self.root_identity
        for identity in sorted(self.alive):
            expected[identity.pid] = identity
        pending: list[tuple[int, ProcessIdentity | None]] = list(expected.items())
        visited_pids: set[int] = set()
        snapshots: dict[ProcessIdentity, ProcessSnapshot] = {}
        while pending:
            pid, expected_identity = pending.pop(0)
            if pid in visited_pids:
                continue
            visited_pids.add(pid)
            snapshot = self.backend.process(pid)
            if snapshot is None or (
                expected_identity is not None and
                snapshot.identity != expected_identity
            ):
                continue
            snapshots[snapshot.identity] = snapshot
            for child in self.backend.children(pid):
                if child not in visited_pids:
                    pending.append((child, None))
        return snapshots

    def _process_record(
        self, snapshot: ProcessSnapshot, role: str
    ) -> dict[str, Any]:
        value = asdict(snapshot)
        value["identity"] = asdict(snapshot.identity)
        value["role"] = role
        value["compressed_bytes"] = None
        return value

    def start(self) -> None:
        root = self.backend.process(self.root_pid)
        if root is None:
            raise SamplerError(f"root process {self.root_pid} is unavailable")
        self.root_identity = root.identity
        self.system_baseline = self.backend.system()
        self.system_last = self.system_baseline
        self.writer.write({
            "type": "sampler_start",
            "revision": self.backend.revision,
            "mono_ns": self.clock.monotonic_ns(),
            "wall_time_ns": self.clock.wall_time_ns(),
            "root_pid": self.root_pid,
            "root_identity": asdict(root.identity),
            "root_role": self.root_role,
            "role_rules": [asdict(rule) for rule in self.role_rules],
            "interval_ns": self.interval_ns,
            "max_gap_ns": self.max_gap_ns,
            "correlation_id": self.correlation_id,
            "platform": platform.platform(),
            "capabilities": self.backend.capabilities(),
        }, sync=True)

    def sample(self) -> bool:
        if self.root_identity is None or self.system_baseline is None:
            raise SamplerError("sampler was not started")
        mono_ns = self.clock.monotonic_ns()
        wall_ns = self.clock.wall_time_ns()
        gap_ns = 0 if self.last_sample_ns is None else mono_ns - self.last_sample_ns
        if gap_ns < 0:
            raise SamplerError("monotonic clock moved backwards")
        self.max_observed_gap_ns = max(self.max_observed_gap_ns, gap_ns)
        snapshots = self._discover()
        current_identities = set(snapshots)
        self.last_root_alive = self.root_identity in current_identities
        for identity in sorted(current_identities - set(self.known)):
            snapshot = snapshots[identity]
            role = self._role(snapshot, identity == self.root_identity)
            self.known[identity] = snapshot
            self.roles[identity] = role
            self.writer.write({
                "type": "process_start",
                "mono_ns": mono_ns,
                "wall_time_ns": wall_ns,
                "correlation_id": self.correlation_id,
                "process": self._process_record(snapshot, role),
            })
        for identity in sorted(self.alive - current_identities):
            self.writer.write({
                "type": "process_exit",
                "mono_ns": mono_ns,
                "wall_time_ns": wall_ns,
                "correlation_id": self.correlation_id,
                "identity": asdict(identity),
                "role": self.roles.get(identity, "unknown"),
            })
        self.alive = current_identities
        for identity, snapshot in snapshots.items():
            self.known[identity] = snapshot

        system = self.backend.system()
        self.system_last = system
        phase = read_phase(self.phase_file)
        processes = [
            self._process_record(snapshots[identity], self.roles[identity])
            for identity in sorted(snapshots)
        ]
        tree_rss = sum(process["resident_bytes"] for process in processes)
        tree_phys = sum(process["phys_footprint_bytes"] for process in processes)
        if tree_phys > self.tree_peak_phys_footprint_bytes:
            self.tree_peak_phys_footprint_bytes = tree_phys
            self.tree_peak_phys_footprint_mono_ns = mono_ns
        if tree_rss > self.tree_peak_rss_bytes:
            self.tree_peak_rss_bytes = tree_rss
            self.tree_peak_rss_mono_ns = mono_ns
        system_value = asdict(system)
        system_value.update(system_delta(system, self.system_baseline))
        system_value["compressor_bytes"] = system.compressor_pages * system.page_size
        system_value["swapped_bytes"] = system.swapped_pages * system.page_size
        self.writer.write({
            "type": "sample",
            "sample_index": self.sample_count,
            "mono_ns": mono_ns,
            "wall_time_ns": wall_ns,
            "gap_ns": gap_ns,
            "gap_exceeded": gap_ns > self.max_gap_ns,
            "correlation_id": self.correlation_id,
            "root_alive": self.last_root_alive,
            "processes": processes,
            "tree_rss_bytes": tree_rss,
            "tree_phys_footprint_bytes": tree_phys,
            "system": system_value,
            "runtime": phase,
        }, sync=True)
        self.sample_count += 1
        self.last_sample_ns = mono_ns
        return bool(current_identities)

    def finish(
        self, command_exit_code: int | None, *, stop_reason: str = "root_exit"
    ) -> dict[str, Any]:
        if self.system_baseline is None or self.system_last is None:
            raise SamplerError("sampler was not started")
        deltas = system_delta(self.system_last, self.system_baseline)
        reasons: list[str] = []
        if self.sample_count == 0:
            reasons.append("no_samples")
        if self.max_observed_gap_ns > self.max_gap_ns:
            reasons.append("sample_gap_exceeded")
        if self.unknown_children:
            reasons.append("unknown_child")
        if self.errors:
            reasons.append("sampler_error")
        if command_exit_code not in (None, 0):
            reasons.append("command_failed")
        complete = not reasons
        terminal = {
            "type": "terminal",
            "revision": self.backend.revision,
            "mono_ns": self.clock.monotonic_ns(),
            "wall_time_ns": self.clock.wall_time_ns(),
            "correlation_id": self.correlation_id,
            "terminal_sample": True,
            "stop_reason": stop_reason,
            "root_alive_at_stop": self.last_root_alive,
            "complete": complete,
            "status": "complete" if complete else "inconclusive",
            "reasons": reasons,
            "sample_count": self.sample_count,
            "process_count": len(self.known),
            "unknown_children": [
                asdict(identity) for identity in sorted(self.unknown_children)
            ],
            "sampler_errors": list(self.errors),
            "max_gap_ns": self.max_observed_gap_ns,
            "allowed_max_gap_ns": self.max_gap_ns,
            "tree_peak_rss_bytes": self.tree_peak_rss_bytes,
            "tree_peak_phys_footprint_bytes":
                self.tree_peak_phys_footprint_bytes,
            "tree_peak_rss_mono_ns": self.tree_peak_rss_mono_ns,
            "tree_peak_phys_footprint_mono_ns":
                self.tree_peak_phys_footprint_mono_ns,
            "swap_in_bytes": deltas["swap_in_bytes"],
            "swap_out_bytes": deltas["swap_out_bytes"],
            "compression_bytes": deltas["compression_bytes"],
            "decompression_bytes": deltas["decompression_bytes"],
            "command_exit_code": command_exit_code,
        }
        return self.writer.write(terminal, sync=True)

    def run(
        self,
        command: subprocess.Popen[bytes] | None = None,
        *,
        stop_event: threading.Event | None = None,
        ready_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        self.start()
        next_sample_ns = self.clock.monotonic_ns()
        stop_reason = "root_exit"
        try:
            alive = self.sample()
            if ready_event is not None:
                ready_event.set()
            while True:
                command_done = command is not None and command.poll() is not None
                if stop_event is not None and stop_event.is_set():
                    stop_reason = "external_stop"
                    break
                if not alive and (command is None or command_done):
                    break
                next_sample_ns += self.interval_ns
                now = self.clock.monotonic_ns()
                if next_sample_ns <= now:
                    next_sample_ns = now
                else:
                    self.clock.sleep((next_sample_ns - now) / 1_000_000_000)
                alive = self.sample()
        except (OSError, SamplerError) as exc:
            self.errors.append(str(exc))
            stop_reason = "sampler_error"
            if ready_event is not None:
                ready_event.set()
            if command is not None and command.poll() is None:
                command.terminate()
                try:
                    command.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    command.kill()
        exit_code = command.wait() if command is not None else None
        return self.finish(exit_code, stop_reason=stop_reason)


def launch_command(command: Sequence[str]) -> subprocess.Popen[bytes]:
    if not command:
        raise SamplerError("launch mode requires a command")
    return subprocess.Popen(list(command), start_new_session=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pid", type=int, help="attach to an existing root PID")
    source.add_argument(
        "--launch", action="store_true",
        help="launch the command following -- and sample it as the root",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--correlation-id", required=True)
    parser.add_argument("--root-role", default="turbocider-root")
    parser.add_argument(
        "--role", action="append", type=parse_role_rule, default=[],
        metavar="ROLE=GLOB",
        help="classify a descendant by process name or executable glob",
    )
    parser.add_argument("--interval-ms", type=int, default=DEFAULT_INTERVAL_MS)
    parser.add_argument("--max-gap-ms", type=int, default=DEFAULT_MAX_GAP_MS)
    parser.add_argument("--phase-file", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command_args = list(args.command)
    if command_args and command_args[0] == "--":
        command_args.pop(0)
    if args.launch != bool(command_args):
        raise SamplerError("--launch requires a command after --; --pid forbids one")
    command: subprocess.Popen[bytes] | None = None
    writer: HashChainJsonlWriter | None = None
    try:
        if args.launch:
            command = launch_command(command_args)
            root_pid = command.pid
        else:
            root_pid = args.pid
        backend = DarwinBackend()
        writer = HashChainJsonlWriter(args.output)
        sampler = ProcessTreeSampler(
            backend, RealClock(), writer, root_pid,
            root_role=args.root_role,
            role_rules=args.role,
            interval_ns=args.interval_ms * 1_000_000,
            max_gap_ns=args.max_gap_ms * 1_000_000,
            correlation_id=args.correlation_id,
            phase_file=args.phase_file,
        )

        interrupted_signal: int | None = None

        def forward(signum: int, _frame: object) -> None:
            nonlocal interrupted_signal
            interrupted_signal = signum
            if command is not None and command.poll() is None:
                command.send_signal(signum)

        old_term = signal.signal(signal.SIGTERM, forward)
        old_int = signal.signal(signal.SIGINT, forward)
        try:
            terminal = sampler.run(command)
        finally:
            signal.signal(signal.SIGTERM, old_term)
            signal.signal(signal.SIGINT, old_int)
        if interrupted_signal is not None:
            return 128 + interrupted_signal
        if command is not None and terminal["command_exit_code"] not in (None, 0):
            return EXIT_INCONCLUSIVE
        return 0 if terminal["complete"] else EXIT_INCONCLUSIVE
    except (OSError, SamplerError) as exc:
        if writer is not None:
            writer.write({
                "type": "sampler_error",
                "mono_ns": time.monotonic_ns(),
                "wall_time_ns": time.time_ns(),
                "correlation_id": args.correlation_id,
                "error": str(exc),
            }, sync=True)
        if command is not None and command.poll() is None:
            command.terminate()
            try:
                command.wait(timeout=5)
            except subprocess.TimeoutExpired:
                command.kill()
                command.wait()
        print(f"process-tree sampler failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
