#!/usr/bin/env python3
"""Deterministic contract tests for the process-tree memory sampler."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))

from process_tree_sampler import (  # noqa: E402
    HashChainJsonlWriter,
    ProcessIdentity,
    ProcessSnapshot,
    ProcessTreeSampler,
    SystemSnapshot,
)
from verify_process_tree_samples import verify  # noqa: E402


class FakeClock:
    def __init__(self):
        self.mono = 1_000_000_000
        self.wall = 2_000_000_000
        self.sleeps: list[float] = []

    def monotonic_ns(self) -> int:
        return self.mono

    def wall_time_ns(self) -> int:
        return self.wall

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        step = max(1, int(seconds * 1_000_000_000))
        self.mono += step
        self.wall += step


class FakeBackend:
    revision = "fake-sampler-v1"

    def __init__(self):
        self.tick = 0
        self.states: list[dict[int, ProcessSnapshot]] = []
        self.systems: list[SystemSnapshot] = []

    def capabilities(self):
        return {
            "process_resident_bytes": True,
            "process_phys_footprint_bytes": True,
            "process_compressed_bytes": False,
            "system_swap_counters": True,
            "gpu_driver_bytes": False,
        }

    def process(self, pid: int):
        state = self.states[min(self.tick, len(self.states) - 1)]
        return state.get(pid)

    def children(self, pid: int):
        state = self.states[min(self.tick, len(self.states) - 1)]
        return sorted(snapshot.identity.pid for snapshot in state.values()
                      if snapshot.ppid == pid)

    def system(self):
        value = self.systems[min(self.tick, len(self.systems) - 1)]
        self.tick += 1
        return value


def process(pid: int, ppid: int, name: str, rss: int, footprint: int):
    return ProcessSnapshot(
        ProcessIdentity(pid, 10, pid), ppid, name, f"/bin/{name}",
        rss, footprint, footprint, 0, 0, 0, 0,
    )


def system(swapouts: int = 0, compressor: int = 0):
    return SystemSnapshot(4096, 100, 20, 30, 10, 4, swapouts,
                          compressor, compressor // 2)


class SamplerTests(unittest.TestCase):
    def make_sampler(self, backend: FakeBackend, clock: FakeClock, path: Path,
                     *, roles=("worker=worker",)):
        writer = HashChainJsonlWriter(path)
        sampler = ProcessTreeSampler(
            backend, clock, writer, 1, root_role="root",
            role_rules=[
                __import__("process_tree_sampler").RoleRule(*rule.split("=", 1))
                for rule in roles
            ], interval_ns=20_000_000, max_gap_ns=100_000_000,
            correlation_id="test-correlation",
        )
        return sampler, writer

    def test_recursive_spawn_exit_peak_and_hash_chain(self):
        backend = FakeBackend()
        root = process(1, 0, "root", 100, 140)
        worker = process(2, 1, "worker", 200, 300)
        backend.states = [
            {1: root}, {1: root}, {1: root, 2: worker}, {1: root},
        ]
        backend.systems = [system(0), system(0), system(2), system(3)]
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as raw:
            sampler, writer = self.make_sampler(backend, clock, Path(raw) / "samples.jsonl")
            sampler.start()
            self.assertTrue(sampler.sample())
            self.assertTrue(sampler.sample())
            self.assertTrue(sampler.sample())
            terminal = sampler.finish(0, stop_reason="external_stop")
            writer.close()
            rows = [json.loads(line) for line in (Path(raw) / "samples.jsonl").read_text().splitlines()]
            checked = verify(Path(raw) / "samples.jsonl")
        self.assertEqual(terminal["status"], "complete")
        self.assertEqual(checked["status"], "complete")
        self.assertEqual(terminal["sample_count"], 3)
        self.assertEqual(terminal["tree_peak_phys_footprint_bytes"], 440)
        self.assertEqual(terminal["swap_out_bytes"], 3 * 4096)
        self.assertEqual(rows[-1]["type"], "terminal")
        previous = "0" * 64
        for index, row in enumerate(rows):
            self.assertEqual(row["sequence"], index)
            self.assertEqual(row["previous_digest"], previous)
            digest_input = dict(row)
            actual = digest_input.pop("record_digest")
            import hashlib
            self.assertEqual(
                hashlib.sha256(json.dumps(
                    digest_input, sort_keys=True, separators=(",", ":"),
                    ensure_ascii=False).encode()).hexdigest(), actual
            )
            previous = actual

    def test_gap_and_unknown_child_are_inconclusive(self):
        backend = FakeBackend()
        backend.states = [
            {1: process(1, 0, "root", 100, 100)},
            {1: process(1, 0, "root", 100, 100),
             3: process(3, 1, "unclassified", 10, 10)},
            {1: process(1, 0, "root", 100, 100)},
        ]
        backend.systems = [system(), system(), system()]
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as raw:
            sampler, writer = self.make_sampler(
                backend, clock, Path(raw) / "samples.jsonl", roles=()
            )
            sampler.start()
            sampler.sample()
            clock.mono += 200_000_000
            sampler.sample()
            terminal = sampler.finish(0, stop_reason="external_stop")
            writer.close()
            checked = verify(Path(raw) / "samples.jsonl")
        self.assertEqual(terminal["status"], "inconclusive")
        self.assertEqual(checked["status"], "inconclusive")
        self.assertIn("sample_gap_exceeded", terminal["reasons"])
        self.assertIn("unknown_child", terminal["reasons"])

    def test_nonzero_command_exit_is_inconclusive(self):
        backend = FakeBackend()
        backend.states = [
            {1: process(1, 0, "root", 100, 100)},
            {1: process(1, 0, "root", 100, 100)},
        ]
        backend.systems = [system(), system()]
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as raw:
            sampler, writer = self.make_sampler(backend, clock, Path(raw) / "samples.jsonl")
            sampler.start()
            sampler.sample()
            terminal = sampler.finish(17, stop_reason="external_stop")
            writer.close()
            checked = verify(Path(raw) / "samples.jsonl")
        self.assertEqual(terminal["status"], "inconclusive")
        self.assertEqual(checked["status"], "inconclusive")
        self.assertIn("command_failed", terminal["reasons"])


if __name__ == "__main__":
    unittest.main()
