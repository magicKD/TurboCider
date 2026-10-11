import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/native"))
from process_tree_sampler import AsyncHashChainJsonlWriter, ProcessTreeSampler, SamplerError
from verify_process_tree_samples import EvidenceError, verify
from test_process_tree_sampler import FakeBackend, FakeClock, process, system


class AsyncEvidenceTests(unittest.TestCase):
    def test_blocked_fsync_does_not_block_sampling_and_exact_chain_drains(self):
        entered, release = threading.Event(), threading.Event()
        original = os.fsync
        def stalled(fd):
            if not entered.is_set():
                entered.set()
                if not release.wait(2):
                    raise OSError("fixture writer was not released")
            return original(fd)
        backend = FakeBackend()
        backend.states = [{1: process(1, 0, "root", 100, 140)}] * 5
        backend.systems = [system()] * 5
        clock = FakeClock()
        with tempfile.TemporaryDirectory(prefix="tc-async-evidence-") as raw, mock.patch("os.fsync", side_effect=stalled):
            path = Path(raw) / "samples.jsonl"
            writer = AsyncHashChainJsonlWriter(path)
            sampler = ProcessTreeSampler(backend, clock, writer, 1, root_role="root", role_rules=[],
                interval_ns=100_000_000, max_gap_ns=500_000_000, correlation_id="async-fixture")
            try:
                sampler.start()
                self.assertTrue(entered.wait(1))
                for _ in range(3):
                    sampler.sample()
                    clock.sleep(.1)
                terminal = sampler.finish(0, stop_reason="external_stop")
                self.assertIs(terminal["complete"], True)
                self.assertTrue(writer.worker.is_alive())
                self.assertGreater(writer.pending.qsize(), 0)
            finally:
                release.set()
                writer.close()
            checked = verify(path)
            self.assertIs(checked["complete"], True)
            self.assertEqual(checked["sample_count"], 3)
            self.assertEqual(checked["max_gap_ns"], 100_000_000)
            self.assertEqual(checked["allowed_max_gap_ns"], 500_000_000)
            self.assertEqual(checked["evidence_writer_policy"], writer.policy)
            self.assertEqual(checked["evidence_queue_capacity"], 64)

    def test_writer_failure_cannot_publish_success(self):
        with tempfile.TemporaryDirectory(prefix="tc-async-write-error-") as raw:
            path = Path(raw) / "samples.jsonl"
            writer = AsyncHashChainJsonlWriter(path)
            with mock.patch.object(writer, "_emit", side_effect=OSError("write failed")):
                writer.write(dict(type="sampler_start"), sync=True)
                writer.worker.join(timeout=1)
                self.assertFalse(writer.worker.is_alive())
                with self.assertRaisesRegex(SamplerError, "write failed"):
                    writer.close()
            with self.assertRaises(EvidenceError):
                verify(path)

    def test_queue_is_bounded_and_overflow_is_not_dropped_silently(self):
        entered, release = threading.Event(), threading.Event()
        def held(*_):
            entered.set()
            if not release.wait(2):
                raise OSError("fixture wait failed")
        with tempfile.TemporaryDirectory(prefix="tc-async-backpressure-") as raw:
            path = Path(raw) / "samples.jsonl"
            writer = AsyncHashChainJsonlWriter(path)
            with mock.patch.object(writer, "_emit", side_effect=held):
                writer.write(dict(type="sampler_start"))
                self.assertTrue(entered.wait(1))
                try:
                    for _ in range(64):
                        writer.write(dict(type="sample"))
                    self.assertEqual(writer.pending.qsize(), 64)
                    with self.assertRaisesRegex(SamplerError, "queue exhausted"):
                        writer.write(dict(type="sample"))
                finally:
                    release.set()
                    writer.close()
            with self.assertRaises(EvidenceError):
                verify(path)


if __name__ == "__main__":
    unittest.main()
