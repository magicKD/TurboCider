#!/usr/bin/env python3
"""CPU fixtures for benchmark resume and evidence retention, no native execution."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/native'))
import benchmark_public_streaming_pair as benchmark


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.library = self.root/'library'; self.library.write_bytes(b'fixture')
        self.request = self.root/'request.json'; benchmark.write(self.request, {'fixture': True})
        self.output = self.root/'campaign'
        self.catalogs = []
        for name, d, q in [('baseline', 0, 1), ('candidate', 1, 2)]:
            stage = dict(resident_prefix_blocks=7, block_group_size=1, slot_count=2, prefetch_distance=d, io_workers=q)
            record = dict(source={}, workload={}, runtime={}, device={}, plan=dict(layout_digest=name,
                canonical_config=dict(stages=dict(denoiser=stage))))
            path = self.root/(name+'.json'); benchmark.write(path, {'records': [record]}); self.catalogs.append(path)
        self.argv = ['benchmark', '--library', str(self.library), '--model', str(self.root),
            '--request', str(self.request), '--baseline-catalog', str(self.catalogs[0]),
            '--candidate-catalog', str(self.catalogs[1]), '--output', str(self.output),
            '--pairs', '2', '--expected-image-sha256', 'c'*64]
        self.calls = 0

    def sample(self, command, **unused):
        self.calls += 1
        run = Path(command[-1]); run.mkdir()
        complete = self.calls != 1
        benchmark.write(run.parent/'memory-summary.json', dict(complete=complete,
            status='complete' if complete else 'inconclusive', command_exit_code=0,
            reasons=[] if complete else ['sample_gap_exceeded'], max_gap_ns=20 if complete else 200,
            tree_peak_phys_footprint_bytes=1000, swap_out_bytes=0))
        (run.parent/'memory.jsonl').write_text('retained raw evidence')
        benchmark.write(run/'observation.json', dict(status='succeeded', cleanup_returned=True,
            image_sha256='c'*64, wall_seconds=2))
        record = json.loads(Path(command[command.index('--catalog')+1]).read_text())['records'][0]
        actual = dict(record['plan']['canonical_config']['stages']['denoiser'], drained=True,
            source_lease_verified=True, group_count=23, pass_count=9, receipt=dict(fills=207))
        benchmark.write(run/'result.json', dict(result=dict(block_streaming=dict(actual_layout=actual, request_wait_seconds=0.1),
            timings_seconds=dict(request_wall=1, denoise=0.8), memory=dict(mlx_peak_bytes=900),
            public_streaming=dict(receipt_digest='fixture'))))
        return SimpleNamespace(returncode=0 if complete else 3)

    def invoke(self, *extra):
        with patch.object(sys, 'argv', self.argv+list(extra)), patch.object(benchmark.subprocess, 'run', side_effect=self.sample), patch('builtins.print'):
            benchmark.main()

    def test_resume_retains_inconclusive_without_repeating_or_overwriting(self):
        with self.assertRaisesRegex(RuntimeError, '01-baseline failed'):
            self.invoke()
        plan = (self.output/'plan.json').read_bytes()
        raw = (self.output/'01-baseline/memory-summary.json').read_bytes()
        self.invoke('--resume', '--retain-inconclusive-memory')
        summary = json.loads((self.output/'summary.json').read_text())
        self.assertEqual(self.calls, 4)
        self.assertEqual(summary['memory_qualification'], 'INCONCLUSIVE')
        self.assertEqual(len(summary['samples']), 4)
        self.assertEqual((self.output/'plan.json').read_bytes(), plan)
        self.assertEqual((self.output/'01-baseline/memory-summary.json').read_bytes(), raw)
        self.invoke('--resume', '--retain-inconclusive-memory')
        self.assertEqual(self.calls, 4)

    def test_changed_library_rejected_before_resume_launch(self):
        with self.assertRaises(RuntimeError): self.invoke()
        self.library.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed frozen'):
            self.invoke('--resume', '--retain-inconclusive-memory')
        self.assertEqual(self.calls, 1)

    def test_incomplete_existing_sample_is_not_retried(self):
        with self.assertRaises(RuntimeError): self.invoke()
        (self.output/'01-baseline/memory-summary.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.invoke('--resume', '--retain-inconclusive-memory')
        self.assertEqual(self.calls, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
