import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/validation'))
import runtime_ane_load as load
from runtime_ane_memory import run_owned


class LoadTests(unittest.TestCase):
    def test_owned_tree_excluded_and_external_arguments_not_retained(self):
        snapshot=f'{os.getpid()} 1 90 python\n10 {os.getpid()} 80 turbocider\n11 10 80 turbocider\n20 1 50 python\n'
        args='20 python ComfyUI/main.py --secret private-value\n'
        with mock.patch.object(load.subprocess,'check_output',side_effect=[snapshot,args]):
            result=load.competing_processes(10)
        self.assertEqual(result,[{'pid':20,'cpu_percent':50.,'executable':'python'}])
        self.assertNotIn('private-value',json.dumps(result))

    def observed(self, directory):
        observer=load.LoadObservation(Path(directory)/'load.jsonl',interval_ms=10,max_gap_ms=500)
        observer.start(os.getpid())
        observer.finish()
        return observer

    def test_complete_quiet_evidence_is_heuristic_and_byte_checked(self):
        with tempfile.TemporaryDirectory() as directory,mock.patch.object(load,'competing_processes',return_value=[]):
            observer=self.observed(directory)
            receipt=observer.verify()
            self.assertTrue(receipt['complete'])
            self.assertFalse(receipt['gpu_exclusivity_proven'])
            self.assertEqual(receipt['samples'],2)
            observer.path.write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'changed'):observer.verify()

    def test_competing_load_errors_and_gaps_reject_without_killing_external_jobs(self):
        for result in ([{'pid':20,'cpu_percent':50.,'executable':'ComfyUI'}],RuntimeError('ps failed')):
            with tempfile.TemporaryDirectory() as directory,mock.patch.object(load,'competing_processes') as inspect:
                if isinstance(result,Exception):inspect.side_effect=result
                else:inspect.return_value=result
                observer=self.observed(directory)
                with self.assertRaises(ValueError):observer.verify()
                self.assertTrue(observer.path.exists())
        with tempfile.TemporaryDirectory() as directory,mock.patch.object(load,'competing_processes',return_value=[]):
            observer=self.observed(directory)
            observer.start_ns-=1_000_000_000
            with self.assertRaisesRegex(ValueError,'gap'):observer.verify()

    def test_owned_runner_closes_observer_on_success_and_timeout(self):
        for timed in (False,True):
            with tempfile.TemporaryDirectory() as directory,mock.patch.object(load,'competing_processes',return_value=[]):
                observer=load.LoadObservation(Path(directory)/'load.jsonl',interval_ms=10,max_gap_ms=1000)
                command=[sys.executable,'-c','import time; time.sleep(2)' if timed else 'pass']
                if timed:
                    with self.assertRaises(load.subprocess.TimeoutExpired):
                        run_owned(command,timeout=.05,observer=observer)
                else:
                    self.assertEqual(run_owned(command,timeout=10,observer=observer).returncode,0)
                self.assertFalse(observer.thread.is_alive())
                self.assertTrue(observer.file.closed)


if __name__=='__main__':unittest.main(verbosity=2)
