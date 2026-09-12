"""Disk inventory must not take the GPU lease or survive an unresponsive read."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / 'build/native/turbocider'

class InventoryTests(unittest.TestCase):
    def test_inventory_without_gpu_lease(self):
        with tempfile.TemporaryDirectory(prefix='tc-inventory-') as folder:
            root = Path(folder)
            artifacts = {}
            for i in range(20):
                name = f'block{i}.mlpackage'
                (root / name).mkdir()
                artifacts[str(i)] = {'int8_pc': name}
            source = root / 'manifest.json'
            source.write_text(json.dumps({'artifacts': artifacts}))
            request = root / 'request.json'
            request.write_text(json.dumps({'action': 'inventory', 'cache': str(root/'cache'), 'source_manifest': str(source)}))
            lock = Path(tempfile.gettempdir()) / f'turbocider-gpu-{os.geteuid()}.lock'
            with lock.open('a') as lease:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = subprocess.run([str(CLI), 'coreml', str(request)], capture_output=True, text=True, timeout=8)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)['action'], 'inventory')

    def test_invalid_source_is_reported(self):
        with tempfile.TemporaryDirectory(prefix='tc-inventory-blocked-') as folder:
            root = Path(folder)
            source = root / 'blocked.json'
            os.mkfifo(source)
            request = root / 'request.json'
            request.write_text(json.dumps({'action': 'inventory', 'cache': str(root/'cache'), 'source_manifest': str(source)}))
            result = subprocess.run([str(CLI), 'coreml', str(request)], capture_output=True, timeout=8)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(any('error' in item for item in json.loads(result.stdout)['entries']))

if __name__ == '__main__':
    unittest.main(verbosity=2)
