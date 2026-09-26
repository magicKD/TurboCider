#!/usr/bin/env python3
"""Real CLI startup barrier, no model required."""
import os
import subprocess
import tempfile
import time
from pathlib import Path

root = Path(__file__).resolve().parents[2]
cli = Path(os.environ.get('TURBOCIDER_TEST_NATIVE_DIR', root / 'build/native')) / 'turbocider'
with tempfile.TemporaryDirectory(prefix='tc-admission-') as directory:
    missing = str(Path(directory) / 'absent.json')
    for mode in ['worker-query', 'worker-generate']:
        for token in [b'', b'\0', b'\1']:
            proc = subprocess.run([str(cli), mode, missing, '--supervised'], input=token,
                                  capture_output=True, timeout=5)
            expected = b'worker_input_open_failed' if token == b'\1' else b'worker_start_not_admitted'
            assert proc.returncode == 1 and not proc.stdout and expected in proc.stderr
        proc = subprocess.Popen([str(cli), mode, missing, '--supervised'], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            time.sleep(0.25)
            assert proc.poll() is None, 'worker ran before parent admission'
            proc.stdin.close()
            proc.stdin = None
            stdout, stderr = proc.communicate(timeout=5)
            assert proc.returncode == 1 and not stdout and b'worker_start_not_admitted' in stderr
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        print(mode, 'startup admission/EOF PASS')
