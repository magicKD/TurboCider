"""Continuous process-load evidence, not proof of GPU/ANE exclusivity.

Only read process state. Exclude the explicitly owned launch tree, retain no
arguments/prompts, and never signal a competing process.
"""
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from runtime_ane_common import sha256_file


def competing_processes(root_pid):
    snapshot = subprocess.check_output(['ps', '-Ao', 'pid=,ppid=,pcpu=,comm='], text=True, timeout=2)
    arguments = subprocess.check_output(['ps', '-Ao', 'pid=,args='], text=True, timeout=2)
    args = {}
    for line in arguments.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) == 2:
            args[int(fields[0])] = fields[1]
    processes = []
    for line in snapshot.splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) == 4:
            processes.append((int(fields[0]), int(fields[1]), float(fields[2]), fields[3]))
    owned = {root_pid, os.getpid()}
    while True:
        descendants = {pid for pid, parent, _, _ in processes if parent in owned}
        if descendants <= owned:
            break
        owned.update(descendants)
    busy = []
    for pid, _, cpu, command in processes:
        workload = (command + ' ' + args.get(pid, '')).lower()
        if pid not in owned and cpu > 5 and any(token in workload for token in
                ('comfyui', 'turbocider', 'ane-runtime-probe', 'llama-server', 'mlx_lm', 'download_ltx25.py')):
            busy.append({'pid': pid, 'cpu_percent': cpu, 'executable': command})
    return busy


class LoadObservation:
    def __init__(self, path, *, interval_ms=500, max_gap_ms=2500):
        if interval_ms <= 0 or max_gap_ms < interval_ms:
            raise ValueError('invalid load observation interval/gap')
        self.path = Path(path)
        self.interval_ms, self.max_gap_ms = interval_ms, max_gap_ms
        self.rows, self.errors = [], []
        self.stop = threading.Event()
        self.thread = None
        self.root_pid = None

    def _capture(self):
        try:
            busy = competing_processes(self.root_pid)
            row = {'monotonic_ns': time.monotonic_ns(), 'competing_processes': busy}
            self.file.write(json.dumps(row) + '\n')
            self.file.flush()
            self.rows.append(row)
        except Exception as error:
            self.errors.append(str(error))

    def start(self, root_pid):
        self.root_pid = root_pid
        self.start_ns = time.monotonic_ns()
        self.file = self.path.open('x')
        self._capture()

        def sample():
            while not self.stop.wait(self.interval_ms / 1000):
                self._capture()
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()

    def finish(self):
        self.end_ns = time.monotonic_ns()
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                self.errors.append('load observer did not stop within deadline')
                return
        if hasattr(self, 'file'):
            self._capture()
            self.file.close()

    def verify(self):
        if self.root_pid is None or not hasattr(self, 'end_ns') or self.errors or len(self.rows) < 2:
            raise ValueError('load observation incomplete: ' + '; '.join(self.errors))
        if [json.loads(line) for line in self.path.read_text().splitlines()] != self.rows:
            raise ValueError('load evidence changed after observation')
        points = sorted([self.start_ns, self.end_ns, *[r['monotonic_ns'] for r in self.rows]])
        gap = max(b-a for a, b in zip(points, points[1:]))
        if gap > self.max_gap_ms * 1_000_000:
            raise ValueError('load observation gap exceeded')
        if any(r['competing_processes'] for r in self.rows):
            raise ValueError('competing inference CPU load observed; comparison is not qualified')
        return {'complete': True, 'heuristic_only': True, 'gpu_exclusivity_proven': False,
                'scope': 'continuous process CPU load; owned launch tree excluded; no arguments retained',
                'interval_ms': self.interval_ms, 'max_gap_ms': self.max_gap_ms,
                'observed_max_gap_ns': gap, 'samples': len(self.rows),
                'evidence': self.path.name, 'evidence_sha256': sha256_file(self.path)}
