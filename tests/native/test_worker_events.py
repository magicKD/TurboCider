#!/usr/bin/env python3
"""Emitter resource bounds, no model or GPU execution."""
import json
import subprocess
import tempfile
from pathlib import Path
root = Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory() as tmp:
    binary = Path(tmp) / 'events'
    subprocess.run(['clang++', '-std=c++20', '-fobjc-arc', '-Wall', '-Wextra', '-Werror',
                    str(root / 'tests/native/worker_events_test.mm'), '-framework', 'Foundation', '-o', str(binary)], check=True)
    result = subprocess.run([str(binary)], capture_output=True, check=True, timeout=10)
    assert not result.stdout and 500 * 1024 < len(result.stderr) <= 512 * 1024
    lines = result.stderr.splitlines()
    assert all(line.startswith(b'TC_EVENT\t') and len(line) <= 128 * 1024 + 9 for line in lines)
    events = [json.loads(line[9:]) for line in lines]
    assert events[0]['kind'] == 'resolved'
    assert all(e['kind'] == 'progress' for e in events[1:])
    assert [e['sequence'] for e in events] == list(range(1, len(events) + 1))
    result = subprocess.run([str(binary), 'oversized'], capture_output=True, check=True, timeout=10)
    assert not result.stdout and not result.stderr, 'Progress emitted without its oversized resolved event'
    print('PASS: lifetime 512 KiB limit, bounded frames, monotonic ordering, oversized resolved suppresses dependent telemetry')
