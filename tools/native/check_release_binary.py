#!/usr/bin/env python3
"""Reject instrumented native libraries before creating a distributable App."""
import argparse
import json
from pathlib import Path
import subprocess


def check(library: Path, manifest: Path) -> None:
    build = json.loads(manifest.read_text())
    policy = build['inputs']['policy']
    if policy['test_hooks'] != '0' or policy['audit_counters'] != '0':
        raise ValueError('release packaging requires test hooks and audit counters disabled')
    symbols = subprocess.check_output(['nm', '-gU', str(library)], text=True)
    for line in symbols.splitlines():
        symbol = line.split()[-1] if line.split() else ''
        if symbol.startswith(('_tc_engine_test_', '_tc_streaming_audit_')):
            raise ValueError('release library exposes instrumentation: ' + symbol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    args = parser.parse_args()
    try:
        check(args.library, args.manifest)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'cannot package native library: {error}\n')
