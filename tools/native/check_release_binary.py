#!/usr/bin/env python3
"""Require a public-only, non-instrumented library for stable App packaging."""
import argparse
import json
from pathlib import Path
import subprocess


def check(library: Path, manifest: Path) -> None:
    build = json.loads(manifest.read_text())
    policy = build['inputs']['policy']
    if policy['test_hooks'] != '0' or policy['audit_counters'] != '0':
        raise ValueError('release packaging requires test hooks and audit counters disabled')
    flags = policy.get('common_flags', [])
    if not isinstance(flags, list) or any(not isinstance(flag, str) for flag in flags):
        raise ValueError('release library has invalid build flags')
    if any(flag.startswith('-DTURBOCIDER_ENABLE_PRIVATE_ANE') and
           flag != '-DTURBOCIDER_ENABLE_PRIVATE_ANE=0' for flag in flags):
        raise ValueError('stable release packaging requires private ANE disabled; use a public-only build')
    symbols = subprocess.check_output(['nm', '-gU', str(library)], text=True)
    for line in symbols.splitlines():
        symbol = line.split()[-1] if line.split() else ''
        if symbol.startswith(('_tc_engine_test_', '_tc_streaming_audit_')):
            raise ValueError('release library exposes instrumentation: ' + symbol)
    # Audit actual bytes/links as well as the claimed build policy. Private
    # classes are dynamically loaded, so nm alone cannot detect them. Do this
    # BEFORE package.sh can replace an existing App or distributable CLI.
    strings = subprocess.check_output(['strings', str(library)], text=True)
    for name in ('_ANEClient', '_ANERequest', '_ANEIOSurfaceObject', '_ANESharedEvents', '_ANEInMemoryModel'):
        if name in strings:
            raise ValueError('stable release library contains private ANE class: ' + name)
    links = subprocess.check_output(['otool', '-L', str(library)], text=True)
    if '/PrivateFrameworks/' in links:
        raise ValueError('stable release library links a private framework')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    args = parser.parse_args()
    try:
        check(args.library, args.manifest)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'cannot package native library: {error}\n')
