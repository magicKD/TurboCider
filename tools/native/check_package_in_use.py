#!/usr/bin/env python3
"""Refuse to replace binaries still mapped by a running App or CLI worker."""
import argparse
import ctypes
from pathlib import Path
import subprocess
import sys


def running_processes() -> str:
    listing = subprocess.check_output(['ps', '-axww', '-o', 'pid=,comm='], text=True)
    if sys.platform != 'darwin':
        return listing
    # ps preserves relative argv[0] paths. libproc identifies the executable
    # independently, including a process still mapping an unlinked old file.
    libproc = ctypes.CDLL('/usr/lib/libproc.dylib')
    libproc.proc_pidpath.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    libproc.proc_pidpath.restype = ctypes.c_int
    paths = []
    for line in listing.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        path = ctypes.create_string_buffer(4096)
        if libproc.proc_pidpath(int(fields[0]), path, len(path)) > 0:
            paths.append(fields[0] + ' ' + path.value.decode('utf-8', errors='surrogateescape'))
        else:
            paths.append(line)
    return '\n'.join(paths)


def package_processes(app: Path, cli: Path, processes: str) -> list[tuple[int, str]]:
    roots = (app.resolve(), cli.resolve())
    matches = []
    for line in processes.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        pid, executable = int(fields[0]), fields[1]
        path = Path(executable)
        if path.is_absolute() and any(root in path.resolve().parents for root in roots):
            matches.append((pid, executable))
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app', required=True, type=Path)
    parser.add_argument('--cli', required=True, type=Path)
    args = parser.parse_args()
    try:
        processes = running_processes()
        matches = package_processes(args.app, args.cli, processes)
    except (OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'cannot check running package processes: {error}\n')
    if matches:
        details = '\n'.join(f'  PID {pid}: {path}' for pid, path in matches)
        parser.exit(1, 'cannot replace a running TurboCider package. Quit the App and wait for its workers '
                    'to exit, then package again; replacing files leaves the process running old code.\n'
                    + details + '\n')


if __name__ == '__main__':
    main()
