"""Capture process evidence for the encoder benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import time


def capture_encoder_probe(
    command: list[str], folder: Path, timeout: int, *, environment: dict[str, str],
) -> tuple[subprocess.CompletedProcess[str], dict]:
    started = time.perf_counter()
    process = subprocess.run(
        ["/usr/bin/time", "-l", *command], text=True, capture_output=True,
        env=environment, timeout=timeout, check=False,
    )
    wall_seconds = time.perf_counter() - started
    (folder / "stdout.log").write_text(process.stdout)
    (folder / "stderr.log").write_text(process.stderr)
    peak = re.search(r"(\d+)\s+maximum resident set size", process.stderr)
    record = {
        "command": command,
        "returncode": process.returncode,
        "process_wall_seconds": wall_seconds,
        "process_peak_rss_bytes": int(peak.group(1)) if peak else None,
    }
    (folder / "process.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return process, record
