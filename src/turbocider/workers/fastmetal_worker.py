#!/usr/bin/env python3
"""Stream FastMetal output and expose its metrics to TurboCider jobs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: fastmetal_worker.py SCRIPT [FASTMETAL_ARGS...]")
    script = Path(sys.argv[1]).expanduser().resolve()
    engine_args = sys.argv[2:]
    try:
        metrics_index = engine_args.index("--metrics-json") + 1
        metrics_path = Path(engine_args[metrics_index]).expanduser().resolve()
    except (ValueError, IndexError) as error:
        raise SystemExit("FastMetal worker requires --metrics-json PATH") from error

    process = subprocess.Popen(
        [sys.executable, "-u", str(script), *engine_args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        raise SystemExit(return_code)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    result = dict(metrics)
    result["wall_seconds"] = float(metrics["total_s"])
    print("turbocider_result=" + json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
