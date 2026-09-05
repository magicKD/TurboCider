"""Native-engine installation bootstrap."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, Optional

from turbocider.paths import engine_root


ENGINE_MAPPINGS = {
    "h3": ["h3.c"],
    "ltx-mac": ["ltx-mac"],
    "flux2/engine": ["gpu_ane/flux2-engine"],
}


def bootstrap_engines(
    source: Path,
    *,
    target: Optional[Path] = None,
    copy: bool = False,
    engines: Iterable[str] = (),
) -> dict:
    """Link or copy native engine directories into the self-contained root."""
    source = source.expanduser().resolve()
    target = (target or engine_root()).expanduser().resolve()
    selected = set(engines)
    unknown = sorted(selected - set(ENGINE_MAPPINGS))
    if unknown:
        raise ValueError("unknown engine(s): %s" % ", ".join(unknown))

    report: dict = {"target": str(target)}
    for relative, candidates in ENGINE_MAPPINGS.items():
        if selected and relative not in selected:
            continue
        present = next(
            (item for item in (source / name for name in candidates) if item.is_dir()),
            None,
        )
        if present is None:
            report[relative] = "missing"
            continue
        destination = target / relative
        if destination.exists() or destination.is_symlink():
            report[relative] = "present"
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if copy:
            shutil.copytree(present, destination, symlinks=True)
            report[relative] = "copied"
        else:
            destination.symlink_to(present, target_is_directory=True)
            report[relative] = "linked"
    return report


def print_bootstrap_report(report: dict) -> None:
    print(json.dumps(report, indent=2, sort_keys=True))
