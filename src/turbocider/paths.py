"""Repository and state path helpers."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _bundle_application_support() -> Path | None:
    if (
        PACKAGE_ROOT.name == "TurboCider"
        and PACKAGE_ROOT.parent.name == "Resources"
        and PACKAGE_ROOT.parent.parent.name == "Contents"
    ):
        return (Path.home() / "Library" / "Application Support" / "TurboCider").resolve()
    return None


def workspace_root() -> Path:
    configured = os.environ.get("TURBOCIDER_WORKSPACE")
    if configured:
        return Path(configured).expanduser().resolve()
    pointer = PACKAGE_ROOT / "workspace.path"
    if pointer.is_file():
        value = pointer.read_text(encoding="utf-8").strip()
        if value:
            return Path(value).expanduser().resolve()
    return PACKAGE_ROOT.parent.resolve()


WORKSPACE_ROOT = workspace_root()


def state_root() -> Path:
    configured = os.environ.get("TURBOCIDER_STATE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    application_support = _bundle_application_support()
    return ((application_support / "state") if application_support else (PACKAGE_ROOT / "state")).resolve()


def output_root() -> Path:
    configured = os.environ.get("TURBOCIDER_OUTPUT_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    application_support = _bundle_application_support()
    return ((application_support / "outputs") if application_support else (PACKAGE_ROOT / "outputs")).resolve()


def model_root() -> Path:
    configured = os.environ.get("TURBOCIDER_MODELS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    application_support = _bundle_application_support()
    return ((application_support / "models") if application_support else (PACKAGE_ROOT / "models")).resolve()


def expand_path(value: str) -> Path:
    expanded = value.replace("${TURBOCIDER}", str(PACKAGE_ROOT))
    expanded = expanded.replace("${WORKSPACE}", str(workspace_root()))
    expanded = expanded.replace("${TURBOCIDER_MODELS}", str(model_root()))
    expanded = os.path.expandvars(os.path.expanduser(expanded))
    return Path(expanded).resolve()
