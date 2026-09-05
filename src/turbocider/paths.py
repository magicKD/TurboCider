"""Repository and state path helpers."""

from __future__ import annotations

import os
import sys
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
    """Return the TurboCider project root used by the legacy ${WORKSPACE} token.

    The repository is a standalone project. By default this is the package
    root itself, not its parent directory. A developer working inside the
    larger video-generation monorepo can still opt into the monorepo layout
    by setting ``TURBOCIDER_WORKSPACE`` or writing a local ``workspace.path``.
    """
    configured = os.environ.get("TURBOCIDER_WORKSPACE")
    if configured:
        return Path(configured).expanduser().resolve()
    pointer = PACKAGE_ROOT / "workspace.path"
    if pointer.is_file():
        value = pointer.read_text(encoding="utf-8").strip()
        if value:
            return Path(value).expanduser().resolve()
    return PACKAGE_ROOT.resolve()


WORKSPACE_ROOT = workspace_root()


def engine_root() -> Path:
    """Return the self-contained native-engine installation directory.

    Engine executables, conversion tools, and per-engine Python runtimes are
    installed below this directory. The default is ``PACKAGE_ROOT/engines`` so
    a fresh checkout never depends on a sibling monorepo. Set
    ``TURBOCIDER_ENGINES_DIR`` to use an external engine installation.
    """
    configured = os.environ.get("TURBOCIDER_ENGINES_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (PACKAGE_ROOT / "engines").resolve()


ENGINE_ROOT = engine_root()


def data_root() -> Path:
    """Return the directory containing installed model packs and profiles.

    In a source checkout these files live at the repository root. A wheel
    installs them below ``sys.prefix/share/turbocider``; prefer that location
    when the source layout is not present so ``turbocider`` works after a
    regular ``pip install .``.
    """
    if (PACKAGE_ROOT / "model-packs").is_dir() and (
        PACKAGE_ROOT / "device-profiles"
    ).is_dir():
        return PACKAGE_ROOT
    installed = Path(sys.prefix) / "share" / "turbocider"
    if (installed / "model-packs").is_dir():
        return installed
    return PACKAGE_ROOT


DATA_ROOT = data_root()


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
    expanded = expanded.replace("${TURBOCIDER_ENGINES}", str(engine_root()))
    expanded = os.path.expandvars(os.path.expanduser(expanded))
    return Path(expanded).resolve()
