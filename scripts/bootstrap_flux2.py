#!/usr/bin/env python3
"""Create the reproducible FLUX.2 runtime used by TurboCider on Apple Silicon."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


MFLUX_VERSION = "0.19.1"
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
MODEL_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"


def run(command, *, cwd: Path, environment=None) -> None:
    print("+ " + " ".join(str(item) for item in command), flush=True)
    subprocess.run(
        [str(item) for item in command],
        cwd=str(cwd),
        env=environment,
        check=True,
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python 3.10+ interpreter used to create the MLX environment",
    )
    root.add_argument("--skip-model", action="store_true")
    root.add_argument("--skip-ane", action="store_true")
    root.add_argument("--overwrite-ane", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    package = Path(__file__).resolve().parents[1]
    workspace = package.parent
    gpu_root = workspace / "gpu_ane"
    engine = gpu_root / "flux2-engine"
    research = gpu_root / "mac_local_ai"
    runtime = gpu_root / "flux2-runtime"
    mflux_root = gpu_root / "mflux-runtime"
    model = runtime / "models" / "FLUX.2-klein-4B"
    manifest = runtime / "models" / "coreml" / "flux2_stack_runtime_m1088" / "manifest.json"
    for required in (engine, research):
        if not required.is_dir():
            raise SystemExit("required FLUX source directory is missing: %s" % required)

    environment = os.environ.copy()
    environment.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    venv = runtime / ".venv"
    if not (venv / "bin" / "python").is_file():
        run([args.python, "-m", "venv", venv], cwd=workspace)
    python = venv / "bin" / "python"
    run(
        [python, "-m", "pip", "install", "-r", package / "requirements-flux2.txt"],
        cwd=workspace,
        environment=environment,
    )
    mflux_source = mflux_root / "src"
    mflux_source.mkdir(parents=True, exist_ok=True)
    run(
        [
            python, "-m", "pip", "install", "--upgrade", "--no-deps",
            "--target", mflux_source, "mflux==" + MFLUX_VERSION,
        ],
        cwd=workspace,
        environment=environment,
    )

    if not args.skip_model:
        script = (
            "from huggingface_hub import snapshot_download; "
            "snapshot_download(repo_id=%r, revision=%r, local_dir=%r, "
            "allow_patterns=%r, max_workers=8)"
            % (
                MODEL_ID,
                MODEL_REVISION,
                str(model),
                [
                    "vae/*.safetensors", "vae/*.json",
                    "transformer/*.safetensors", "transformer/*.json",
                    "text_encoder/*.safetensors", "text_encoder/*.json",
                    "tokenizer/**", "added_tokens.json", "chat_template.jinja",
                ],
            )
        )
        run([python, "-c", script], cwd=workspace, environment=environment)

    run([python, engine / "scripts" / "build_native.py"], cwd=workspace)
    if not args.skip_ane:
        command = [
            python,
            research / "scripts" / "prepare_flux2_ane_stack.py",
            "--checkpoint", model,
            "--out-dir", manifest.parent,
            "--buckets", "1088",
            "--blocks", *(str(block) for block in range(20)),
            "--variants", "int8_pc",
        ]
        if args.overwrite_ane:
            command.append("--overwrite")
        if manifest.is_file() and not args.overwrite_ane:
            print("ANE manifest already exists: %s" % manifest)
        else:
            run(command, cwd=workspace, environment=environment)

    audit = {
        "format": "turbocider-flux2-runtime-v1",
        "mflux_version": MFLUX_VERSION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "python": str(python),
        "model": str(model),
        "mflux_root": str(mflux_root),
        "ane_manifest": str(manifest) if manifest.is_file() else None,
    }
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "runtime.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
