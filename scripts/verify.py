#!/usr/bin/env python3
"""Run reproducible TurboCider control-plane, engine, SDK, and app checks."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import json
import socket
from pathlib import Path


def run(command, *, cwd: Path, environment=None) -> None:
    print("+ " + " ".join(str(item) for item in command), flush=True)
    subprocess.run(
        [str(item) for item in command], cwd=str(cwd), env=environment, check=True
    )


def available_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def run_app_smoke(root: Path, environment) -> None:
    with tempfile.TemporaryDirectory(prefix="turbocider-app-smoke-") as temporary:
        smoke_root = Path(temporary)
        model = smoke_root / "model"
        model.mkdir()
        executable = smoke_root / "fake_h3.py"
        executable.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib, sys, time\n"
            "print('denoise 1/2', flush=True)\n"
            "time.sleep(.4)\n"
            "print('denoise 2/2', flush=True)\n"
            "time.sleep(.2)\n"
            "output = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])\n"
            "output.parent.mkdir(parents=True, exist_ok=True)\n"
            "output.write_bytes(b'turbocider-app-smoke')\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)
        packs = smoke_root / "model-packs"
        packs.mkdir()
        (packs / "app-smoke.json").write_text(
            json.dumps({
                "id": "app-smoke-h3",
                "name": "App Smoke H3",
                "engine": "h3",
                "version": "1",
                "capabilities": {
                    "tasks": ["video"],
                    "inputs": ["text"],
                    "audio_output": False,
                    "profiles": ["quality"],
                    "execution": ["gpu"],
                    "recommended_width": 32,
                    "recommended_height": 32,
                    "recommended_frames": 1,
                    "recommended_fps": 24,
                    "recommended_steps": 1,
                },
                "config": {
                    "executable_path": str(executable),
                    "model_path": str(model),
                },
                "plans": [{
                    "id": "app-smoke.gpu",
                    "execution": "gpu",
                    "quality": "exact",
                    "profile": "*",
                    "production": True,
                    "priority": 1,
                    "requirements": {
                        "config_paths": ["executable_path", "model_path"]
                    },
                }],
            }),
            encoding="utf-8",
        )
        smoke_environment = dict(environment)
        smoke_environment.update({
            "TURBOCIDER_API_URL": "http://127.0.0.1:%d" % available_loopback_port(),
            "TURBOCIDER_MODEL_PACKS": str(packs),
            "TURBOCIDER_STATE_DIR": str(smoke_root / "state"),
            "TURBOCIDER_OUTPUT_DIR": str(smoke_root / "outputs"),
            "TURBOCIDER_SMOKE_MODEL": "app-smoke-h3",
        })
        run(
            [root / ".build" / "debug" / "TurboCiderApp", "--smoke"],
            cwd=root,
            environment=smoke_environment,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="include real model doctor checks")
    parser.add_argument(
        "--bundle", action="store_true",
        help="build and exercise the signed macOS app bundle",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    workspace = root.parent
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=root,
        environment=environment,
    )
    flux_python = workspace / "gpu_ane" / "flux2-runtime" / ".venv" / "bin" / "python"
    if flux_python.is_file():
        flux_environment = os.environ.copy()
        flux_environment["PYTHONPATH"] = os.pathsep.join((
            str(workspace / "gpu_ane" / "flux2-engine" / "src"),
            str(workspace / "gpu_ane" / "mflux-runtime" / "src"),
        ))
        run(
            [flux_python, "-m", "pytest", "-q", workspace / "gpu_ane" / "flux2-engine" / "tests"],
            cwd=workspace,
            environment=flux_environment,
        )
    fastmetal_python = (
        workspace / "gpu_ane" / "fastmetal-runtime" / ".venv" / "bin" / "python"
    )
    fastmetal_engine = workspace / "gpu_ane" / "fastmetal-engine"
    if fastmetal_python.is_file() and fastmetal_engine.is_dir():
        run(
            [
                fastmetal_python,
                "-m", "pytest", "-q",
                "fastvideo/tests/mlx/test_mlx_fastmetal_ane.py",
                "fastvideo/tests/mlx/test_mlx_compile_parity.py",
            ],
            cwd=fastmetal_engine,
        )
    run(["swift", "build"], cwd=root)
    run([root / ".build" / "debug" / "turbocider-swift-selftest"], cwd=root)
    run_app_smoke(root, environment)
    if args.bundle:
        run(["sh", root / "scripts" / "build_app_bundle.sh"], cwd=root)
        bundled_root = (
            root / "dist" / "TurboCider.app" / "Contents" / "Resources" / "TurboCider"
        )
        required_bundle_files = (
            "scripts/prepare_model.py",
            "scripts/prepare_fastmetal_ane.py",
            "src/turbocider/workers/fastmetal_worker.py",
            "model-packs/fastmetal-1.3b-qad.json",
            "benchmarks/fastmetal-1.3b-qad-gpu.json",
            "benchmarks/fastmetal-1.3b-qad-gpu-ane.json",
        )
        missing_bundle_files = [
            relative for relative in required_bundle_files
            if not (bundled_root / relative).is_file()
        ]
        if missing_bundle_files:
            raise SystemExit(
                "app bundle is missing required files: "
                + ", ".join(missing_bundle_files)
            )
        bundled_environment = os.environ.copy()
        bundled_environment["PYTHONPATH"] = str(bundled_root / "src")
        run(
            [
                sys.executable, "-m", "turbocider.cli", "prepare-model",
                "flux2-klein-4b",
            ],
            cwd=bundled_root,
            environment=bundled_environment,
        )
        run(
            [
                sys.executable, "-m", "turbocider.cli", "prepare-model",
                "fastmetal-1.3b-qad",
            ],
            cwd=bundled_root,
            environment=bundled_environment,
        )
        bundle_cli = root / "dist" / "TurboCider.app" / "Contents" / "MacOS" / "turbocider-swift"
        with tempfile.TemporaryDirectory(prefix="turbocider-bundle-") as temporary:
            bundle_environment = os.environ.copy()
            bundle_environment["TURBOCIDER_STATE_DIR"] = str(Path(temporary) / "state")
            bundle_environment["TURBOCIDER_OUTPUT_DIR"] = str(Path(temporary) / "outputs")
            run(
                [bundle_cli, "--self-start"],
                cwd=Path(temporary),
                environment=bundle_environment,
            )
    if args.real:
        run(
            [sys.executable, "-m", "turbocider.cli", "doctor", "--json"],
            cwd=root,
            environment=environment,
        )


if __name__ == "__main__":
    main()
