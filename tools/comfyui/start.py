#!/usr/bin/env python3
"""Start a standalone TurboCider service and its loopback HTTP gateway.

No environment installation, weight download, registry mutation or model copy.
Ctrl-C stops both owned processes. State and generated artifacts are retained.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bindings/python"))
from turbocider_local import Client, TransportError


def main():
    candidates = [ROOT / "dist/cli/turbocider", ROOT.parent / "cli/turbocider", ROOT / "build/native/turbocider"]
    default_cli = next((p for p in candidates if p.is_file()), candidates[0])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, default=default_cli)
    parser.add_argument("--state-dir", type=Path, default=ROOT / "state/comfyui")
    parser.add_argument("--socket", default="/tmp/turbocider-comfyui.sock")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--comfyui", type=Path, help="Optionally launch an existing ComfyUI in CPU/offline mode")
    parser.add_argument("--comfy-python", type=Path, help="ComfyUI Python; defaults to COMFYUI/.venv/bin/python")
    parser.add_argument("--comfy-port", type=int, default=8188)
    args = parser.parse_args()
    if args.comfyui:
        args.comfyui = args.comfyui.resolve()
        args.comfy_python = (args.comfy_python or args.comfyui / ".venv/bin/python").absolute()
        if not (args.comfyui / "main.py").is_file() or not args.comfy_python.is_file():
            parser.error("--comfyui and --comfy-python must point to an existing ComfyUI and Python environment")
        if not (args.comfyui / "custom_nodes/turbocider/__init__.py").is_file():
            parser.error("Install TurboCider nodes with tools/comfyui/install.py first")

    if not args.cli.is_file():
        parser.error("Build/package TurboCider first or pass --cli /path/to/turbocider")
    args.state_dir = args.state_dir.resolve()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    children = []
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        native = subprocess.Popen([str(args.cli.resolve()), "serve", args.socket, str(args.state_dir / "native")])
        children.append(native)
        client = Client(args.socket, timeout=1)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if native.poll() is not None:
                raise RuntimeError("Native service exited; choose an unused socket/state directory")
            try:
                status = client.service_status()
                if status["pid"] == native.pid:
                    break
            except TransportError:
                pass
            time.sleep(0.1)
        else:
            raise TimeoutError("Native API did not start within 30 seconds")
        http = subprocess.Popen([sys.executable, str(Path(__file__).with_name("serve.py")), "--socket", args.socket,
                                 "--data-dir", str(args.state_dir / "http"), "--port", str(args.port)])
        children.append(http)
        if args.comfyui:
            environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                           "TURBOCIDER_API_URL": f"http://127.0.0.1:{args.port}"}
            comfy = subprocess.Popen([str(args.comfy_python), str(args.comfyui / "main.py"),
                                      "--cpu", "--offline", "--listen", "127.0.0.1", "--port", str(args.comfy_port),
                                      "--disable-auto-launch"], cwd=args.comfyui, env=environment)
            children.append(comfy)
        while True:
            for child in children:
                if child.poll() is not None:
                    raise RuntimeError(f"API process {child.pid} exited with status {child.returncode}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping TurboCider API processes; waiting for native cancellation…", flush=True)
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
        for child in reversed(children):
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                # Only this launcher's exact owned process, never other services.
                child.kill()
                child.wait()


if __name__ == "__main__":
    main()
