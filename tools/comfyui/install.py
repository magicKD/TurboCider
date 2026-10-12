#!/usr/bin/env python3
"""Install only the TurboCider node code into an existing ComfyUI. No pip or model downloads."""
import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comfyui", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    comfy = args.comfyui.resolve()
    if not (comfy / "main.py").is_file() or not (comfy / "custom_nodes").is_dir():
        parser.error("Expected a ComfyUI source directory containing main.py and custom_nodes/")
    target = comfy / "custom_nodes" / "turbocider"
    target.mkdir(exist_ok=True)
    for source in (root / "integrations/comfyui/turbocider").glob("*.py"):
        shutil.copy2(source, target / source.name)
    shutil.copy2(root / "bindings/python/turbocider_http.py", target / "turbocider_http.py")
    print(f"Installed TurboCider nodes: {target}\nStart the API, restart ComfyUI, then refresh node definitions.")


if __name__ == "__main__":
    main()
