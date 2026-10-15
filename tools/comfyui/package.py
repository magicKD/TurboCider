#!/usr/bin/env python3
"""Package the standalone gateway, node installer, workflows and API clients."""
import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    for folder in ("tools/comfyui", "bindings/python", "integrations/comfyui/turbocider"):
        target = args.output_dir / folder
        target.mkdir(parents=True, exist_ok=True)
        for source in (root / folder).glob("*.py"):
            shutil.copy2(source, target / source.name)
    shutil.copy2(root / "docs/public/COMFYUI.md", args.output_dir / "README.md")
    tutorial = (root / "docs/public/COMFYUI_TUTORIAL.zh-CN.md").read_text(encoding="utf-8")
    (args.output_dir / "COMFYUI_TUTORIAL.zh-CN.md").write_text(
        tutorial.replace("(COMFYUI.md)", "(README.md)"), encoding="utf-8"
    )
    shutil.copy2(root / "LICENSE", args.output_dir / "LICENSE")
    print(f"Packaged ComfyUI integration: {args.output_dir}")


if __name__ == "__main__":
    main()
