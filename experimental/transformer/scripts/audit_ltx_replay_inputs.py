"""Audit artifacts needed for a no-decoder LTX Stage-2 replay.

This deliberately does not synthesize a replay when an artifact is missing;
silently substituting a different conditioning tensor would invalidate timing
comparisons.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def file_info(path: Path) -> dict:
    if not path.is_file():
        return {"path": str(path), "present": False}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "present": True, "bytes": path.stat().st_size,
            "sha256": digest}


def audit(root: Path, conditioning: Path) -> dict:
    required = [
        conditioning / "video_context.bf16",
        conditioning / "audio_context.bf16",
        conditioning / "text_mask.bf16",
        root / "request-0-tensors/stage2_input_video.bf16",
        root / "request-0-tensors/stage1_audio.bf16",
        root / "request-0-tensors/stage2_video.bf16",
        root / "request-0-tensors/stage2_audio.bf16",
    ]
    info = [file_info(path) for path in required]
    metadata = root / "request-0-tensors/metadata.json"
    parsed = json.loads(metadata.read_text()) if metadata.is_file() else None
    return {"schema": "ltx-stage2-replay-input-audit-v1", "root": str(root),
            "conditioning": str(conditioning), "files": info,
            "metadata": parsed, "artifacts_present": all(x["present"] for x in info),
            "missing": [x["path"] for x in info if not x["present"]],
            "limitation": "Presence audit only: the fixed-geometry native replay validates tensor sizes and requires exact video/audio reference parity. Use stage1_audio as the input, not stage2_audio."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("conditioning", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root, args.conditioning), indent=2))
