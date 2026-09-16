"""Compare untiled/tiled resident videos with byte-exact denoising tensors.

Only spatial decoded RGB is scored; no speed, semantic, temporal, or lip-sync
gate is inferred. Contact sheets sample frames and cannot qualify motion.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/native"))
from video_quality_gate import compare

TENSOR_NAMES = (
    "stage1_video",
    "stage1_audio",
    "stage2_input_video",
    "stage2_video",
    "stage2_audio",
)
REPORT_NAME = "report.json"
LABEL_HEIGHT = 26


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compare_tensors(reference: Path, candidate: Path, run: int) -> dict:
    result = {}
    for name in TENSOR_NAMES:
        paths = [
            directory / f"request-{run}-tensors/{name}.bf16"
            for directory in (reference, candidate)
        ]
        sizes = [path.stat().st_size for path in paths]
        if not sizes[0] or sizes[0] != sizes[1]:
            raise ValueError(f"invalid tensor sizes: {name}")
        hashes = [file_sha256(path) for path in paths]
        result[name] = {"sha256": hashes, "byte_exact": hashes[0] == hashes[1]}
    if not all(value["byte_exact"] for value in result.values()):
        raise ValueError("denoising inputs/outputs differ; not an isolated VAE comparison")
    return result


def extract_frames(video: Path, width: int, height: int,
                   indices: list[int]) -> list[bytes]:
    selection = "+".join(f"eq(n\\,{index})" for index in indices)
    command = [
        "ffmpeg", "-v", "error", "-i", str(video),
        "-vf", f"select={selection}", "-fps_mode", "passthrough",
        "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    frame_bytes = width * height * 3
    if len(raw) != frame_bytes * len(indices):
        raise ValueError("contact-sheet extraction frame count mismatch")
    return [raw[offset:offset + frame_bytes]
            for offset in range(0, len(raw), frame_bytes)]


def render_contact_sheet(videos: list[Path], output: Path,
                         width: int, height: int, indices: list[int]) -> None:
    from PIL import Image, ImageDraw

    sheet = Image.new(
        "RGB", (width * 2, (height + LABEL_HEIGHT) * len(indices)), "white")
    draw = ImageDraw.Draw(sheet)
    for column, video in enumerate(videos):
        frames = extract_frames(video, width, height, indices)
        label = "Untiled" if column == 0 else "Tiled"
        for row, (index, payload) in enumerate(zip(indices, frames)):
            top = row * (height + LABEL_HEIGHT)
            frame = Image.frombytes("RGB", (width, height), payload)
            sheet.paste(frame, (column * width, top + LABEL_HEIGHT))
            draw.text((column * width + 5, top + 5),
                      f"{label} | frame {index}", fill="black")
    sheet.save(output)


def write_report(output: Path, report: dict) -> None:
    (output / REPORT_NAME).write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n")


def evaluate(reference: Path, candidate: Path, output: Path, runs: int) -> None:
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "ltx-vae-tiling-quality-v1",
        "complete": False,
        "timing_comparison": None,
        "comparisons": [],
    }
    for run in range(runs):
        videos = [directory / f"request-{run}.mp4"
                  for directory in (reference, candidate)]
        tensors = compare_tensors(reference, candidate, run)
        metrics = compare(*videos)
        contract_keys = ("shape_equal", "fps_equal", "frame_count_equal", "finite")
        if not all(metrics[key] for key in contract_keys):
            raise ValueError("decoded video contract failed")
        width = metrics["reference"]["width"]
        height = metrics["reference"]["height"]
        count = metrics["frame_count_compared"]
        indices = sorted({0, count // 2, count - 1})
        render_contact_sheet(
            videos, output / f"request-{run}-comparison.png",
            width, height, indices)
        report["comparisons"].append({
            "run": run,
            "videos": [str(path.resolve()) for path in videos],
            "video_sha256": [file_sha256(path) for path in videos],
            "tensor_checks": tensors,
            "metrics": metrics,
            "contact_sheet_indices": indices,
        })
        summary = {
            key: value for key, value in metrics.items()
            if key not in ("frame_metrics", "reference", "candidate")
        }
        print(json.dumps({"run": run, **summary}), flush=True)
        write_report(output, report)
    report["complete"] = True
    write_report(output, report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be positive")
    evaluate(args.reference, args.candidate, args.output, args.runs)


if __name__ == "__main__":
    main()
