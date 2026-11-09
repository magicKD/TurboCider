#!/usr/bin/env python3
"""Check decoded RGB frames from two matching native CLI batch reports."""

import argparse
import hashlib
import json
from pathlib import Path

from ltx098_comfy_reference import sha256
from ltx098_vae_full_abba import rgb_frame_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference, candidate = args.reference.resolve(strict=True), args.candidate.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or reference == candidate:
        parser.error("new output file and two distinct batch directories required")
    old_path, new_path = reference / "summary.json", candidate / "summary.json"
    old, new = json.loads(old_path.read_text()), json.loads(new_path.read_text())
    identity = ("checkpoint_sha256", "text_encoder_sha256", "tokenizer_sha256", "input_sha256")
    if (old.get("status"), new.get("status")) != ("complete", "complete") or any(
            old.get(key) != new.get(key) or not old.get(key) for key in identity):
        raise ValueError("batch status or model/input identities differ")
    if len(old.get("runs", [])) != len(new.get("runs", [])) or not old.get("runs"):
        raise ValueError("batch lengths differ or are empty")
    results = []
    for i, (left, right) in enumerate(zip(old["runs"], new["runs"])):
        if (left.get("index"), left.get("seed"), left.get("video_stream")) != (
                right.get("index"), right.get("seed"), right.get("video_stream")):
            raise ValueError(f"request {i} seed or video geometry differs")
        old_request = json.loads((reference / f"request-{i}.json").read_text())
        new_request = json.loads((candidate / f"request-{i}.json").read_text())
        old_request.pop("output", None)
        new_request.pop("output", None)
        if old_request != new_request:
            raise ValueError(f"request {i} parameters differ")
        first, second = rgb_frame_hashes(reference / f"{i}.mp4"), rgb_frame_hashes(candidate / f"{i}.mp4")
        results.append({"index": i, "seed": left["seed"], "frames": len(first),
                        "reference_media_sha256": sha256(reference / f"{i}.mp4"),
                        "candidate_media_sha256": sha256(candidate / f"{i}.mp4"),
                        "reference_rgb_frame_list_sha256": hashlib.sha256("\n".join(first).encode()).hexdigest(),
                        "candidate_rgb_frame_list_sha256": hashlib.sha256("\n".join(second).encode()).hexdigest(),
                        "all_rgb_frames_identical": first == second})
    report = {"schema": "ltx098-package-exact-v1", "scope": "decoded RGB frames from two matching CLI batches",
              "reference_summary_sha256": sha256(old_path), "candidate_summary_sha256": sha256(new_path),
              "reference_cli_sha256": old["cli_sha256"], "candidate_cli_sha256": new["cli_sha256"],
              "reference_library_sha256": old["library_sha256"], "candidate_library_sha256": new["library_sha256"],
              **{key: old[key] for key in identity}, "runs": results,
              "all_rgb_frames_identical": all(row["all_rgb_frames_identical"] for row in results)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))
    if not report["all_rgb_frames_identical"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
