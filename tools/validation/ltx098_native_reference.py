#!/usr/bin/env python3
"""Measure the public TurboCider CLI's resident GPU 0.9.8 I2V route.

The output directory must be new. This captures complete per-request wall time,
MP4 metadata and source identities; it is not a multi-prompt quality claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import selectors
import subprocess
import time

from ltx098_comfy_reference import MODEL_NAME, MODEL_SHA256, PROMPT, TEXT_NAME, TEXT_SHA256


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--count", type=int, choices=(1, 2), default=2)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.seed < 0 or args.seed + args.count - 1 > 0xffffffffffffffff:
        parser.error("seed sequence must fit uint64")
    cli, model, image = (path.resolve(strict=True) for path in (args.cli, args.model, args.input))
    output = args.output.resolve()
    if output.exists():
        parser.error("output directory must not exist")
    library = cli.parent / "libturbocider.dylib"
    tokenizer = model / "tokenizer/tokenizer.json"
    if digest(model / MODEL_NAME) != MODEL_SHA256 or digest(model / "text_encoders" / TEXT_NAME) != TEXT_SHA256:
        raise ValueError("model or text encoder SHA-256 differs from the pinned source")
    output.mkdir(parents=True)
    requests = []
    for index in range(args.count):
        request = {"schema_version": 1, "model": "ltx-video-0.9.8-distilled", "operation": "video.image",
                   "prompt": args.prompt, "inputs": [{"kind": "image", "role": "first_frame",
                       "path": str(image), "strength": args.strength}], "output": str(output / f"{index}.mp4"),
                   "width": args.size, "height": args.size, "frames": args.frames, "fps": 24,
                   "steps": 8, "seed": args.seed + index, "audio": False, "execution": "gpu", "residency": "resident"}
        path = output / f"request-{index}.json"
        path.write_text(json.dumps(request, indent=2) + "\n")
        requests.append(path)
    report = {"schema": "ltx098-native-cli-reference-v1", "status": "running",
              "cli_sha256": digest(cli), "library_sha256": digest(library),
              "checkpoint_sha256": MODEL_SHA256, "text_encoder_sha256": TEXT_SHA256,
              "tokenizer_sha256": digest(tokenizer), "input_sha256": digest(image),
              "scope": f"{args.size}x{args.size}, {args.frames} frames, 24fps, 8 Euler steps, first-frame strength {args.strength}; resident native GPU; wall includes MP4 export",
              "runs": []}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    started = previous = time.perf_counter()
    with (output / "stderr.txt").open("x") as stderr, (output / "stdout.jsonl").open("x") as stdout:
        process = subprocess.Popen([str(cli), "batch", str(model), *map(str, requests)],
                                   stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1)
        ready = selectors.DefaultSelector()
        ready.register(process.stdout, selectors.EVENT_READ)
        try:
            deadline = time.monotonic() + 1800 * args.count
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError("native batch did not complete before the deadline")
                if not ready.select(timeout=1):
                    if process.poll() is not None:
                        break
                    continue
                line = process.stdout.readline()
                if not line:
                    break
                stdout.write(line); stdout.flush()
                now = time.perf_counter()
                row = json.loads(line)
                index = len(report["runs"])
                if index >= args.count or row.get("execution") != "gpu" or row.get("actual_denoise_steps") != 8:
                    raise ValueError("native response does not prove the expected eight-step GPU request")
                movie = output / f"{index}.mp4"
                metadata = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=width,height,nb_frames,r_frame_rate,duration", "-of", "json", str(movie)], text=True))
                video = metadata["streams"][0]
                if (video["width"], video["height"], int(video["nb_frames"]), video["r_frame_rate"]) != (args.size, args.size, args.frames, "24/1"):
                    raise ValueError("native MP4 geometry differs from the request")
                report["runs"].append({"index": index, "seed": args.seed + index, "client_wall_seconds": now - previous,
                                       "media_sha256": digest(movie), "video_stream": video, "result": row})
                previous = now
                (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(json.dumps({"completed": index, "client_wall_seconds": report["runs"][-1]["client_wall_seconds"]}), flush=True)
            code = process.wait(timeout=20)
            if code or len(report["runs"]) != args.count:
                raise RuntimeError(f"native batch exited {code} after {len(report['runs'])} completed requests")
            report["status"] = "complete"
            report["total_process_seconds"] = time.perf_counter() - started
        except BaseException as error:
            report.update(status="failed", error=str(error))
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=10)
            raise
        finally:
            ready.close()
            process.stdout.close()
            (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
