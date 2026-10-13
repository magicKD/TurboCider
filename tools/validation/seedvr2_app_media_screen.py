#!/usr/bin/env python3
"""Compare complete App exports with bounded, one-frame host memory.

Decoded RGB statistics are diagnostics, not shared-noise tensor qualification
or a claim of equivalence between GPU and the mixed Core ML numerical recipe.
"""
import argparse
from contextlib import contextmanager
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image

from seedvr2_framework_screen import sha256, write_json


class PixelStats:
    def __init__(self):
        self.count = self.frames = self.squared_error = 0
        self.reference_squared = self.actual_squared = self.dot = 0
        self.absolute_error = self.maximum_error = 0
        self.exact = True

    def add(self, reference, actual):
        if reference.shape != actual.shape or reference.dtype != np.uint8 or actual.dtype != np.uint8:
            raise ValueError("matching uint8 decoded pixels required")
        x, y = reference.astype(np.float64), actual.astype(np.float64)
        delta = x-y
        self.frames += 1
        self.count += x.size
        self.squared_error += float(np.sum(delta*delta))
        self.reference_squared += float(np.sum(x*x))
        self.actual_squared += float(np.sum(y*y))
        self.dot += float(np.sum(x*y))
        self.absolute_error += float(np.sum(np.abs(delta)))
        self.maximum_error = max(self.maximum_error, int(np.max(np.abs(delta), initial=0)))
        self.exact &= bool(np.array_equal(reference, actual))

    def result(self):
        if not self.count:
            raise ValueError("empty decoded media")
        mse = self.squared_error / self.count
        denominator = math.sqrt(self.reference_squared*self.actual_squared)
        return dict(decoded_frames=self.frames, decoded_pixels_bitwise_equal=self.exact,
            pixel_values=self.count, mae_8bit=self.absolute_error/self.count,
            rmse_8bit=math.sqrt(mse), maximum_error_8bit=self.maximum_error,
            psnr_db=10*math.log10(255**2/mse) if mse else None,
            cosine=self.dot/denominator if denominator else None,
            relative_l2=math.sqrt(self.squared_error/self.reference_squared) if self.reference_squared else None)


def read_frame(stream, count):
    data = bytearray()
    while len(data) < count:
        chunk = stream.read(count-len(data))
        if not chunk:
            if data:
                raise ValueError("truncated decoded frame")
            return None
        data.extend(chunk)
    return np.frombuffer(data, dtype=np.uint8)


@contextmanager
def video_frames(path, size):
    # Drain errors into an owned temporary file, never an unconsumed PIPE.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path),
            "-map", "0:v:0", "-fps_mode", "passthrough", "-pix_fmt", "rgb24",
            "-f", "rawvideo", "pipe:1"], stdout=subprocess.PIPE, stderr=errors)
        try:
            yield process.stdout
            if process.wait(timeout=30):
                errors.seek(0)
                raise ValueError("video decode failed: "+errors.read(4096).decode(errors="replace"))
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def inspect_video(path, size):
    data = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=width,height,nb_read_frames,avg_frame_rate",
        "-of", "json", str(path)]))
    if len(data["streams"]) != 1:
        raise ValueError("one output video stream required")
    stream = data["streams"][0]
    if (stream["width"], stream["height"], int(stream["nb_read_frames"]), Fraction(stream["avg_frame_rate"])) != (size, size, 25, 24):
        raise ValueError("App video frame/FPS/dimension mismatch")
    return stream


def compare(left, right, kind, size):
    suffix = "png" if kind == "image" else "mp4"
    paths = [root / f"app-{kind}-{size}.{suffix}" for root in (left, right)]
    identities = [sha256(path) for path in paths]
    stats = PixelStats()
    if kind == "image":
        with Image.open(paths[0]) as first, Image.open(paths[1]) as second:
            if first.size != (size,size) or second.size != first.size:
                raise ValueError("App image dimensions mismatch")
            # Include alpha in equality/statistics: do not silently drop it.
            stats.add(np.asarray(first.convert("RGBA")), np.asarray(second.convert("RGBA")))
    else:
        for path in paths:
            inspect_video(path,size)
        with video_frames(paths[0],size) as first, video_frames(paths[1],size) as second:
            while True:
                x, y = read_frame(first,size*size*3), read_frame(second,size*size*3)
                if x is None or y is None:
                    if x is not None or y is not None:
                        raise ValueError("decoded frame counts differ")
                    break
                stats.add(x,y)
        if stats.frames != 25:
            raise ValueError("not every source frame was decoded")
    return dict(kind=kind,size=size,paths=list(map(str,paths)),artifact_sha256=identities,
                artifact_bitwise_equal=identities[0] == identities[1], **stats.result())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=3, action="append", metavar=("NAME","LEFT_APP_DIR","RIGHT_APP_DIR"), required=True)
    parser.add_argument("--require-equal", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = [item[0] for item in args.pair]
    if len(names) != len(set(names)) or not set(args.require_equal) <= set(names) or args.output.exists():
        parser.error("unique named pairs, known equality checks and a fresh output required")
    report = dict(schema="seedvr2-app-export-comparison-v1",status="running",pairs=[],
        scope="every PNG RGBA value and every t25 video RGB frame; diagnostic exported-media statistics, not tensor qualification")
    try:
        for name, left, right in args.pair:
            roots = [Path(path).resolve(strict=True) for path in (left,right)]
            generations = []
            for root in roots:
                record = json.loads((root / "generation-report.json").read_text())
                if len(record) != 7 or any(item["seed"] != 42 for item in record):
                    raise ValueError("complete seven-request App regression with seed42 required")
                generations.append(dict(path=str(root / "generation-report.json"),
                    sha256=sha256(root / "generation-report.json"),
                    numerical_recipes=sorted(set(item["numerical_recipe"] for item in record))))
            rows = [compare(*roots,kind,size) for kind in ("image","video") for size in (512,1024)]
            passed = all(row["decoded_pixels_bitwise_equal"] and (row["kind"] != "image" or row["artifact_bitwise_equal"]) for row in rows)
            report["pairs"].append(dict(name=name,generation_reports=generations,media=rows,
                equality_required=name in args.require_equal,all_decoded_pixels_equal=passed))
            if name in args.require_equal and not passed:
                raise ValueError("required exact App export equality failed: "+name)
            print(name, json.dumps([{k:row[k] for k in ("kind","size","decoded_pixels_bitwise_equal","rmse_8bit","psnr_db")} for row in rows]), flush=True)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed",failure=str(error))
        raise
    finally:
        write_json(args.output,report)


if __name__ == "__main__":
    main()
