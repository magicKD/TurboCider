#!/usr/bin/env python3
"""Create face-detail contact sheets for local video-quality inspection.

Detection is only used to choose crops; it is not an identity or quality score.
"""
import argparse
import math
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--frame-step", type=int, default=24)
    parser.add_argument("--last-frame", type=int, default=96)
    parser.add_argument("--columns", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    if args.frame_step <= 0 or args.last_frame < 0 or args.columns <= 0:
        parser.error("frame step and columns must be positive; last frame must be nonnegative")
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise RuntimeError("face crop detector is unavailable")
    frames, tile, label = range(0, args.last_frame + 1, args.frame_step), 224, 24
    rows_per_video = math.ceil(len(frames) / args.columns)
    sheet = Image.new("RGB", (tile * args.columns, (tile + label) * rows_per_video * len(args.videos)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, video in enumerate(args.videos):
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise ValueError(f"cannot read video: {video}")
        try:
            for column, index in enumerate(frames):
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, bgr = capture.read()
                if not ok:
                    raise ValueError(f"missing frame {index} in {video}")
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.05, 4, minSize=(25, 25))
                faces = [face for face in faces if face[1] + face[3] / 2 < bgr.shape[0] * .6]
                if not faces:
                    raise ValueError(f"no face detected at frame {index} in {video}")
                x, y, width, height = map(int, max(faces, key=lambda face: int(face[2]) * int(face[3])))
                margin = int(max(width, height) * .25)
                x0, y0 = max(0, x - margin), max(0, y - margin)
                x1, y1 = min(bgr.shape[1], x + width + margin), min(bgr.shape[0], y + height + margin)
                rgb = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb).resize((tile, tile), Image.Resampling.LANCZOS)
                x_position = (column % args.columns) * tile
                y_position = (row * rows_per_video + column // args.columns) * (tile + label)
                title = video.parent.name.removeprefix("wan22-portrait-").split("-2026")[0]
                draw.text((x_position + 3, y_position + 4), f"{title} · {index}", fill="black")
                sheet.paste(image, (x_position, y_position + label))
        finally:
            capture.release()
    sheet.save(args.output)


if __name__ == "__main__":
    main()
