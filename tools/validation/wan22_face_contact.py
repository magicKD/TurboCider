#!/usr/bin/env python3
"""Create face-detail contact sheets for local video-quality inspection.

Detection is only used to choose crops; it is not an identity or quality score.
"""
import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("videos", nargs="+", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise RuntimeError("face crop detector is unavailable")
    frames, tile, label = (0, 24, 48, 72, 96), 224, 24
    sheet = Image.new("RGB", (tile * len(frames), (tile + label) * len(args.videos)), "white")
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
                y_position = row * (tile + label)
                title = video.parent.name.removeprefix("wan22-portrait-").split("-2026")[0]
                draw.text((column * tile + 3, y_position + 4), f"{title} · {index}", fill="black")
                sheet.paste(image, (column * tile, y_position + label))
        finally:
            capture.release()
    sheet.save(args.output)


if __name__ == "__main__":
    main()
