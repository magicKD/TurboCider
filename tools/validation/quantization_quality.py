"""Compare matched benchmark tensors/images; similarity is not aesthetic quality."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from safetensors.torch import load_file


def metrics(reference, candidate):
    a, b = reference.astype(np.float64).reshape(-1), candidate.astype(np.float64).reshape(-1)
    delta = a - b
    mse = np.mean(delta ** 2)
    return {"cosine": float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
            "rmse": float(np.sqrt(mse)), "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(a)),
            "max_abs": float(np.max(np.abs(delta))), "finite": bool(np.isfinite(b).all())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    a = p.parse_args()
    root = a.directory
    report = {"encoder": {}, "images": {}, "limitation": "One prompt/seed; pixel similarity is not prompt adherence or human preference."}
    for count in (32, 128):
        ref = load_file(root / f"encoder-bf16-{count}/conditioning.safetensors")["conditioning"].float().numpy()
        for bits in (4, 8):
            value = load_file(root / f"encoder-q{bits}-{count}/conditioning.safetensors")["conditioning"].float().numpy()
            report["encoder"][f"q{bits}-{count}"] = metrics(ref, value)
    folders = ["bf16-text-bf16", "int8_convrot-text-bf16", "bf16-text-q4", "bf16-text-q8", "int8_convrot-text-q4"]
    if (root / "nvfp4-text-bf16/images/0.png").exists():
        folders.insert(2, "nvfp4-text-bf16")
    reference = np.asarray(Image.open(root / folders[0] / "images/0.png").convert("RGB")) / 255.0
    sheet = Image.new("RGB", (960, 350 * ((len(folders) + 2) // 3)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, folder in enumerate(folders):
        image = Image.open(root / folder / "images/0.png").convert("RGB")
        report["images"][folder] = metrics(reference, np.asarray(image) / 255.0)
        x, y = i % 3 * 320, i // 3 * 350
        sheet.paste(image.resize((320, 320)), (x, y + 30))
        draw.text((x + 8, y + 8), folder, fill="black")
    sheet.save(root / "comparison.png")
    (root / "quality.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
