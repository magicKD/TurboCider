"""CPU pixel/timing report for qwen21_dit_cache_benchmark.py (NumPy + Pillow).

SSIM uses an 11x11 Gaussian window, sigma=1.5, population covariance, RGB
channels averaged, and valid image interiors. No perceptual model is downloaded.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def image_metrics(a, b):
    if a.shape != b.shape:
        raise ValueError("image dimensions differ")
    x, y = a.astype(np.float64), b.astype(np.float64)
    delta = x - y
    mse = float(np.mean(delta ** 2))
    axis = np.arange(-5, 6, dtype=np.float64)
    kernel = np.exp(-(axis ** 2) / (2 * 1.5 ** 2))
    kernel /= kernel.sum()
    def blur(value):
        value = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 0, value)
        return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, value)
    ux, uy = blur(x), blur(y)
    vx, vy = blur(x * x) - ux * ux, blur(y * y) - uy * uy
    covariance = blur(x * y) - ux * uy
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = ((2 * ux * uy + c1) * (2 * covariance + c2) /
            ((ux * ux + uy * uy + c1) * (vx + vy + c2)))
    return dict(mae_255=float(np.mean(np.abs(delta))), rmse_255=math.sqrt(mse),
        max_abs_255=float(np.max(np.abs(delta))),
        psnr_db=None if mse == 0 else 10 * math.log10(255 ** 2 / mse),
        identical=bool(np.array_equal(a, b)), ssim=float(ssim.mean()),
        correlation=float(np.corrcoef(x.reshape(-1), y.reshape(-1))[0, 1])
            if x.std() > 0 and y.std() > 0 else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    records = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines() if line.strip()]
    by_name = {Path(result["output"]).stem: result for result in records}
    groups = sorted({name.rsplit("-", 1)[0] for name in by_name
        if name.endswith("-off")})
    rows = []
    for group in groups:
        base = by_name[group + "-off"]
        base_request = json.loads((root / (group + "-off.json")).read_text())
        baseline_rgba = np.asarray(Image.open(base["output"]).convert("RGBA"))
        if baseline_rgba.shape != (512, 512, 4):
            raise ValueError(f"expected a 512x512 output: {group}")
        baseline = baseline_rgba[:, :, :3]
        repeat = by_name.get(group + "-off-repeat")
        controls = [base["timings_seconds"]] + ([repeat["timings_seconds"]] if repeat else [])
        denoise_control = sum(item["denoise"] for item in controls) / len(controls)
        wall_control = sum(item["request_wall"] for item in controls) / len(controls)
        images = []
        for mode in ["off", "conservative", "balanced", "fast", "sglang", "off-repeat"]:
            key = group + "-" + mode
            if key not in by_name:
                continue
            candidate = by_name[key]
            request = json.loads((root / (key + ".json")).read_text())
            normalize = lambda r: {k: v for k, v in r.items() if k not in ("output", "allow_approximation", "qwen21_dit_cache")}
            if normalize(request) != normalize(base_request):
                raise ValueError(f"unmatched request inputs: {key}")
            rgba = np.asarray(Image.open(candidate["output"]).convert("RGBA"))
            if rgba.shape != baseline_rgba.shape:
                raise ValueError(f"output dimensions differ: {key}")
            rgb = Image.fromarray(rgba[:, :, :3])
            alpha_delta = rgba[:, :, 3].astype(np.float64) - baseline_rgba[:, :, 3]
            b = candidate["timings_seconds"]
            row = dict(name=key, mode=mode, wall_seconds=b["request_wall"],
                denoise_seconds=b["denoise"], wall_speedup=wall_control / b["request_wall"],
                denoise_speedup=denoise_control / b["denoise"],
                control_count=len(controls),
                denoise_control_seconds=[item["denoise"] for item in controls],
                wall_control_seconds=[item["request_wall"] for item in controls],
                cached_steps=(candidate.get("qwen21_dbcache") or {}).get("cached_steps", 0),
                rgba_identical=bool(np.array_equal(rgba, baseline_rgba)),
                alpha_mae_255=float(np.mean(np.abs(alpha_delta))),
                alpha_max_abs_255=float(np.max(np.abs(alpha_delta))),
                alpha_range=[int(rgba[:, :, 3].min()), int(rgba[:, :, 3].max())],
                **image_metrics(baseline, np.asarray(rgb)))
            rows.append(row)
            images.append((mode, rgb, row))
        sheet = Image.new("RGB", (512 * len(images), 554), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (mode, rgb, row) in enumerate(images):
            sheet.paste(rgb, (512 * i, 42))
            draw.text((512 * i + 8, 8), f'{group} / {mode} / {row["wall_seconds"]:.1f}s / SSIM {row["ssim"]:.4f}', fill="black")
        sheet.save(root / (group + "-comparison.png"))
    manifest = json.loads((root / "manifest.json").read_text())
    report = dict(scope="Matched seed/prompt/LoRA/geometry pixel comparison; visual and semantic review required. Warm encoding, staged GPU.",
        prefix_snapshot=manifest.get("prefix_snapshot", "0"), rows=rows)
    (root / "quality.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
