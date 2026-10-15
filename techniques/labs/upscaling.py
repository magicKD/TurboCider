"""Spatial-only overlap/halo/feather/alpha reference with an explicit toy model."""
import math
import random

try:
    from .common import metrics, report
except ImportError:
    from common import metrics, report


def reflect(index, length):
    if length <= 1:
        return 0
    period = 2 * length - 2
    index %= period
    return index if index < length else period - index


def tiled_upscale(seed=17):
    rng, height, width, scale, tile, overlap = random.Random(seed), 7, 9, 2, 4, 2
    image = [[[rng.random() for _ in range(3)] for _ in range(width)] for _ in range(height)]
    alpha = [[rng.random() for _ in range(width)] for _ in range(height)]
    def model_pixel(row, col, crop=None):
        # Toy receptive-field-one blur -> nearest x2, NOT trained Real-ESRGAN.
        values = [0., 0., 0.]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                y, x = row + dy, col + dx
                if crop:
                    top, bottom, left, right = crop
                    y = top + reflect(y - top, bottom - top)
                    x = left + reflect(x - left, right - left)
                else:
                    y, x = reflect(y, height), reflect(x, width)
                for c in range(3):
                    values[c] += image[y][x][c] / 9
        return values
    reference = [[model_pixel(row // scale, col // scale) for col in range(width * scale)] for row in range(height * scale)]
    def assemble(use_context):
        out = [[[0.] * 3 for _ in range(width * scale)] for _ in range(height * scale)]
        mass = [[0.] * (width * scale) for _ in range(height * scale)]
        for top in range(0, height, tile - overlap):
            for left in range(0, width, tile - overlap):
                bottom, right = min(top + tile, height), min(left + tile, width)
                for row in range(top * scale, bottom * scale):
                    for col in range(left * scale, right * scale):
                        weight = min(row - top * scale + 1, bottom * scale - row,
                                     col - left * scale + 1, right * scale - col)
                        value = model_pixel(row // scale, col // scale, None if use_context else (top, bottom, left, right))
                        mass[row][col] += weight
                        for c in range(3):
                            out[row][col][c] += weight * value[c]
        if any(value <= 0 for row in mass for value in row):
            raise ValueError("uncovered output pixels")
        return [[[value / mass[row][col] for value in pixel] for col, pixel in enumerate(values)] for row, values in enumerate(out)]
    exact, approximate = assemble(True), assemble(False)
    resized_alpha = [[alpha[row // scale][col // scale] for col in range(width * scale)] for row in range(height * scale)]
    assert metrics(reference, exact)["max_abs"] < 1e-12
    assert resized_alpha[0][0] == alpha[0][0] and resized_alpha[-1][-1] == alpha[-1][-1]
    return report("upscale-tiles", {"scale": scale, "tile": tile, "overlap": overlap,
                  "correct_halo_feather": metrics(reference, exact),
                  "local_reflect_feather_approximation": metrics(reference, approximate),
                  "alpha_preserved_separately": True,
                  "limits": "Toy blur/upscale callback and nearest alpha, not a learned SR model or quality recommendation. Real alpha resize/orientation should match the image pipeline."})
