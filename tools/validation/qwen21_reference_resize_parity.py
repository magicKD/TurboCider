"""Native area/aspect/Lanczos RGBA resize vs independent Pillow float filters.

Native keeps float colors, deliberately avoiding Pillow RGBA's intermediate
8-bit premultiplication. Compare floating premultiplied channels, alpha edges,
orientation-independent dimensions, and identity behavior separately.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile

import mlx.core as mx
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(923)
    for h, w, resolution in [(47, 81, 128), (121, 73, 64), (13, 19, 64),
                             (512, 512, 1024), (32, 32, 32), (511, 113, 128)]:
        pixels = rng.random((h, w, 4), dtype=np.float32)
        pixels[:h // 3, :, 3] = 0
        pixels[-h // 3:, :, 3] = 1
        rh = round(math.sqrt(resolution * resolution * h / w) / 32) * 32
        rw = round(math.sqrt(resolution * resolution * w / h) / 32) * 32
        premult = pixels.copy()
        premult[..., :3] *= premult[..., 3:]
        expected = np.stack([np.asarray(Image.fromarray(premult[..., c]).resize((rw, rh), Image.Resampling.LANCZOS))
                             for c in range(4)], axis=-1)
        expected[..., 3] = np.clip(expected[..., 3], 0, 1)
        np.divide(expected[..., :3], expected[..., 3:], out=expected[..., :3], where=expected[..., 3:] > 1e-8)
        expected[..., :3][expected[..., 3] <= 1e-8] = 0
        expected = np.clip(expected, 0, 1)
        if (rh, rw) == (h, w):
            expected = pixels
        with tempfile.TemporaryDirectory(prefix="tc-qwen21-resize-") as temp:
            root = Path(temp)
            mx.save_safetensors(str(root / "input.safetensors"), {"pixels": mx.array(pixels[None])},
                                {"reference_resolution": str(resolution)})
            subprocess.run([str(args.probe.resolve()), str(root / "input.safetensors"), str(root / "output.safetensors")], check=True)
            actual = np.array(mx.load(str(root / "output.safetensors"))["pixels"])[0]
        assert actual.shape == expected.shape
        error = float(np.abs(actual - expected).max())
        # Near-zero alpha amplifies harmless filter rounding; test premult too.
        visible = expected[..., 3] > .01
        visible_error = float(np.abs(actual[..., :3][visible] - expected[..., :3][visible]).max())
        alpha_error = float(np.abs(actual[..., 3] - expected[..., 3]).max())
        assert visible_error < 2e-5 and alpha_error < 1e-6, (visible_error, alpha_error)
        assert np.isfinite(actual).all() and actual.min() >= 0 and actual.max() <= 1
        if (rh, rw) != (h, w):
            assert np.all(actual[..., :3][actual[..., 3] <= 1e-8] == 0)
        print(json.dumps({"input_hw": [h, w], "output_hw": [rh, rw], "max_abs": error,
                          "visible_max_abs": visible_error, "alpha_max_abs": alpha_error}), flush=True)


if __name__ == "__main__":
    main()
