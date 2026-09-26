"""Roundtrip straight-alpha PNGs and compare native Qwen21 schedules."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--mflux", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.mflux / "src"))
    from mflux.models.common.config import Config, ModelConfig
    with tempfile.TemporaryDirectory(prefix="tc-qwen21-media-") as directory:
        subprocess.run([str(args.probe.resolve()), directory], check=True)
        rgba = Image.open(Path(directory) / "rgba.png")
        rgb = Image.open(Path(directory) / "rgb.png")
        expected = [(255,0,0,0), (0,255,0,128), (0,0,255,255), (255,255,255,255)]
        assert rgba.mode == "RGBA" and list(rgba.getdata()) == expected, list(rgba.getdata())
        assert rgb.mode == "RGB" and list(rgb.getdata()) == [x[:3] for x in expected]
        decoded = mx.load(str(Path(directory) / "decoded.safetensors"))
        pixels = np.array(decoded["rgba"]).reshape(-1, 4)
        reference = np.asarray(expected) / 255.
        assert np.max(np.abs(pixels[:, 3] - reference[:, 3])) < 1e-6, pixels
        visible = reference[:, 3] > 0
        assert np.max(np.abs(pixels[visible, :3] - reference[visible, :3])) < 1e-5, pixels
        opaque = np.array(decoded["rgb"]).reshape(-1, 4)
        assert np.max(np.abs(opaque[:, :3] - reference[:, :3])) < 1e-5, opaque
        assert np.all(opaque[:, 3] == 1), opaque
        rng = np.random.default_rng(43)
        test_pixels = rng.integers(0, 256, (11, 7, 4), dtype=np.uint8)
        test_pixels[:, :, 3] = rng.choice([0, 1, 128, 253, 254, 255], (11, 7))
        for orientation in range(1, 9):
            path = Path(directory) / f"oriented-{orientation}.png"
            source = Image.fromarray(test_pixels)
            exif = source.getexif()
            exif[274] = orientation
            source.save(path, exif=exif)
            output = Path(directory) / "import.safetensors"
            subprocess.run([str(args.probe.resolve()), "--import", str(path), str(output)], check=True)
            actual = np.array(mx.load(str(output))["pixels"])[0]
            expected_pixels = np.asarray(ImageOps.exif_transpose(Image.open(path)), dtype=np.float32) / 255.
            assert actual.shape == expected_pixels.shape, (orientation, actual.shape)
            assert np.max(np.abs(actual[..., 3] - expected_pixels[..., 3])) < 1e-6, orientation
            visible = expected_pixels[..., 3] > 0
            error = np.max(np.abs(actual[..., :3][visible] - expected_pixels[..., :3][visible]))
            assert error < 1e-5, (orientation, float(error))
        schedules = mx.load(str(Path(directory) / "schedules.safetensors"))
        for steps in (2,4,40):
            reference = Config(ModelConfig.qwen_image_21(), num_inference_steps=steps, height=512, width=512).scheduler.sigmas
            error = mx.max(mx.abs(reference - schedules[str(steps)])).item()
            assert error < 1e-6, error
        print("PASS: straight alpha export, RGBA import/visible colors, RGB compatibility, sigma schedules and native input guards")


if __name__ == "__main__":
    main()
