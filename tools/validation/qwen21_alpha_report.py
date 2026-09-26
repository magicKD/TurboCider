"""CPU-only alpha inspection/composites. Alpha coverage alone is not segmentation quality."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New report directory")
    args = parser.parse_args()
    source = Image.open(args.image)
    assert "A" in source.getbands(), "image has no alpha channel"
    rgba = source.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"))
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(source=str(args.image), size=list(rgba.size),
                  alpha_min=int(alpha.min()), alpha_max=int(alpha.max()),
                  transparent_fraction=float(np.mean(alpha < 10)), opaque_fraction=float(np.mean(alpha > 245)),
                  intermediate_fraction=float(np.mean((alpha >= 10) & (alpha <= 245))),
                  scope="alpha statistics only; inspect subject boundaries and composites for semantic quality")
    rgba.getchannel("A").save(args.output / "alpha.png")
    for name, shade in (("gray", 128), ("white", 255), ("black", 0)):
        background = Image.new("RGBA", rgba.size, (shade, shade, shade, 255))
        Image.alpha_composite(background, rgba).convert("RGB").save(args.output / f"{name}.png")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
