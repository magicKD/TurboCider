"""Offline spatial edit differences, not a semantic quality acceptance gate."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def measure(source, output, mask):
    if source.shape != output.shape or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError('source/output must be matching RGB arrays')
    if mask.shape != source.shape[:2] or mask.dtype != np.bool_:
        raise ValueError('mask must be a matching boolean array')
    if not mask.any() or mask.all():
        raise ValueError('mask must contain both edit and preserve regions')
    if source.dtype != np.uint8 or output.dtype != np.uint8:
        raise ValueError('RGB arrays must be uint8')
    delta = (output.astype(np.float64) - source.astype(np.float64)) / 255.0
    regions = {}
    for name, region in [('edit', mask), ('preserve', ~mask)]:
        values = delta[region]
        regions[name] = dict(pixels=int(region.sum()), rgb_rmse=float(np.sqrt(np.mean(values ** 2))),
                             rgb_mae=float(np.mean(np.abs(values))),
                             rgb_mean_change=values.mean(axis=0).tolist(),
                             fraction_pixels_changed=float(np.any(values != 0, axis=1).mean()))
    return dict(regions=regions, quality_accepted=False,
                scope='spatial RGB differences only; no alignment, semantic edit correctness or hard-inpainting claim')


def main():
    from PIL import Image
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'image', 'mask', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    with Image.open(args.source) as image:
        source = image.convert('RGB')
    with Image.open(args.image) as image:
        output = image.convert('RGB')
    with Image.open(args.mask) as image:
        if image.size != source.size:
            raise ValueError('mask must share source geometry')
        if 'A' in image.getbands() and image.getchannel('A').getextrema() != (255, 255):
            raise ValueError('mask must be opaque; composite it explicitly before measurement')
        mask = image.convert('L').resize(output.size, Image.Resampling.NEAREST)
    source_size = source.size
    source = source.resize(output.size, Image.Resampling.LANCZOS)
    result = measure(np.asarray(source), np.asarray(output), np.asarray(mask) >= 128)
    result.update(source_size=list(source_size), measurement_size=list(output.size),
                  source_resampling='LANCZOS', mask_resampling='NEAREST', mask_threshold=128,
                  rgb_policy='stored RGB, alpha ignored, no ICC conversion or EXIF transpose',
                  files={name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                         for name, path in [('source', args.source), ('image', args.image), ('mask', args.mask)]})
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
