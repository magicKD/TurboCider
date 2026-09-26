"""Offline PE file-decode checks; no Python inference or GPU work."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image, __version__ as pillow_version
from safetensors import safe_open
from safetensors.numpy import load_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-image', type=Path)
    parser.add_argument('--pixels-fixture', type=Path)
    parser.add_argument('--processor-fixture', type=Path)
    args = parser.parse_args()
    supplied = [args.source_image, args.pixels_fixture, args.processor_fixture]
    if any(supplied) and not all(supplied):
        parser.error('source image and both fixtures must be provided together')
    rng = np.random.default_rng(427)
    rgba = rng.integers(0, 256, (11, 7, 4), dtype=np.uint8)
    rgba[..., 3] = rng.choice([0, 1, 2, 127, 128, 254, 255], (11, 7))
    cases = []
    rejected = []
    fixture_report = None
    with tempfile.TemporaryDirectory(prefix='tc-pe-decode-') as directory:
        root = Path(directory)
        images = dict(rgb=Image.fromarray(rgba[..., :3]), rgba=Image.fromarray(rgba),
                      gray=Image.fromarray(rgba[..., 0]),
                      gray_alpha=Image.fromarray(rgba[..., [0, 3]], 'LA'))
        palette = Image.fromarray(rng.integers(0, 16, (11, 7), dtype=np.uint8), 'P')
        palette.putpalette(rng.integers(0, 256, 768, dtype=np.uint8).tolist())
        palette.info['transparency'] = bytes([0, 1, 127, 255] * 64)
        images['palette_alpha'] = palette
        for orientation in range(1, 9):
            oriented = images['rgba'].copy()
            exif = oriented.getexif()
            exif[274] = orientation
            oriented.info['exif'] = exif.tobytes()
            images[f'exif_{orientation}'] = oriented
        paths = []
        for name, image in images.items():
            path = root / f'{name}.png'
            image.save(path)
            paths.append(path)
        jpeg = root / 'rgb.jpg'
        images['rgb'].save(jpeg, quality=95, subsampling=0)
        paths.append(jpeg)
        for path in paths:
            output = root / 'decoded.safetensors'
            run = subprocess.run([str(args.probe.resolve()), str(path), str(output)], capture_output=True, text=True)
            record = dict(case=path.name, native_exit=run.returncode, passed=False)
            if run.returncode:
                record['error'] = run.stderr.strip()
            else:
                # Exact reference pe_core.load_image(path, max_pixels=0)
                # behavior: RGB conversion without EXIF transposition/ICC.
                expected = np.asarray(Image.open(path).convert('RGB'))
                actual = load_file(output)['pixels']
                shape_ok = actual.shape == (1, *expected.shape)
                record.update(shape=list(actual.shape), dtype=str(actual.dtype))
                if shape_ok and actual.dtype == np.uint8:
                    delta = np.abs(actual[0].astype(np.int16) - expected.astype(np.int16))
                    record.update(max_abs=int(delta.max()), mismatch_count=int(np.count_nonzero(delta)),
                                  passed=bool(not delta.any()))
            cases.append(record)
        invalid = root / 'not-an-image.png'
        invalid.write_bytes(b'not an image')
        gray16 = root / 'gray16.png'
        Image.fromarray(rng.integers(0, 65536, (11, 7), dtype=np.uint16)).save(gray16)
        cmyk = root / 'cmyk.jpg'
        Image.fromarray(rgba, 'CMYK').save(cmyk)
        for path in [root / 'missing.png', invalid, gray16, cmyk]:
            run = subprocess.run([str(args.probe.resolve()), str(path), str(root / 'invalid.safetensors')],
                                 capture_output=True, text=True)
            rejected.append(dict(case=path.name, rejected=run.returncode != 0, error=run.stderr.strip()))
        if args.source_image:
            output = root / 'source.safetensors'
            subprocess.run([str(args.probe.resolve()), str(args.source_image), str(output), '--processor'], check=True)
            actual = load_file(output)
            pixels = load_file(args.pixels_fixture)['pixels0'][..., :3]
            with safe_open(args.processor_fixture, framework='numpy') as file:
                # Select only FP32/int tensors: other fixture tensors are BF16.
                patches = file.get_tensor('patches0')
                grids = file.get_tensor('grids')
            fixture_report = dict(source=str(args.source_image),
                                  pixels_exact=bool(np.array_equal(actual['pixels'], pixels)),
                                  grids_exact=bool(np.array_equal(actual['grids'], grids)),
                                  patches_exact=bool(np.array_equal(actual['patches0'], patches)))
            fixture_report['passed'] = all(fixture_report[key] for key in ('pixels_exact', 'grids_exact', 'patches_exact'))
    report = dict(passed=all(x['passed'] for x in cases) and all(x['rejected'] for x in rejected),
                  png_exact_passed=all(x['passed'] for x in cases if x['case'].endswith('.png')),
                  real_image_fixture=fixture_report,
                  cases=cases, rejections=rejected, pillow_version=pillow_version,
                  scope='raw RGB bytes only; hidden RGB and untransposed EXIF; no processor/inference quality')
    if fixture_report is not None:
        report['passed'] = report['passed'] and fixture_report['passed']
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit('PE file decoder parity failed')


if __name__ == '__main__':
    main()
