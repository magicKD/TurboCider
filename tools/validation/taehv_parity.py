"""Offline conversion and real-weight native TAEHV parity; never used by App."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    digest = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    if digest != 'd26151e76cdc2c9424bef988de874b33d9a53f30ef3060cd556c429c469c797e':
        raise ValueError('not the validated Wan2.1 TAEHV checkpoint')
    sys.path.insert(0, str(a.reference_root.resolve(strict=True)))
    import torch
    import mlx.core as mx
    import numpy as np
    from fastvideo.mlx_runtime.wan_vae import MLXTAEHVDecoder
    a.output.mkdir(parents=True, exist_ok=False)
    weights = {k: mx.array(v.float().numpy()) for k, v in
               torch.load(a.checkpoint, weights_only=True, map_location='cpu').items()
               if k.startswith('decoder.')}
    mx.save_safetensors(str(a.output / 'decoder.safetensors'), weights,
                        metadata={'source_sha256': digest, 'architecture': 'taew2_1'})
    reference = MLXTAEHVDecoder(a.checkpoint, z_dim=16)
    rng = np.random.default_rng(42)
    reports = []
    for i, shape in enumerate([(1, 1, 16, 4, 6), (1, 3, 16, 4, 6), (2, 2, 16, 4, 6)]):
        latent = mx.array(rng.standard_normal(shape).astype(np.float32))
        expected = reference.decode_ntchw(latent)
        mx.eval(expected)
        path = a.output / f'input-{i}.safetensors'
        output = a.output / f'output-{i}.safetensors'
        mx.save_safetensors(str(path), {'latent': latent})
        subprocess.run([str(a.probe.resolve()), str((a.output / 'decoder.safetensors').resolve()),
                        str(path.resolve()), str(output.resolve())], check=True)
        actual = mx.load(str(output))['output']
        mx.eval(actual)
        x, y = np.asarray(expected), np.asarray(actual)
        if x.shape != y.shape or x.dtype != y.dtype:
            raise AssertionError('TAEHV output shape/dtype mismatch')
        reports.append({'input_shape': shape, 'output_shape': x.shape,
                        'exact': bool(np.array_equal(x, y)),
                        'finite': bool(np.isfinite(x).all() and np.isfinite(y).all()),
                        'max_abs': float(np.max(np.abs(x - y)))})
    (a.output / 'report.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports, indent=2))
    if not all(row['exact'] and row['finite'] for row in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
