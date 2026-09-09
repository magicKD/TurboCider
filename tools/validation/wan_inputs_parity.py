"""Validate native Wan RoPE and request-local MLX re-noising without model weights."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', type=Path, required=True)
    a = p.parse_args()
    import mlx.core as mx
    import numpy as np
    import torch
    reports = []
    with tempfile.TemporaryDirectory(prefix='tc-wan-inputs-') as directory:
        for grid in [(1, 1, 1), (5, 8, 12), (21, 30, 52)]:
            positions = torch.meshgrid(*(torch.arange(n, dtype=torch.float32) for n in grid), indexing='ij')
            cosine, sine = [], []
            for position, dim in zip(positions, (44, 42, 42)):
                frequency = 1. / (10000. ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
                angle = torch.outer(position.reshape(-1), frequency)
                cosine.append(angle.cos().repeat_interleave(2, dim=-1))
                sine.append(angle.sin().repeat_interleave(2, dim=-1))
            reference = {'cosine': torch.cat(cosine, dim=1).numpy(), 'sine': torch.cat(sine, dim=1).numpy()}
            for seed in (0, 42, 4294967297):
                mx.random.seed(seed)
                shape = (1, 16, grid[0], grid[1] * 2, grid[2] * 2)
                for index in range(2):
                    reference[f'renoise_{index}'] = np.asarray(mx.random.normal(shape))
                output = Path(directory) / 'inputs.safetensors'
                subprocess.run([str(a.probe.resolve()), *map(str, grid), str(seed), str(output)], check=True)
                actual = mx.load(str(output))
                row = {'grid': grid, 'seed': seed}
                for key, ref in reference.items():
                    got = np.asarray(actual[key])
                    if got.shape != ref.shape or got.dtype != ref.dtype or not np.isfinite(got).all():
                        raise AssertionError(f'{key}: invalid shape/dtype/nonfinite')
                    error = float(np.max(np.abs(got - ref)))
                    row[key] = {'exact': bool(np.array_equal(ref, got)), 'max_abs': error}
                    # GPU transcendental functions may differ from torch CPU.
                    # This is a table-error bound, not a video-quality gate.
                    if key.startswith('renoise') and not row[key]['exact']:
                        raise AssertionError(f're-noising sequence differs: {row}')
                    if key in ('cosine', 'sine') and error > 1e-5:
                        raise AssertionError(f'RoPE table error exceeds bound: {row}')
                reports.append(row)
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
