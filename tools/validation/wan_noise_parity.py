"""Development-only seed compatibility test against the local ARM64 PyTorch."""
import argparse
import subprocess
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', required=True)
    a = p.parse_args()
    for count in (16, 32, 4096, 2096640):
        for seed in (0, 42, 4294967297):
            generated = subprocess.run([a.probe, str(count), str(seed)], check=True, capture_output=True)
            native = np.frombuffer(generated.stdout, dtype=np.float32)
            reference = torch.randn(count, generator=torch.Generator().manual_seed(seed)).numpy()
            if not np.array_equal(native, reference):
                raise AssertionError(f'count={count}, seed={seed}, max_abs={np.max(np.abs(native-reference))}')
    print('12 ARM64 FP32 initial-noise cases exact')


if __name__ == '__main__':
    main()
