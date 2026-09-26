"""Pre-load rejection checks for the experimental native PE-I2I sampler."""
import argparse
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--pixels', type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='tc-pe-sample-validation-') as directory:
        output = Path(directory) / 'must-not-exist.json'
        cases = (
            ('0', 'Edit image', 'token budget', []),
            ('-1', 'Edit image', 'token budget', []),
            ('24001', 'Edit image', 'token budget', []),
            ('1', '', 'empty', []),
            ('1', ' \n', 'empty', []),
            ('0', 'Edit image', 'token budget', ['--vision-fp32', '--recurrent']),
            ('0', 'Edit image', 'token budget', ['--chunked', '--vision-fp32']),
            ('1', 'Edit image', 'duplicate/conflicting', ['--chunked', '--recurrent']),
            ('1', 'Edit image', 'duplicate', ['--vision-fp32', '--vision-fp32']),
            ('1', 'Edit image', 'unknown', ['--unknown']),
        )
        for budget, prompt, expected, options in cases:
            result = subprocess.run([
                str(args.probe.resolve()), str(args.weights.resolve()),
                str(args.pixels.resolve()), prompt, str(output), budget, *options,
            ], capture_output=True, text=True, timeout=30)
            assert result.returncode != 0, (budget, prompt, result.stdout)
            assert expected in result.stderr, result.stderr
            assert 'load ' not in result.stderr, 'invalid request loaded checkpoint weights'
            assert not output.exists(), 'invalid request wrote a successful result'
    print(f'PASS {len(cases)} native PE-I2I sampler pre-load rejection cases')


if __name__ == '__main__':
    main()
