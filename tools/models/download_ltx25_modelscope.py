#!/usr/bin/env python3
"""Download pinned LTX 2.5 INT8, tokenizer, VAE and upscaler from ModelScope."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/validation'))
from download_qwen21_gguf import local_size
from modelscope_ranges import download_ranges


def main():
    if sys.version_info < (3, 11):
        raise SystemExit('Use .venv/bin/python3 (Python 3.11+ is required for SHA256 verification)')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'models/LTX-2.5-INT8')
    parser.add_argument('--jobs', type=int, choices=range(1, 5), default=4,
                        help='Concurrent independent component transfers (1-4)')
    parser.add_argument('--connections', type=int, choices=range(1, 9), default=4,
                        help='Bounded range connections per component (1-8)')
    args = parser.parse_args()
    assets = json.loads(Path(__file__).with_name('ltx25_modelscope_assets.json').read_text())
    root = args.root.absolute()
    root.mkdir(parents=True, exist_ok=True)
    remaining = 0
    for asset in assets:
        target = root / asset['relative_path']
        local_size(target)
        offset = local_size(target.with_name(target.name + '.part'))
        if offset > asset['bytes']:
            raise ValueError('partial exceeds pinned asset size')
        remaining += 0 if target.exists() else asset['bytes'] - offset
    if shutil.disk_usage(root).free < remaining + (4 << 30):
        raise ValueError('insufficient space for assets and 4 GiB reserve')
    def fetch(asset):
        return download_ranges(asset, root, connections=args.connections)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(fetch, assets))
    # The native audio route also needs the verified provenance sidecar.
    subprocess.run([sys.executable, str(ROOT / 'tools/ltx/prepare_ltx_audio_manifest.py'),
                    '--model-root', str(root)], check=True)
    (root / 'download-receipt.json').write_text(json.dumps({
        'schema': 'tc-ltx25-modelscope-pinned-v1', 'assets': results,
        'runtime_support_verified': False, 'performance_verified': False,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
