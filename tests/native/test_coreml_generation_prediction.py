#!/usr/bin/env python3
"""Opt-in native Core ML load/predict from a sealed private generation."""
import argparse
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifact', type=Path, required=True)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[2]
sources = ['tests/native/coreml_generation_prediction_test.mm', 'native/models/z_image/coreml_generation.cpp',
           'native/runtime/streaming/source_lease.cpp', 'native/runtime/streaming/canonical_encoding.cpp',
           'native/runtime/memory_manifest.cpp', 'native/core/common.cpp']
with tempfile.TemporaryDirectory(prefix='tc-generation-predict-') as raw:
    root = Path(raw).resolve()
    binary = root / 'test'
    subprocess.run(['clang++', '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror', '-fobjc-arc',
                    '-I', str(ROOT / 'native'), '-I', str(ROOT / 'native/core'),
                    *(str(ROOT / s) for s in sources), '-framework', 'Foundation', '-framework', 'CoreML',
                    '-o', str(binary)], check=True)
    subprocess.run([str(binary), str(args.artifact.resolve()), str(root)], check=True, timeout=180)
