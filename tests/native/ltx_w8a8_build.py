"""Compile the standalone native LTX TensorOps bridge for GPU tests."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def compile_w8a8(directory, clangxx, sdk):
    directory = Path(directory)
    subprocess.run([sys.executable, str(ROOT / 'tools/native/embed_qwen21_cider_kernels.py'),
                    '--output', str(directory / 'qwen21_cider_kernels_generated.hpp')], check=True)
    output = directory / 'ltx_w8a8.o'
    subprocess.run([clangxx, '-std=c++20', '-O2', '-fobjc-arc', '-isysroot', sdk,
                    '-I', str(directory), '-I', str(ROOT / 'native/models/ltx_runtime'), '-c', str(ROOT / 'native/platform/apple/ltx_w8a8.mm'),
                    '-o', str(output)], check=True)
    return output
