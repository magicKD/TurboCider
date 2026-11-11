"""Validate spatial pooling/gating against independent FP64 attention math."""
import os
from pathlib import Path
import subprocess
import tempfile

from ltx_w8a8_build import ROOT, compile_w8a8


def main():
    clang = subprocess.check_output(['xcrun', '--find', 'clang'], text=True).strip()
    clangxx = subprocess.check_output(['xcrun', '--find', 'clang++'], text=True).strip()
    sdk = subprocess.check_output(['xcrun', '--sdk', 'macosx', '--show-sdk-path'], text=True).strip()
    ltx = ROOT / 'native/models/ltx_runtime'
    with tempfile.TemporaryDirectory(prefix='tc-ltx-kv-pool-') as folder:
        output = Path(folder)
        bridge = compile_w8a8(output, clangxx, sdk)
        gpu = output / 'ltx_gpu.o'
        subprocess.run([clang, '-std=c11', '-O2', '-fobjc-arc', '-isysroot', sdk,
                        '-c', str(ltx / 'ltx_gpu.m'), '-o', str(gpu)], check=True)
        binary = output / 'ltx-kv-pool-test'
        subprocess.run([clangxx, '-std=c++20', '-O2', '-isysroot', sdk, '-I', str(ltx),
                        str(ROOT / 'tests/native/ltx_kv_pool_test.mm'), str(gpu), str(bridge),
                        '-framework', 'Foundation', '-framework', 'Metal',
                        '-framework', 'MetalPerformanceShaders', '-framework', 'MetalPerformanceShadersGraph',
                        '-o', str(binary)], check=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith('TURBOCIDER_LTX_')}
        subprocess.run([str(binary), str(ltx / 'ltx_shaders.metal')], env=env, check=True, timeout=120)


if __name__ == '__main__':
    main()
