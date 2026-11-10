"""Validate original-code LTX W8A8 against an integer oracle on Metal."""
import subprocess
import tempfile
from pathlib import Path
from ltx_w8a8_build import compile_w8a8, ROOT


def main():
    clang = subprocess.check_output(['xcrun', '--find', 'clang'], text=True).strip()
    clangxx = subprocess.check_output(['xcrun', '--find', 'clang++'], text=True).strip()
    sdk = subprocess.check_output(['xcrun', '--sdk', 'macosx', '--show-sdk-path'], text=True).strip()
    ltx = ROOT / 'native/models/ltx_runtime'
    with tempfile.TemporaryDirectory(prefix='tc-ltx-w8a8-') as name:
        directory = Path(name)
        bridge = compile_w8a8(directory, clangxx, sdk)
        gpu = directory / 'ltx_gpu.o'
        subprocess.run([clang, '-std=c11', '-O2', '-fobjc-arc', '-isysroot', sdk,
                        '-c', str(ltx / 'ltx_gpu.m'), '-o', str(gpu)], check=True)
        binary = directory / 'ltx-w8a8-test'
        subprocess.run([clangxx, '-std=c++20', '-O2', '-isysroot', sdk, '-I', str(ltx),
                        str(ROOT / 'tests/native/ltx_w8a8_test.mm'), str(gpu), str(bridge),
                        '-framework', 'Foundation', '-framework', 'Metal',
                        '-framework', 'MetalPerformanceShaders', '-framework', 'MetalPerformanceShadersGraph',
                        '-o', str(binary)], check=True)
        subprocess.run([str(binary), str(ltx / 'ltx_shaders.metal')], check=True, timeout=120)


if __name__ == '__main__':
    main()
