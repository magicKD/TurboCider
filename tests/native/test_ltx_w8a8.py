"""Validate original-code LTX W8A8 against an integer oracle on Metal."""
import subprocess
import os
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
        configurations = [
            {},  # Qualified M5 default: fused BF16 output and rotation/quantization.
            {'TURBOCIDER_LTX_W8A8_FUSED_OUTPUT': '0'},
            {'TURBOCIDER_LTX_W8A8_FUSED_OUTPUT': '1'},
            {'TURBOCIDER_LTX_W8A8_FUSED_OUTPUT': '1', 'TURBOCIDER_LTX_CONVROT_SIMD': '0'},
            {'TURBOCIDER_LTX_W8A8_FUSED_OUTPUT': '1', 'TURBOCIDER_LTX_W8A8_TILE_M': '32'},
            {'TURBOCIDER_LTX_W8A8_FUSED_OUTPUT': '1', 'TURBOCIDER_LTX_W8A8_TILE_M': '64'},
            {'TURBOCIDER_LTX_W8A8_FUSED_OUTPUT': '1', 'TURBOCIDER_LTX_W8A8_NO_TG_BARRIER': '1'},
            {'TURBOCIDER_LTX_W8A8_FUSED_OUTPUT': '0', 'TURBOCIDER_LTX_CONVROT_SIMD': '1'},
            {'TURBOCIDER_LTX_W8A8_FUSED_OUTPUT': '1', 'TURBOCIDER_LTX_CONVROT_SIMD': '1'},
            {'TURBOCIDER_LTX_W8A8_FUSED_GELU_CONVROT': '1'},
            {'TURBOCIDER_LTX_W8A8_FUSED_GELU_CONVROT': '1', 'TURBOCIDER_LTX_CONVROT_SIMD': '0'},
            {'TURBOCIDER_LTX_W8A8_FUSED_ROT_QUANT': '1'},
            {'TURBOCIDER_LTX_W8A8_FUSED_ROT_QUANT': '0'},
            {'TURBOCIDER_LTX_W8A8_FUSED_ROT_QUANT': '1', 'TURBOCIDER_LTX_W8A8_FUSED_GELU_CONVROT': '1'},
            {'TURBOCIDER_LTX_W8A8_FUSED_ROT_QUANT': '1', 'TURBOCIDER_LTX_CONVROT_SIMD': '0'},
            {'TURBOCIDER_LTX_W8A8_FUSED_ROT_QUANT': '1', 'TURBOCIDER_LTX_W8A8_FUSED_OUTPUT': '0'},
            {'TURBOCIDER_LTX_W8A8_FUSED_GELU_OUTPUT': '1'},
            {'TURBOCIDER_LTX_W8A8_FUSED_GELU_OUTPUT': '1', 'TURBOCIDER_LTX_W8A8_FUSED_GELU_CONVROT': '1'},
            {'TURBOCIDER_LTX_W8A8_FUSED_GELU_OUTPUT': '1', 'TURBOCIDER_LTX_W8A8_FUSED_OUTPUT': '0'},
            {'TURBOCIDER_LTX_W8A8_FUSED_GELU_OUTPUT': '1', 'TURBOCIDER_LTX_CONVROT_SIMD': '0'},
            {'TURBOCIDER_LTX_W8A8_FUSED_GELU_OUTPUT': '1', 'TURBOCIDER_LTX_W8A8_TILE_M': '32'},
            {'TURBOCIDER_LTX_W8A8_FUSED_GELU_OUTPUT': '1', 'TURBOCIDER_LTX_W8A8_TILE_M': '64'},
        ]
        for config in configurations:
            for stage_workspace in (False, True):
                environment = dict(os.environ, TURBOCIDER_LTX_W8A8_TILE_M='128',
                                   TURBOCIDER_LTX_W8A8_NO_TG_BARRIER='0')
                environment.pop('TURBOCIDER_LTX_W8A8_FUSED_OUTPUT', None)
                environment.pop('TURBOCIDER_LTX_CONVROT_SIMD', None)
                environment.pop('TURBOCIDER_LTX_W8A8_FUSED_GELU_CONVROT', None)
                environment.pop('TURBOCIDER_LTX_W8A8_FUSED_ROT_QUANT', None)
                environment.pop('TURBOCIDER_LTX_W8A8_FUSED_GELU_OUTPUT', None)
                environment.pop('TURBOCIDER_TEST_STAGE_WORKSPACE', None)
                if stage_workspace:
                    environment['TURBOCIDER_TEST_STAGE_WORKSPACE'] = '1'
                environment.update(config)
                print('LTX W8A8 configuration:', config, 'stage_workspace:', stage_workspace, flush=True)
                subprocess.run([str(binary), str(ltx / 'ltx_shaders.metal')],
                               env=environment, check=True, timeout=120)
        stage_gpu = directory / 'ltx_gpu_stage_hooks.o'
        subprocess.run([clang, '-std=c11', '-O2', '-fobjc-arc', '-isysroot', sdk,
                        '-DTURBOCIDER_ENABLE_TEST_HOOKS=1', '-c', str(ltx / 'ltx_gpu.m'), '-o', str(stage_gpu)], check=True)
        stage_test = directory / 'ltx-stage-workspace-test'
        subprocess.run([clangxx, '-std=c++20', '-O2', '-fobjc-arc', '-isysroot', sdk, '-I', str(ltx),
                        '-DTURBOCIDER_ENABLE_TEST_HOOKS=1',
                        str(ROOT / 'tests/native/ltx_stage_workspace_test.mm'), str(stage_gpu), str(bridge),
                        '-framework', 'Foundation', '-framework', 'Metal',
                        '-framework', 'MetalPerformanceShaders', '-framework', 'MetalPerformanceShadersGraph',
                        '-o', str(stage_test)], check=True)
        for config in ({}, {'TURBOCIDER_LTX_W8A8_FUSED_OUTPUT': '0'},
                       {'TURBOCIDER_LTX_CONVROT_SIMD': '0'},
                       {'TURBOCIDER_LTX_W8A8_FUSED_ROT_QUANT': '0'}):
            environment = {k: v for k, v in os.environ.items() if not k.startswith('TURBOCIDER_LTX_')}
            environment.update(config)
            environment['TURBOCIDER_LTX_W8A8_FUSED_QK_NORM_ROPE'] = '1'
            subprocess.run([str(stage_test), str(ltx / 'ltx_shaders.metal')], env=environment, check=True, timeout=120)
        quant_test = directory / 'ltx-rot-quant-test'
        subprocess.run([clangxx, '-std=c++20', '-O2', '-fobjc-arc', '-isysroot', sdk, '-I', str(ltx),
                        str(ROOT / 'tests/native/ltx_rot_quant_test.mm'), str(gpu), str(bridge),
                        '-framework', 'Foundation', '-framework', 'Metal',
                        '-framework', 'MetalPerformanceShaders', '-framework', 'MetalPerformanceShadersGraph',
                        '-o', str(quant_test)], check=True)
        environment = {k: v for k, v in os.environ.items() if not k.startswith('TURBOCIDER_LTX_')}
        subprocess.run([str(quant_test), str(ltx / 'ltx_shaders.metal'), str(ltx / 'ltx_shaders.metal')],
                       env=environment, check=True, timeout=120)


if __name__ == '__main__':
    main()
