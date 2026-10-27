"""Actual new-library mixed BF16 input/F32 coefficient ConvRot partial checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


@unittest.skipUnless(sys.platform=='darwin','requires Apple MLX')
class ConvrotSourceRuntimeTests(unittest.TestCase):
    def test_original_scales_f32_partial_and_source_adapter(self):
        native=Path(os.environ.get('TURBOCIDER_TEST_NATIVE_DIR',ROOT/'build/native')).resolve()
        manifest=json.loads((native/'runtime-build/runtime-build-manifest.json').read_text());mlx=Path(sysconfig.get_paths()['purelib'])/'mlx'
        for source in manifest['inputs']['sources']:
            self.assertEqual(hashlib.sha256((ROOT/source['path']).read_bytes()).hexdigest(),source['sha256'],
                'C++ source/header ABI requires the matching native tree: '+source['path'])
        with tempfile.TemporaryDirectory(prefix='tc-qwen-convrot-src-') as d:
            binary=Path(d)/'probe'
            subprocess.run(['xcrun','clang++','-std=c++20','-O2','-Wall','-Wextra','-Werror','-Wno-missing-field-initializers','-mmacosx-version-min=26.2',
                '-DTC_TEST_BOUND_NATIVE_ID="'+manifest['runtime_build_id']+'"','-I',str(ROOT/'native/core'),'-I',str(ROOT/'bindings/c/include'),
                '-isystem',str(mlx/'include'),str(ROOT/'tests/native/qwen21_convrot_sources_test.cpp'),'-L'+str(native),'-lturbocider',
                '-L'+str(mlx/'lib'),'-lmlx','-Wl,-rpath,'+str(native),'-Wl,-rpath,'+str(mlx/'lib'),'-o',str(binary)],check=True,cwd=ROOT)
            command=[str(binary)]
            environment=None
            if os.environ.get('TURBOCIDER_TEST_CONVROT_PRIVATE_STAGE')=='1':
                sys.path.insert(0,str(ROOT/'tools/profiling/qwen_base'))
                import collect
                environment={k:v for k,v in os.environ.items() if not k.startswith('TURBOCIDER_')}
                environment.update(collect.environment('runtime','stock',6144,fixed_program=True))
                environment['TURBOCIDER_PRIVATE_ANE_DATA_PATH']='convrot_w8a8'
                environment['TURBOCIDER_PRIVATE_ANE_PREFETCH']='1'
                command.append(str(ROOT/'outputs/qwen-base-runtime-c1024-h4096-f12288-20261018.json'))
            subprocess.run(command,check=True,cwd=ROOT,env=environment,timeout=180)


if __name__=='__main__':unittest.main()
