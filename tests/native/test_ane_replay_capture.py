"""CPU-only full-input capture guards; no model or hardware invocation."""
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]

@unittest.skipUnless(sys.platform == 'darwin', 'Apple MLX CPU')
class ReplayCaptureTest(unittest.TestCase):
    def test_original_bf16_and_bounded_completion(self):
        mlx=Path(sysconfig.get_paths()['purelib'])/'mlx'
        native=Path(os.environ.get('TURBOCIDER_TEST_NATIVE_DIR',ROOT/'build/native')).resolve()
        with tempfile.TemporaryDirectory(prefix='tc-ffn-replay-') as tmp:
            binary=Path(tmp)/'probe'
            build=subprocess.run(['xcrun','clang++','-std=c++20','-O2','-Wall','-Wextra','-Werror',
                '-mmacosx-version-min=26.2',str(ROOT/'tests/native/ane_replay_capture_test.cpp'),
                '-I'+str(ROOT/'native/core'),'-isystem',str(mlx/'include'),'-L'+str(native),'-lturbocider',
                '-L'+str(mlx/'lib'),'-lmlx','-Wl,-rpath,'+str(native),'-Wl,-rpath,'+str(mlx/'lib'),
                '-o',str(binary)],capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stderr)
            env={k:v for k,v in os.environ.items() if not k.startswith('TURBOCIDER_')}
            env['TURBOCIDER_TEST_MLX_CPU']='1'
            run=subprocess.run([str(binary),tmp],env=env,capture_output=True,text=True,timeout=30)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertIn('PASS replay',run.stdout)
