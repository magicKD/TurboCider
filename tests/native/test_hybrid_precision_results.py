#!/usr/bin/env python3
"""Native serialization regression for loaded versus uninspected hybrid precision."""
import argparse
from pathlib import Path
import subprocess
import sysconfig
import tempfile
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--native-dir',type=Path,required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[2]
mlx=Path(sysconfig.get_paths()['purelib'])/'mlx';native=a.native_dir.resolve()
with tempfile.TemporaryDirectory(prefix='tc-hybrid-precision-') as raw:
    binary=Path(raw)/'test'
    subprocess.run(['clang++','-std=c++20','-O2','-Wall','-Wextra','-Werror','-fobjc-arc',
        '-mmacosx-version-min=26.2','-I',str(root/'native'),'-I',str(root/'native/core'),
        '-isystem',str(mlx/'include'),str(root/'tests/native/hybrid_precision_results_test.mm'),
        '-L'+str(native),'-lturbocider','-L'+str(mlx/'lib'),'-lmlx','-framework','Foundation',
        '-Wl,-rpath,'+str(native),'-Wl,-rpath,'+str(mlx/'lib'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
