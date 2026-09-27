#!/usr/bin/env python3
"""Metal arithmetic test, with real Z FFN geometry and independent sparse oracle."""
import argparse
from pathlib import Path
import subprocess
import sysconfig
import tempfile
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native-dir',type=Path,required=True)
parser.add_argument('--compile-helper',action='store_true',help='compile helper source separately against an older support library')
args=parser.parse_args()
native=args.native_dir.resolve(); mlx=Path(sysconfig.get_paths()['purelib'])/'mlx'
with tempfile.TemporaryDirectory(prefix='tc-hybrid-math-') as raw:
    binary=Path(raw)/'test'
    subprocess.run(['clang++','-std=c++20','-O2','-Wall','-Wextra','-Werror','-I',str(ROOT/'native'),'-I',str(ROOT/'native/core'),
        '-isystem',str(mlx/'include'),str(ROOT/'tests/native/z_image_hybrid_math_test.cpp'),
        *([str(ROOT/'native/models/z_image/hybrid_math.cpp')] if args.compile_helper else []),'-L'+str(native),'-lturbocider','-L'+str(mlx/'lib'),'-lmlx',
        '-Wl,-rpath,'+str(native),'-Wl,-rpath,'+str(mlx/'lib'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True,timeout=120)
