#!/usr/bin/env python3
"""Opt-in real-weight Core ML/Metal FFN joins; zero oracle, no full-image qualification."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sysconfig
import tempfile
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native-dir',type=Path,required=True)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--manifest',type=Path,required=True)
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[2]
mlx=Path(sysconfig.get_paths()['purelib'])/'mlx'; native=args.native_dir.resolve()
with tempfile.TemporaryDirectory(prefix='tc-verified-join-') as raw:
    root=Path(raw).resolve(); bank=root/'bank'; bank.mkdir(); managed=root/'managed'; managed.mkdir()
    manifest=args.manifest.resolve(); payload=manifest.read_bytes(); data=json.loads(payload)
    for entry in data['artifacts'].values():
        relative=Path(entry['int8_pc'])
        if relative.is_absolute() or '..' in relative.parts or relative.suffix!='.mlmodelc':
            raise ValueError('invalid compiled path')
        shutil.copytree(manifest.parent/relative,bank/relative,symlinks=True)
    if manifest.read_bytes()!=payload: raise ValueError('manifest changed during fixture assembly')
    (bank/manifest.name).write_bytes(payload)
    binary=root/'test'
    subprocess.run(['clang++','-std=c++20','-O2','-Wall','-Wextra','-Werror',
        '-I',str(ROOT/'native'),'-I',str(ROOT/'native/core'),'-isystem',str(mlx/'include'),
        str(ROOT/'tests/native/z_image_verified_join_test.cpp'),'-L'+str(native),'-lturbocider',
        '-L'+str(mlx/'lib'),'-lmlx','-Wl,-rpath,'+str(native),'-Wl,-rpath,'+str(mlx/'lib'),'-o',str(binary)],check=True)
    subprocess.run([str(binary),str(args.checkpoint.resolve()),str(bank/manifest.name),str(managed)],check=True,timeout=300)
