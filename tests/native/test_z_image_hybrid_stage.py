#!/usr/bin/env python3
"""Opt-in 9-step transformer execution and owner fault tests; no image qualification."""
import argparse
import hashlib
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
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
ROOT=Path(__file__).resolve().parents[2];native=args.native_dir.resolve();mlx=Path(sysconfig.get_paths()['purelib'])/'mlx'
plan={'scope':__doc__,'modes':['complete','cancel','unsafe'],'shape':[512,512],'caption':'64 zero BF16 rows of width 2560; synthetic, not a prompt encoding',
      'latent':'FP32 reshape(sin(arange(65536)*0.01)); BF16 cast at model input','steps':9,'schedule':'existing z_sigmas and Euler update','policy':'P1/G1/K2/D1/Q2',
      'qualification':'NONE: synthetic conditioning, no GPU-only quality/performance reference or VAE','library_sha256':hashlib.sha256((native/'libturbocider.dylib').read_bytes()).hexdigest(),
      'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'test_sha256':hashlib.sha256((ROOT/'tests/native/z_image_hybrid_stage_test.cpp').read_bytes()).hexdigest(),
      'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest()}
(args.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
with tempfile.TemporaryDirectory(prefix='tc-hybrid-stage-') as raw:
    root=Path(raw).resolve();bank=root/'bank';bank.mkdir();managed=root/'managed';managed.mkdir()
    manifest=args.manifest.resolve();payload=manifest.read_bytes();data=json.loads(payload)
    for entry in data['artifacts'].values():
        relative=Path(entry['int8_pc'])
        if relative.is_absolute() or '..' in relative.parts or relative.suffix!='.mlmodelc':raise ValueError('invalid compiled path')
        shutil.copytree(manifest.parent/relative,bank/relative,symlinks=True)
    if manifest.read_bytes()!=payload:raise ValueError('manifest changed')
    (bank/manifest.name).write_bytes(payload)
    binary=root/'test'
    subprocess.run(['clang++','-std=c++20','-O2','-Wall','-Wextra','-Werror','-DTURBOCIDER_ENABLE_TEST_HOOKS=1',
        '-I',str(ROOT/'native'),'-I',str(ROOT/'native/core'),'-isystem',str(mlx/'include'),
        str(ROOT/'tests/native/z_image_hybrid_stage_test.cpp'),'-L'+str(native),'-lturbocider','-L'+str(mlx/'lib'),'-lmlx',
        '-Wl,-rpath,'+str(native),'-Wl,-rpath,'+str(mlx/'lib'),'-o',str(binary)],check=True)
    for mode in plan['modes']:
        with (args.output/(mode+'.log')).open('w') as log:
            subprocess.run([str(binary),str(args.checkpoint.resolve()),str(bank/manifest.name),str(managed),str(args.output/mode),mode],
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
        print(mode,'PASS',flush=True)
