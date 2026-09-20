#!/usr/bin/env python3
"""Metadata binding using native verified parent and stub compiled trees; no prediction."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native-dir',type=Path,required=True)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--source-manifest',type=Path,required=True)
args=parser.parse_args();native=args.native_dir.resolve()
with tempfile.TemporaryDirectory(prefix='tc-hybrid-layout-') as raw:
    root=Path(raw).resolve();managed=root/'managed';managed.mkdir()
    other=root/'other';other.write_bytes(b'independently verified wrong parent')
    base=json.loads(args.source_manifest.read_text())
    base['artifacts']={str(i):{'int8_pc':f'{i}.mlmodelc'} for i in range(32)}
    banks=[]
    for i,label in enumerate(['baseline','relocated','precision','payload','wrong-parent']):
        d=copy.deepcopy(base);bank=root/label;bank.mkdir();banks.append(bank)
        if i==2:d['export_identity']['variant']='fp16'
        if i==4:
            for section in ('source','export_identity'):
                d[section]['checkpoint_sha256']=hashlib.sha256(other.read_bytes()).hexdigest()
                d[section]['checkpoint_bytes']=other.stat().st_size
        (bank/'manifest.json').write_text(json.dumps(d,sort_keys=True))
        for block in range(32):
            model=bank/f'{block}.mlmodelc';model.mkdir()
            (model/'stub').write_bytes(bytes([block,1 if i==3 else 0]))
    flags=['-std=c++20','-O2','-Wall','-Wextra','-Werror','-I',str(ROOT/'native'),'-I',str(ROOT/'native/core')]
    if os.environ.get('TC_STREAMING_SANITIZER'):
        flags+=['-fsanitize='+os.environ['TC_STREAMING_SANITIZER'],'-fno-omit-frame-pointer']
    binary=root/'test'
    subprocess.run(['clang++',*flags,str(ROOT/'tests/native/z_image_hybrid_layout_test.cpp'),
        str(ROOT/'native/models/z_image/hybrid_layout.cpp'),'-L'+str(native),'-lturbocider',
        '-Wl,-rpath,'+str(native),'-o',str(binary)],check=True)
    subprocess.run([str(binary),str(args.checkpoint.resolve()),str(other),str(managed),*map(str,banks)],check=True,timeout=180)
