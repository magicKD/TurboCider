#!/usr/bin/env python3
"""Native bundle semantic validation with stub files, not Core ML execution."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--real-checkpoint',type=Path)
parser.add_argument('--real-manifest',type=Path)
args=parser.parse_args()
if bool(args.real_checkpoint) != bool(args.real_manifest):
    parser.error('both real paths are required')

ROOT = Path(__file__).resolve().parents[2]
sources = ['tests/native/coreml_bundle_test.cpp', 'native/models/z_image/coreml_bundle.mm',
           'native/models/z_image/coreml_generation.cpp', 'native/runtime/streaming/source_lease.cpp',
           'native/runtime/streaming/canonical_encoding.cpp', 'native/runtime/memory_manifest.cpp',
           'native/core/common.cpp', 'native/core/json_keys.cpp']
flags = ['-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror', '-fobjc-arc', '-I', str(ROOT/'native'), '-I', str(ROOT/'native/core')]
if os.environ.get('TC_STREAMING_SANITIZER'):
    flags += ['-fsanitize='+os.environ['TC_STREAMING_SANITIZER'], '-fno-omit-frame-pointer']
with tempfile.TemporaryDirectory(prefix='tc-bundle-') as raw:
    root = Path(raw).resolve()
    parent = root/'parent'; parent.write_bytes(b'independent parent fixture')
    managed = root/'managed'; managed.mkdir()
    sha = hashlib.sha256(parent.read_bytes()).hexdigest()
    source = {'checkpoint_sha256':sha, 'checkpoint_bytes':parent.stat().st_size, 'blocks':list(range(32)), 'checkpoint':'/old/install/ignored'}
    shape = {'K':3840,'N':3840,'mlp_width':10240,'ane_mlp_start':0,'ane_mlp_end':5120,'buckets':[1088],'activation_scale':8,'output_scale':32}
    export = dict(source, owner='turbocider.z_image.coreml.v1', recipe=1, checkpoint_format='safetensors',convrot_mode='none',hidden=3840,mlp_width=10240,ane_mlp_start=0,ane_mlp_end=5120,bucket=1088,activation_scale=8,output_scale=32,variant='int8_pc')
    base = {'schema_version':2,'shape':shape,'source':source,'export_identity':export,'functions':{'1088':'main'},'artifacts':{str(i):{'int8_pc':f'{i}.mlmodelc'} for i in range(32)}}
    cases = [('pass-original',base),('pass-relocated',copy.deepcopy(base))]
    mutations = [ ('parent',('source','checkpoint_sha256'),'0'*64), ('export-parent',('export_identity','checkpoint_sha256'),'0'*64),
        ('bytes',('source','checkpoint_bytes'),3),('boolean',('shape','ane_mlp_start'),False),('fraction',('shape','ane_mlp_end'),5120.5),
        ('partition',('export_identity','ane_mlp_end'),2560),('geometry',('shape','K'),1),('scale',('shape','output_scale'),16),
        ('precision',('export_identity','variant'),'unknown'),('bucket',('shape','buckets'),[1024]),('lora',('source','loras'),[{}]),
        ('map',('source','blocks'),list(reversed(range(32)))),('owner',('export_identity','owner'),'wrong'),
        ('duplicate-model',('artifacts','1'),{'int8_pc':'0.mlmodelc'}),('escape',('artifacts','1'),{'int8_pc':'../0.mlmodelc'}),
        ('absolute',('artifacts','1'),{'int8_pc':'/0.mlmodelc'}),('function',('functions','1088'),'other') ]
    for label,(section,key),value in mutations:
        d=copy.deepcopy(base); d[section][key]=value; cases.append(('fail-'+label,d))
    d=copy.deepcopy(base); del d['artifacts']['31']; cases.append(('fail-missing',d))
    paths=[]
    for label,data in cases:
        bank=root/label; bank.mkdir(); paths.append(bank)
        (bank/'manifest.json').write_text(json.dumps(data))
        for i in range(32):
            model=bank/f'{i}.mlmodelc'; model.mkdir(); (model/'stub').write_bytes(bytes([i]))
    bank=root/'fail-duplicate-json'; bank.mkdir(); paths.append(bank)
    text=json.dumps(base); (bank/'manifest.json').write_text(text.replace('"schema_version": 2','"schema_version": 2, "schema_version": 2'))
    bank=root/'fail-empty-model'; bank.mkdir(); paths.append(bank)
    (bank/'manifest.json').write_text(json.dumps(base)); (bank/'0.mlmodelc'/'empty').mkdir(parents=True)
    binary=root/'test'
    subprocess.run(['clang++',*flags,*(str(ROOT/s) for s in sources),'-framework','Foundation','-o',str(binary)],check=True)
    subprocess.run([str(binary),str(parent),str(managed),*map(str,paths)],check=True,timeout=180)

    if args.real_checkpoint:
        # Research fixture assembly, not a production installation interface.
        # Preserve manifest bytes/relative model paths; omit cache management files.
        manifest=args.real_manifest.resolve()
        payload=manifest.read_bytes(); document=json.loads(payload)
        bank=root/'real-bank'; bank.mkdir()
        for entry in document['artifacts'].values():
            relative=Path(entry['int8_pc'])
            if relative.is_absolute() or '..' in relative.parts or relative.suffix != '.mlmodelc':
                raise ValueError('invalid compiled artifact path')
            shutil.copytree(manifest.parent/relative,bank/relative,symlinks=True)
        if manifest.read_bytes() != payload:
            raise ValueError('manifest changed during fixture assembly')
        (bank/manifest.name).write_bytes(payload)
        subprocess.run([str(binary),'--real',str(args.real_checkpoint.resolve()),str(bank/manifest.name),str(managed)],check=True,timeout=300)
