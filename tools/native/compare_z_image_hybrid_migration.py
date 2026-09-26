#!/usr/bin/env python3
"""Frozen byte-exact legacy repeatability and typed-stage migration diagnostic.

No tolerance relaxation, release qualification, or performance comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import mlx.core as mx
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--legacy-first',type=Path,required=True)
p.add_argument('--legacy-second',type=Path,required=True)
p.add_argument('--stage',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def tensor(root,name):
    values=mx.load(str(root/'tensors'/f'{name}.safetensors'))
    return np.asarray(values['tensor'].astype(mx.float32))
def compare(x,y):
    require(x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all(),'invalid tensor comparison')
    xf=x.astype(np.float64).ravel();yf=y.astype(np.float64).ravel();delta=yf-xf
    return {'byte_equal':x.dtype==y.dtype and x.tobytes()==y.tobytes(),
            'relative_l2':float(np.linalg.norm(delta)/max(np.linalg.norm(xf),1e-30)),
            'max_abs':float(np.max(np.abs(delta)))}

roots=[a.legacy_first,a.legacy_second]
plans=[json.loads((r/'plan.json').read_text()) for r in roots]
stage=json.loads((a.stage/'plan.json').read_text())
for r,plan in zip(roots,plans):
    result=json.loads((r/'result.json').read_text())
    require(result['status']==0,'legacy request failed')
    require(plan['manifest_sha256']==stage['manifest_sha256'],'different Core ML partition manifest')
    require(plan['library_sha256']==stage['library_sha256'],'different runtime library')
    require(plan['initial_sha256']==stage['inputs']['initial.npy'],'different frozen noise')
    require(result['result']['actual_denoise_steps']==9,'incomplete legacy request')
    require(result['result']['valid_text_tokens']==47,'unexpected prompt token shape')
    require(result['result']['width']==512 and result['result']['height']==512,'unexpected image geometry')
require(plans[0]['request']['inputs']==plans[1]['request']['inputs'],'different prompts')
require(plans[0]['request']['execution']==plans[1]['request']['execution'],'different execution policies')
initial=np.load(a.stage/'complete/initial.npy',allow_pickle=False)
for r in roots:require(compare(initial,tensor(r,'z_latent_initial'))['byte_equal'],'different actual initial noise')
names=['z_latent_initial']+[f'z_latent_step_{i}' for i in range(1,10)]+['z_latent_final','z_decoded']
repeat={name:compare(tensor(roots[0],name),tensor(roots[1],name)) for name in names}
repeatable=all(v['byte_equal'] for v in repeat.values())
migration=compare(tensor(roots[0],'z_latent_final'),np.load(a.stage/'complete/final.npy',allow_pickle=False))
report={'scope':__doc__,'rule':'byte-exact, frozen before reference runs','legacy_repeatable':repeatable,
        'repeatability':repeat,'final_latent_migration':migration,
        'migration_passed':repeatable and migration['byte_equal'],
        'qualification':'NONE; one fixture, no complete-request lifecycle or release/performance evidence',
        'limitations':['Resident legacy and typed streaming do not share P/G/K/D/Q or retention; not same-plan P1',
                       'Legacy recomputes conditioning from the frozen prompt; its caption tensor is not directly observed'],
        'driver_sha256':sha(Path(__file__)),
        'plans':{str(r/'plan.json'):sha(r/'plan.json') for r in [*roots,a.stage]},
        'legacy_results':{str(r/'result.json'):sha(r/'result.json') for r in roots}}
with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({k:report[k] for k in ['legacy_repeatable','final_latent_migration','migration_passed']},indent=2))
