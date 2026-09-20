#!/usr/bin/env python3
"""Report one hybrid-vs-GPU sample; errors do not grant release qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mlx.core as mx
from PIL import Image
p=argparse.ArgumentParser(description=__doc__);p.add_argument('fixture',type=Path);a=p.parse_args();r=a.fixture

def gpu(name):
 tensors=mx.load(str(r/'gpu-reference/tensors'/f'{name}.safetensors'))
 return np.asarray(tensors['tensor'].astype(mx.float32))

def errors(reference,candidate):
 x=np.asarray(reference,dtype=np.float64).ravel();y=np.asarray(candidate,dtype=np.float64).ravel()
 assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
 nx=np.linalg.norm(x);ny=np.linalg.norm(y);d=y-x
 return {'relative_l2':float(np.linalg.norm(d)/max(nx,1e-30)),
         'cosine':float(np.dot(x,y)/max(nx*ny,1e-30)),
         'rmse':float(np.sqrt(np.mean(d*d))),'max_abs':float(np.abs(d).max())}

initial=np.load(r/'inputs/initial.npy');assert np.array_equal(initial,gpu('z_latent_initial'))
reference=json.loads((r/'gpu-reference/result.json').read_text());assert reference['status']==0
hybrid_latent=np.load(r/'stage/complete/final.npy');gpu_latent=gpu('z_latent_final')
hybrid_rgb=np.asarray(Image.open(r/'decoded/image.png').convert('RGB'),dtype=np.float32)/255
reference_rgb=np.asarray(Image.open(r/'gpu-reference/image.png').convert('RGB'),dtype=np.float32)/255
report={'scope':__doc__,'initial_noise_byte_equal':True,'samples':1,
 'latent':errors(gpu_latent,hybrid_latent),
 'decoded':errors(gpu('z_decoded'),np.load(r/'decoded/decoded.npy')),
 'rgb_0_1':errors(reference_rgb,hybrid_rgb),
 'image_sha256':{name:hashlib.sha256((r/path).read_bytes()).hexdigest() for name,path in [('hybrid','decoded/image.png'),('gpu','gpu-reference/image.png')]},
 'qualification':'No pass verdict: exploratory single sample; original INT8 branch 29 gate remains failed',
 'performance':'Not comparable: staged hybrid execution and concurrent builds; no speed claim',
 'gpu_source_verification':json.loads((r/'gpu-reference/source-verification.json').read_text())}
(r/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['latent','decoded','rgb_0_1']},indent=2))
