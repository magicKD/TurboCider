#!/usr/bin/env python3
"""Generate bounded reproducible Z-Turbo or synthetic ordinary-Qwen smoke specs.

The Qwen fixture is NOT a trained adapter, a style-quality test or LoRA speed
evidence. It only exercises nonzero gate/up/down targets under the base schedule.
"""
import argparse
import json
from pathlib import Path
import struct

from runtime_image_collect import ROOT, validate_spec
from collect import environment


def write_new(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def synthetic_adapter(path):
    import numpy as np
    header = {'__metadata__': {'fixture': 'synthetic-rank2-capability-only-not-trained-v1'}}
    payload = []
    offset = 0
    for projection, inputs, outputs in (('gate_layer', 4096, 12288), ('proj', 4096, 12288), ('out', 12288, 4096)):
        for suffix, shape in (('A', (2, inputs)), ('B', (outputs, 2))):
            values = (np.sin(np.arange(np.prod(shape), dtype=np.float32) * .137) * .008).astype('<f4')
            raw = (values.view('<u4') >> 16).astype('<u2').tobytes()
            key = f'transformer.transformer_blocks.0.img_mlp.{projection}.lora_{suffix}.weight'
            header[key] = dict(dtype='BF16', shape=shape, data_offsets=[offset, offset + len(raw)])
            offset += len(raw)
            payload.append(raw)
    text = json.dumps(header, separators=(',', ':')).encode()
    text += b' ' * ((-len(text)) % 8)
    with path.open('xb') as f:
        f.write(struct.pack('<Q', len(text)))
        f.write(text)
        for raw in payload:
            f.write(raw)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family', choices=('z-turbo', 'qwen-synthetic'), required=True)
    p.add_argument('--route', choices=('gpu', 'runtime'), required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--height', type=int, default=512)
    p.add_argument('--rows', type=int, default=1024)
    p.add_argument('--channels', type=int, choices=(4096, 5120, 6144), default=5120)
    p.add_argument('--requests', type=int, choices=(1, 3), default=3)
    p.add_argument('--shape-cycle', action='store_true', help='Z only:512,768x512,1024,512 in one owner; no pooled speed')
    a = p.parse_args()
    a.model = a.model.resolve(strict=True)
    a.output = a.output.absolute()
    if a.output.exists():
        p.error('use a fresh fixture directory')
    if not 128 <= a.rows <= 8192 or a.rows % 128 or not 256 <= a.channels <= 8192 or a.channels % 256:
        p.error('bounded fixed row/channel geometry required')
    if a.family == 'qwen-synthetic' and (a.width, a.height) != (512, 512):
        p.error('ordinary Qwen runtime LoRA smoke domain is512 only')
    if a.shape_cycle and a.family != 'z-turbo':
        p.error('shape cycle is currently Z T2I only')
    a.output.mkdir(parents=True)
    qwen = a.family == 'qwen-synthetic'
    model = 'qwen-image-2.1' if qwen else 'z-image-turbo'
    h, f = (4096, 12288) if qwen else (3840, 10240)
    descriptor = a.output / 'runtime.json'
    write_new(descriptor, dict(schema_version=1, descriptor_version=1, backend='private_runtime_shape',
        kind='swiglu', layout='out_in', biases=False, lora_inputs=qwen, rows=a.rows, hidden=h, width=f,
        tile_k=512 if not qwen else 4096, tile_n=512))
    flags = environment(a.route, 'stock', a.channels if a.route == 'runtime' else 4096,
                        fixed_program=a.route == 'runtime',
                        ffn_activation='exp' if qwen or a.route == 'gpu' else 'sigmoid-complement')
    if qwen and a.route == 'runtime':
        flags['TURBOCIDER_QWEN21_RUNTIME_BASE_LORA'] = '1'
    if qwen:
        adapter = a.output / 'SYNTHETIC-ordinary-rank2-not-trained.safetensors'
        synthetic_adapter(adapter)
        files = [a.model / x for x in ('diffusion_models/qwen_image_2.1_bf16.safetensors',
            'text_encoders/qwen3vl_8b_bf16.safetensors', 'vae/qwen_image_2.1_vae_bf16.safetensors', 'processor/tokenizer.json')]
        files.append(adapter)
    else:
        files = [a.model / x for x in ('split_files/diffusion_models/z_image_turbo_bf16.safetensors',
            'split_files/text_encoders/qwen_3_4b.safetensors', 'split_files/vae/ae.safetensors', 'tokenizer/tokenizer.json')]
    requests = []
    shapes = [(512, 512), (768, 512), (1024, 1024), (512, 512)] if a.shape_cycle else [(a.width, a.height)] * a.requests
    for i, (width, height) in enumerate(shapes):
        seed = 73 if a.shape_cycle else (73, 74, 73)[i]
        q = dict(model=model, operation='image.generate',
            prompt='A blue ceramic teapot on a wooden table, soft daylight, realistic photograph.',
            width=width, height=height, steps=20 if qwen else 8, seed=seed, frames=1, audio=False,
            execution='gpu_ane' if a.route == 'runtime' else 'gpu', residency='resident', allow_approximation=True)
        if a.route == 'runtime':
            q.update(hybrid_mlp_mode='runtime', ane_manifest=str(descriptor))
        if qwen:
            q.update(model_variant='qwen-image-2.1', qwen21_dit_cache='off', qwen21_reference_size=1024,
                     lora_strategy='inference_time', loras=[dict(path=str(adapter), role='transformer', strength=1.)])
        else:
            q['dynamic_text'] = True
        requests.append(q)
    spec = dict(model=model, model_path=str(a.model), environment=flags, requests=requests,
                source_files=[str(x.resolve(strict=True)) for x in files], expected_lora_projections=3 if qwen else 0,
                fixture_scope='synthetic non-trained capability only' if qwen else 'original BF16 Z Turbo eight-step T2I')
    validate_spec(spec)
    write_new(a.output / 'spec.json', spec)
    print(a.output / 'spec.json')


if __name__ == '__main__':
    main()
