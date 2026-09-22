"""Pinned PE processor/tokenizer/mRoPE integration (not quality acceptance)."""
import argparse
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace, MethodType

import numpy as np
from PIL import Image
from safetensors.torch import load_file, save_file
import torch
import transformers
from transformers import AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--images', type=Path, nargs='+', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--prefill', action='store_true')
    a = parser.parse_args()
    assert transformers.__version__ == '5.4.0'
    torch.set_num_threads(2)
    a.output_directory.mkdir(parents=True, exist_ok=False)
    processor = AutoProcessor.from_pretrained(a.weights, local_files_only=True)
    raw, images = {}, []
    for i, path in enumerate(a.images):
        # File decoding is reference fixture preparation, not native inference.
        with Image.open(path) as im:
            im = im.convert('RGBA')
            raw[f'pixels{i}'] = torch.from_numpy(np.asarray(im).copy())[None]
            im = im.convert('RGB')
        if im.width * im.height > 1024 * 1024:
            scale = math.sqrt(1024 * 1024 / (im.width * im.height))
            im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.Resampling.LANCZOS)
        images.append(im)
    system = (a.weights / 'system_prompt.txt').read_text().strip()
    messages = [dict(role='system', content=[dict(type='text', text=system)]),
        dict(role='user', content=[dict(type='image', image=im) for im in images] + [dict(type='text', text=a.prompt)])]
    expected = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=True, return_dict=True, return_tensors='pt')
    geometry = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)))
    geometry.get_vision_position_ids = MethodType(Qwen3_5Model.get_vision_position_ids, geometry)
    types = (expected['input_ids'] == 248056).long()
    positions, delta = Qwen3_5Model.get_rope_index(geometry, expected['input_ids'], types, expected['image_grid_thw'])
    inp, out = a.output_directory / 'pixels.safetensors', a.output_directory / 'native.safetensors'
    save_file(raw, str(inp))
    cmd = [str(a.probe.resolve()), str(a.weights.resolve()), str(inp), a.prompt, str(out)]
    if a.prefill:
        cmd.append('--prefill')
    with (a.output_directory / 'events.log').open('w') as log:
        subprocess.run(cmd, stdout=log, stderr=log, check=True)
    actual = load_file(str(out))
    assert torch.equal(actual['ids'].long(), expected['input_ids']), 'expanded input IDs differ'
    assert torch.equal(actual['grids'].long(), expected['image_grid_thw']), 'patch grids differ'
    assert torch.equal(actual['token_types'].long(), types), 'modality token types differ'
    assert torch.equal(actual['positions'].long(), positions[:, 0]), 'mRoPE positions differ'
    assert actual['rope_delta'].item() == delta.item(), 'decode position delta differs'
    patches = torch.cat([actual[f'patches{i}'] for i in range(len(images))])
    error = float((patches - expected['pixel_values']).abs().max())
    assert error < 1e-6 and torch.isfinite(actual['embeddings']).all()
    if a.prefill:
        assert torch.isfinite(actual['last_hidden']).all() and torch.isfinite(actual['logits']).all()
        assert torch.isfinite(actual['decode_logits']).all()
    report = dict(passed=True, images=len(images), tokens=actual['ids'].numel(),
        grids=actual['grids'].tolist(), rope_delta=actual['rope_delta'].item(), patch_max_abs=error,
        prefill=a.prefill, checkpoint_directory=str(a.weights), quality_accepted=False,
        transformers_version=transformers.__version__, torch_version=torch.__version__,
        scope='exact processor/tokenizer/mRoPE and finite native conditioning/prefill only; not file decoder or rewrite acceptance')
    (a.output_directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
