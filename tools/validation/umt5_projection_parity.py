#!/usr/bin/env python3
"""Same-input UMT5 softmax, aggregation and projection diagnostic.

Development Python MLX vs PyTorch MPS experiment, not a linked native or video
quality gate. Loads only the first attention output weight, not the full model.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True, help='text_encoder directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('output already exists')
    import mlx.core as mx
    import numpy as np
    import torch
    from safetensors import safe_open

    name = 'encoder.block.0.layer.0.SelfAttention.o.weight'
    index = args.weights / 'model.safetensors.index.json'
    shard = (json.loads(index.read_text())['weight_map'][name]
             if index.is_file() else 'model.safetensors')
    source = (args.weights / shard).resolve(strict=True)
    if not source.is_relative_to(args.weights.resolve(strict=True)):
        raise ValueError('shard escapes weights directory')
    with safe_open(source, framework='pt', device='cpu') as checkpoint:
        weight = checkpoint.get_tensor(name).to(torch.bfloat16)
    w = mx.array(weight.float().numpy()).astype(mx.bfloat16)
    tw = weight.to('mps')

    def to_torch(value):
        return torch.from_numpy(np.asarray(value.astype(mx.float32)).copy()).to(
            'mps', dtype=torch.bfloat16)

    def array(value):
        if isinstance(value, torch.Tensor):
            return value.detach().float().cpu().numpy().astype(np.float64)
        return np.asarray(value.astype(mx.float32)).astype(np.float64)

    def metrics(left, right):
        x, y = array(left), array(right)
        if x.shape != y.shape:
            raise ValueError('comparison shape mismatch')
        delta = x - y
        return {'exact': bool(np.array_equal(x, y)),
                'finite': bool(np.isfinite(x).all() and np.isfinite(y).all()),
                'mismatched': int(np.count_nonzero(delta)), 'elements': int(x.size),
                'rmse': float(np.sqrt(np.mean(delta * delta))),
                'max_abs': float(np.max(np.abs(delta)))}

    report = {'scope': 'Python same-input operators; not native/full-video parity',
              'torch_version': torch.__version__, 'inputs': {}}
    with torch.inference_mode():
        for origin in ['reference', 'native']:
            stages = mx.load(str(args.trace / (origin + '.safetensors')))
            scores = stages['first_attention_scores']
            probabilities = stages['first_attention_probabilities']
            value = stages['first_v']
            x = stages['first_attended']
            if any(t.dtype != mx.bfloat16 for t in [scores, probabilities, value, x]):
                raise ValueError('expected captured BF16 tensors')
            batch, heads, rows, keys = scores.shape
            if batch != 1 or rows != keys or value.shape[1] != keys:
                raise ValueError('expected self-attention trace')
            values = value.reshape(batch, keys, heads, -1).transpose(0, 2, 1, 3)
            softmax = mx.softmax(scores.astype(mx.float32), axis=-1).astype(mx.bfloat16)
            torch_softmax = torch.softmax(to_torch(scores).float(), dim=-1).to(torch.bfloat16)
            aggregation = mx.matmul(probabilities, values)
            torch_aggregation = torch.matmul(to_torch(probabilities), to_torch(values))
            projection = mx.matmul(x, w.T)
            torch_projection = torch.nn.functional.linear(to_torch(x), tw)
            fp32_projection = mx.matmul(x.astype(mx.float32), w.astype(mx.float32).T).astype(mx.bfloat16)
            report['inputs'][origin] = {
                'softmax_same_input': metrics(softmax, torch_softmax),
                'aggregation_same_input': metrics(aggregation, torch_aggregation),
                'projection_same_input': metrics(projection, torch_projection),
                'fp32_projection_vs_torch': metrics(fp32_projection, torch_projection),
                'mlx_projection_vs_capture': metrics(projection, stages['first_attention_output']),
                'torch_projection_vs_capture': metrics(torch_projection, stages['first_attention_output']),
            }
    with args.output.open('x') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not all(m['finite'] for stages in report['inputs'].values() for m in stages.values()):
        raise SystemExit('non-finite diagnostic output')


if __name__ == '__main__':
    main()
