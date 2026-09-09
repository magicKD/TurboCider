"""Compare native UMT5 with the installed Transformers BF16/MPS encoder.

This is development tooling, not a production text-encoding route. Reports
numerical differences without asserting that a cosine score proves image or
video quality. Exact dtype, bucket, embedding and padding contracts are gates.
"""
import argparse
import gc
import json
from pathlib import Path
import subprocess
from unittest.mock import patch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--length', type=int, default=32)
    p.add_argument('--prompt', default='A red fox walks slowly across a snowy field.')
    p.add_argument('--trace-first-block', action='store_true')
    a = p.parse_args()
    import numpy as np
    import torch
    import mlx.core as mx
    from transformers import AutoTokenizer, UMT5EncoderModel

    a.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(a.model / 'tokenizer', local_files_only=True)
    inputs = tokenizer([a.prompt], padding='max_length', max_length=a.length,
                       truncation=True, return_tensors='pt')
    ids = inputs.input_ids
    valid = int(inputs.attention_mask.sum())
    mx.save_safetensors(str(a.output / 'input.safetensors'), {
        'ids': mx.array(ids[0].numpy().astype(np.int32)), 'valid': mx.array(valid, dtype=mx.int32)})
    model = UMT5EncoderModel.from_pretrained(a.model / 'text_encoder', dtype=torch.bfloat16,
                                           local_files_only=True).to('mps').eval()
    dtype_map = {name: str(value.dtype) for name, value in model.named_parameters()}
    reference = {}

    def record(name, value):
        reference[name] = mx.array(value.detach().float().cpu().numpy()).astype(mx.bfloat16)
        mx.eval(reference[name])

    handles = []
    if a.trace_first_block:
        block = model.encoder.block[0]
        attn = block.layer[0].SelfAttention
        ff = block.layer[1].DenseReluDense
        modules = {'norm_attention': block.layer[0].layer_norm,
                   'q': attn.q, 'k': attn.k, 'v': attn.v, 'attention_output': attn.o,
                   'norm_ffn': block.layer[1].layer_norm,
                   'gate_input': ff.wi_0, 'gate_activation': ff.act,
                   'up': ff.wi_1, 'ffn_output': ff.wo}
        for name, module in modules.items():
            handles.append(module.register_forward_hook(
                lambda module, args, output, n=name: record('first_' + n, output)))
        handles.append(attn.o.register_forward_pre_hook(
            lambda module, args: record('first_attended', args[0])))
        handles.append(attn.register_forward_hook(
            lambda module, args, output: record('first_attention_probabilities', output[1])))
    for index, block in enumerate(model.encoder.block):
        handles.append(block.register_forward_hook(
            lambda module, args, output, i=index: record(f'block_{i}', output[0])))
    original_softmax = torch.nn.functional.softmax

    def traced_softmax(value, *args, **kwargs):
        # The first softmax is the first encoder attention. Capture the actual
        # input to the reference operator, not a reconstructed approximation.
        if a.trace_first_block and 'first_attention_scores' not in reference:
            if value.ndim != 4 or value.shape[-2:] != (a.length, a.length):
                raise AssertionError('unexpected reference attention softmax geometry')
            record('first_attention_scores', value)
        return original_softmax(value, *args, **kwargs)

    with torch.inference_mode(), patch.object(torch.nn.functional, 'softmax', traced_softmax):
        record('embedding', model.shared(ids.to('mps')))
        positions = torch.arange(a.length)
        buckets = model.encoder.block[0].layer[0].SelfAttention._relative_position_bucket(
            positions[None, :] - positions[:, None])
        reference['buckets'] = mx.array(buckets.numpy().astype(np.int32))
        output = model(input_ids=ids.to('mps'), attention_mask=inputs.attention_mask.to('mps')).last_hidden_state
        output[:, valid:] = 0
        record('output', output)
    for handle in handles:
        handle.remove()
    if a.trace_first_block:
        del modules, module, attn, ff
    del model, output, block, handles
    gc.collect()
    torch.mps.empty_cache()
    mx.save_safetensors(str(a.output / 'reference.safetensors'), reference)
    command = [str(a.probe.resolve()), str((a.model / 'text_encoder').resolve()),
                    str((a.output / 'input.safetensors').resolve()),
                    str((a.output / 'native.safetensors').resolve())]
    if a.trace_first_block:
        command.append('--trace-first-block')
    subprocess.run(command, check=True)
    candidate = mx.load(str(a.output / 'native.safetensors'))
    if set(candidate) != set(reference):
        raise AssertionError('stage sets differ')
    report = {'parameter_dtypes': sorted(set(dtype_map.values())), 'valid_tokens': valid, 'stages': {}}
    for name, ref in reference.items():
        got = candidate[name]
        if ref.shape != got.shape or ref.dtype != got.dtype:
            raise AssertionError(f'{name}: shape/dtype mismatch')
        x = np.asarray(ref.astype(mx.float32)).astype(np.float64)
        y = np.asarray(got.astype(mx.float32)).astype(np.float64)
        delta = x - y
        report['stages'][name] = {
            'exact': bool(np.array_equal(x, y)), 'finite': bool(np.isfinite(x).all() and np.isfinite(y).all()),
            'max_abs': float(np.max(np.abs(delta))), 'rmse': float(np.sqrt(np.mean(delta ** 2))),
            'cosine': float(np.sum(x * y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30)),
        }
    report['padding_zero'] = bool(np.all(np.asarray(candidate['output'][:, valid:].astype(mx.float32)) == 0))
    (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not (report['padding_zero'] and report['stages']['buckets']['exact'] and
            report['stages']['embedding']['exact'] and all(v['finite'] for v in report['stages'].values())):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
