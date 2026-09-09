"""Real-checkpoint Wan DiT stage parity. Python is a development oracle only.

Run using an environment able to import fastvideo, torch and mlx. The reference
repository is explicit and is never copied into the native runtime/package.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--latent-shape', type=int, nargs=3, default=[3, 8, 8])
    parser.add_argument('--text-tokens', type=int, default=16)
    parser.add_argument('--rollout', action='store_true',
                        help='Also compare sequential three-step eager/compiled sampling')
    args = parser.parse_args()
    # Explicitly select the dense reference path, independent of a developer's
    # experimental process settings. Production C++ reads none of these.
    os.environ['FASTVIDEO_MLX_WINDOW'] = '0'
    os.environ['FASTVIDEO_MLX_FAST_NORM'] = '0'
    os.environ['FASTVIDEO_MLX_COMPILE'] = '0'
    sys.path.insert(0, str(args.reference_root.resolve(strict=True)))
    import mlx.core as mx
    import numpy as np
    import torch
    from fastvideo.mlx_runtime.checkpoint import load_mlx_dit_checkpoint
    def rotary(sizes):
        # Isolated copy of the reference's public math; avoids importing the
        # distributed FastVideo package and its unrelated cloudpickle stack.
        dims = [44, 42, 42]
        axes = [torch.arange(n, dtype=torch.float32) for n in sizes]
        grid = torch.meshgrid(*axes, indexing='ij')
        cos, sin = [], []
        for pos, dim in zip(grid, dims):
            freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            angle = torch.outer(pos.reshape(-1), freq)
            cos.append(torch.repeat_interleave(angle.cos(), 2, dim=-1))
            sin.append(torch.repeat_interleave(angle.sin(), 2, dim=-1))
        return torch.cat(cos, dim=1), torch.cat(sin, dim=1)

    args.output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    shape = (1, 16, *args.latent_shape)
    cosine, sine = rotary((shape[2], shape[3] // 2, shape[4] // 2))
    inputs = {
        'latent': mx.array(rng.standard_normal(shape).astype(np.float16)),
        'text': mx.array(rng.standard_normal((1, args.text_tokens, 4096)).astype(np.float16)),
        'timestep': mx.array([757.], dtype=mx.float32),
        'cosine': mx.array(cosine.numpy()),
        'sine': mx.array(sine.numpy()),
    }
    input_path = args.output / 'input.safetensors'
    if args.rollout:
        for index in range(2):
            inputs[f'renoise_{index}'] = mx.array(rng.standard_normal(shape).astype(np.float32))
    mx.save_safetensors(str(input_path), inputs)
    model = load_mlx_dit_checkpoint(args.checkpoint.resolve(), compile=False)
    x = model.patch_embed(inputs['latent'])
    time, modulation, context = model.condition(inputs['timestep'], inputs['text'])
    reference = {'patch': x, 'time': time, 'modulation': modulation, 'context': context}
    mx.eval(*reference.values())
    for index, block in enumerate(model.blocks):
        x = block(x, context, modulation, (inputs['cosine'], inputs['sine']))
        mx.eval(x)
        reference[f'block_{index}'] = x
    reference['output'] = model.output(x, time, batch=1, frames=shape[2], height=shape[3], width=shape[4])
    mx.eval(reference['output'])
    model._enable_compile = True
    reference['compiled_output'] = model(inputs['latent'], inputs['text'], inputs['timestep'],
                                        (inputs['cosine'], inputs['sine']))
    mx.eval(reference['compiled_output'])
    from fastvideo.mlx_runtime.sampling import MLXDMDSchedule, dmd_step
    training = np.linspace(1, 1000, 1000, dtype=np.float32)[::-1].copy()
    sigma = training / np.float32(1000)
    sigma = np.float32(8) * sigma / (np.float32(1) + np.float32(7) * sigma)
    schedule = MLXDMDSchedule(sigma.astype(np.float64), (sigma * np.float32(1000)).astype(np.float64))
    timesteps = [1000., 757., 522.]
    reference['sigmas'] = mx.array([schedule.sigma_for(t) for t in timesteps], dtype=mx.float32)
    for index, timestep in enumerate(timesteps):
        reference[f'dmd_{index}'] = dmd_step(
            latents=inputs['latent'], noise_input_latent=inputs['latent'],
            pred_noise=reference['output'], schedule=schedule, timestep=timestep,
            next_timestep=timesteps[index + 1] if index < 2 else None, noise=inputs['latent'])
        mx.eval(reference[f'dmd_{index}'])
    if args.rollout:
        for compiled in [False, True]:
            model._enable_compile = compiled
            latent = inputs['latent']
            mode = 'compiled' if compiled else 'eager'
            for index, timestep in enumerate(timesteps):
                prediction = model(latent, inputs['text'], mx.array([timestep], dtype=mx.float32),
                                   (inputs['cosine'], inputs['sine']))
                latent = dmd_step(
                    latents=latent.astype(mx.float32), noise_input_latent=latent.astype(mx.float32),
                    pred_noise=prediction.astype(mx.float32), schedule=schedule, timestep=timestep,
                    next_timestep=timesteps[index + 1] if index < 2 else None,
                    noise=inputs[f'renoise_{index}'] if index < 2 else None).astype(mx.float16)
                mx.eval(latent)
                reference[f'{mode}_rollout_{index}'] = latent
    mx.save_safetensors(str(args.output / 'reference.safetensors'), reference)
    # Release the oracle's checkpoint before launching the native process.
    del model, block
    mx.clear_cache()
    native_path = args.output / 'native.safetensors'
    subprocess.run([str(args.probe.resolve()), str(args.checkpoint.resolve()),
                    str(input_path.resolve()), str(native_path.resolve())], check=True)
    candidate = mx.load(str(native_path))
    if set(candidate) != set(reference):
        raise AssertionError('native/reference stage sets differ')
    report = {}
    for name in reference:
        a = np.asarray(reference[name]).astype(np.float64)
        b = np.asarray(candidate[name]).astype(np.float64)
        if a.shape != b.shape:
            raise AssertionError(f'{name}: shape mismatch')
        delta = a - b
        report[name] = {
            'reference_dtype': str(reference[name].dtype), 'native_dtype': str(candidate[name].dtype),
            'shape': list(a.shape), 'finite': bool(np.isfinite(a).all() and np.isfinite(b).all()),
            'exact': bool(reference[name].dtype == candidate[name].dtype and np.array_equal(a, b)),
            'max_abs': float(np.max(np.abs(delta))),
            'rmse': float(np.sqrt(np.mean(delta ** 2))),
        }
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not all(value['exact'] and value['finite'] for value in report.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
