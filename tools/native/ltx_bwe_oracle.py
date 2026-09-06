#!/usr/bin/env python3
"""Dump LTX 48 kHz BWE stages from the Python MLX oracle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--ltx-mac", type=Path, required=True)
    result.add_argument("--weights", type=Path, required=True)
    result.add_argument("--waveform", type=Path, required=True)
    result.add_argument("--batch", type=int, default=1)
    result.add_argument("--samples", type=int, required=True)
    result.add_argument("--dump-dir", type=Path, required=True)
    return result


def dump(path: Path, value) -> None:
    import mlx.core as mx

    path.parent.mkdir(parents=True, exist_ok=True)
    value = mx.contiguous(value.astype(mx.float32))
    mx.eval(value)
    mx.synchronize()
    np.asarray(value, dtype=np.float32).tofile(path)


def main() -> None:
    args = parser().parse_args()
    if args.batch <= 0 or args.samples <= 0:
        raise RuntimeError("batch and samples must be positive")
    sys.path.insert(0, str(args.ltx_mac.resolve()))
    sys.path.insert(0, str((args.ltx_mac / "python").resolve()))

    import mlx.core as mx
    from mlx.utils import tree_flatten
    from tools.finalize_audio import (
        convert_pytorch_convs,
        split_combined_weights,
    )
    from ltx_core_mlx.model.audio_vae.bwe import VocoderWithBWE

    raw = mx.load(str(args.weights))
    _, weights = split_combined_weights(raw)
    weights = {
        key: value
        for key, value in weights.items()
        if key.startswith(("bwe_generator.", "mel_stft."))
    }
    vocoder = VocoderWithBWE()
    model_keys = {
        key
        for key, _ in tree_flatten(vocoder.parameters())
        if key.startswith(("bwe_generator.", "mel_stft."))
    }
    if model_keys != set(weights):
        raise RuntimeError(
            f"BWE keys differ: model={len(model_keys)}, "
            f"checkpoint={len(weights)}"
        )
    weights = convert_pytorch_convs(
        "BWE", vocoder, weights, tree_flatten
    )
    vocoder.load_weights(list(weights.items()), strict=False)
    vocoder.upcast_weights_to_fp32()

    elements = args.batch * args.samples * 2
    waveform_values = np.fromfile(args.waveform, dtype=np.float32)
    if waveform_values.size != elements:
        raise RuntimeError(
            f"waveform has {waveform_values.size} elements, expected {elements}"
        )
    waveform = mx.array(
        waveform_values.reshape(args.batch, args.samples, 2),
        dtype=mx.float32,
    )
    dump(args.dump_dir / "00_waveform_16k_btc.f32", waveform)
    channels_first = waveform.transpose(0, 2, 1)
    padded_samples = (args.samples + 79) // 80 * 80
    if padded_samples != args.samples:
        channels_first = mx.pad(
            channels_first,
            [(0, 0), (0, 0), (0, padded_samples - args.samples)],
        )
    flat = channels_first.reshape(args.batch * 2, padded_samples)
    bwe_mel = vocoder.mel_stft(flat)
    dump(args.dump_dir / "01_bwe_mel_bc.f32", bwe_mel)
    frames = bwe_mel.shape[1]
    bwe_mel = bwe_mel.reshape(args.batch, 2, frames, 64)
    bwe_mel = bwe_mel.transpose(0, 1, 3, 2).reshape(
        args.batch, 128, frames
    )
    bwe_mel = bwe_mel.transpose(0, 2, 1)
    dump(args.dump_dir / "02_bwe_mel_btm.f32", bwe_mel)

    generator = vocoder.bwe_generator
    value = generator.conv_pre(bwe_mel)
    dump(args.dump_dir / "03_bwe_conv_pre.f32", value)
    for stage in range(generator.num_upsamples):
        value = generator.ups[stage](value)
        branches = [
            generator.resblocks[stage * generator.num_kernels + branch](
                value
            )
            for branch in range(generator.num_kernels)
        ]
        value = sum(branches[1:], branches[0]) / generator.num_kernels
        dump(args.dump_dir / f"04_bwe_stage_{stage}.f32", value)
    value = generator.act_post(value)
    residual = generator.conv_post(value).transpose(0, 2, 1)
    dump(args.dump_dir / "08_residual_bct.f32", residual)

    skip = mx.stack(
        [
            vocoder._resampler(channels_first[:, channel, :])
            for channel in range(2)
        ],
        axis=1,
    )
    dump(args.dump_dir / "09_skip_bct.f32", skip)
    shared = min(skip.shape[-1], residual.shape[-1])
    output = mx.clip(skip[:, :, :shared] + residual[:, :, :shared], -1, 1)
    output = output[:, :, : args.samples * 3].transpose(0, 2, 1)
    dump(args.dump_dir / "10_waveform_48k_btc.f32", output)


if __name__ == "__main__":
    main()
