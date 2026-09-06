#!/usr/bin/env python3
"""Dump LTX 16 kHz base-vocoder stages from the Python MLX oracle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--ltx-mac", type=Path, required=True)
    result.add_argument("--weights", type=Path, required=True)
    result.add_argument("--mel", type=Path, required=True)
    result.add_argument("--batch", type=int, default=1)
    result.add_argument("--mel-time", type=int, required=True)
    result.add_argument("--dump-dir", type=Path, required=True)
    return result


def dump(path: Path, value) -> None:
    import mlx.core as mx

    path.parent.mkdir(parents=True, exist_ok=True)
    mx.eval(value)
    mx.synchronize()
    np.asarray(value.astype(mx.float32), dtype=np.float32).tofile(path)


def main() -> None:
    args = parser().parse_args()
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
        if not key.startswith(("bwe_generator.", "mel_stft."))
    }
    vocoder = VocoderWithBWE()
    model_keys = {
        key
        for key, _ in tree_flatten(vocoder.parameters())
        if not key.startswith(("bwe_generator.", "mel_stft."))
    }
    if model_keys != set(weights):
        raise RuntimeError(
            f"base vocoder keys differ: model={len(model_keys)}, "
            f"checkpoint={len(weights)}"
        )
    weights = convert_pytorch_convs(
        "vocoder", vocoder, weights, tree_flatten
    )
    # The reference container also owns BWE parameters; base-only parity
    # intentionally leaves those components at initialization values.
    vocoder.load_weights(list(weights.items()), strict=False)
    vocoder.upcast_weights_to_fp32()

    elements = args.batch * 2 * args.mel_time * 64
    bits = np.fromfile(args.mel, dtype=np.uint16)
    if bits.size != elements:
        raise RuntimeError(f"mel has {bits.size} elements, expected {elements}")
    values = (bits.astype(np.uint32) << 16).view(np.float32)
    mel = mx.array(
        values.reshape(args.batch, 2, args.mel_time, 64),
        dtype=mx.bfloat16,
    )
    value = mel.transpose(0, 2, 1, 3).reshape(
        args.batch, args.mel_time, 128
    ).astype(mx.float32)
    dump(args.dump_dir / "00_mel_btm.f32", value)
    value = vocoder.conv_pre(value)
    dump(args.dump_dir / "01_conv_pre.f32", value)
    for stage in range(vocoder.num_upsamples):
        value = vocoder.ups[stage](value)
        branches = [
            vocoder.resblocks[stage * vocoder.num_kernels + branch](value)
            for branch in range(vocoder.num_kernels)
        ]
        value = sum(branches[1:], branches[0]) / vocoder.num_kernels
        dump(args.dump_dir / f"02_stage_{stage}.f32", value)
    value = vocoder.act_post(value)
    dump(args.dump_dir / "08_act_post.f32", value)
    value = mx.tanh(vocoder.conv_post(value))
    dump(args.dump_dir / "09_waveform_16k.f32", value)


if __name__ == "__main__":
    main()
