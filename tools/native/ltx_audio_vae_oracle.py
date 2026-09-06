#!/usr/bin/env python3
"""Dump LTX Audio VAE stages from the validated Python MLX oracle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--ltx-mac", type=Path, required=True)
    result.add_argument("--weights", type=Path, required=True)
    result.add_argument("--latent", type=Path, required=True)
    result.add_argument("--batch", type=int, default=1)
    result.add_argument("--tokens", type=int, required=True)
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
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    from tools.finalize_audio import (
        convert_pytorch_convs,
        require_exact_keys,
        split_combined_weights,
    )
    from ltx_core_mlx.model.audio_vae.audio_vae import AudioVAEDecoder, pixel_norm
    from ltx_core_mlx.utils.weights import remap_audio_vae_keys

    raw = mx.load(str(args.weights))
    audio, _ = split_combined_weights(raw)
    audio = remap_audio_vae_keys(audio)
    audio = {
        key.replace(".mean-of-means", ".mean_of_means").replace(
            ".std-of-means", ".std_of_means"
        ): value
        for key, value in audio.items()
    }
    decoder = AudioVAEDecoder()
    require_exact_keys("Audio VAE", decoder, audio, tree_flatten)
    audio = convert_pytorch_convs(
        "Audio VAE", decoder, audio, tree_flatten
    )
    decoder.update(tree_unflatten(list(audio.items())))

    elements = args.batch * args.tokens * 128
    bits = np.fromfile(args.latent, dtype=np.uint16)
    if bits.size != elements:
        raise RuntimeError(f"latent has {bits.size} elements, expected {elements}")
    values = (bits.astype(np.uint32) << 16).view(np.float32)
    latent_blc = mx.array(values.reshape(args.batch, args.tokens, 128),
                          dtype=mx.bfloat16)
    dump(args.dump_dir / "00_input_blc.f32", latent_blc)
    value = latent_blc
    mean = decoder.per_channel_statistics.mean_of_means.reshape(1, 1, -1)
    standard_deviation = decoder.per_channel_statistics.std_of_means.reshape(
        1, 1, -1
    )
    value = value * standard_deviation + mean
    dump(args.dump_dir / "01_denormalized_blc.f32", value)
    value = value.reshape(args.batch, args.tokens, 8, 16).transpose(0, 1, 3, 2)
    dump(args.dump_dir / "02_nhwc.f32", value)
    value = decoder.conv_in(value)
    dump(args.dump_dir / "03_conv_in.f32", value)
    value = decoder.mid.block_1(value)
    dump(args.dump_dir / "04_mid_block_1.f32", value)
    value = decoder.mid.block_2(value)
    dump(args.dump_dir / "05_mid_block_2.f32", value)
    value = decoder.up[2](value)
    dump(args.dump_dir / "06_up_2.f32", value)
    value = decoder.up[1](value)
    dump(args.dump_dir / "07_up_1.f32", value)
    value = decoder.up[0](value)
    dump(args.dump_dir / "08_up_0.f32", value)
    value = nn.silu(pixel_norm(value))
    dump(args.dump_dir / "09_pre_out.f32", value)
    value = decoder.conv_out(value)
    dump(args.dump_dir / "10_conv_out_nhwc.f32", value)
    value = value.transpose(0, 3, 1, 2)
    dump(args.dump_dir / "11_output_bctf.f32", value)


if __name__ == "__main__":
    main()
