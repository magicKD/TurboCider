#!/usr/bin/env python3
"""Export a small real-weight LTX-2.5 block-0 parity fixture.

Run this with the Python environment belonging to the local ComfyUI install.
The exporter uses ComfyUI's quantized Linear loader for the real INT8 ConvRot
checkpoint, but applies the reference block semantics used by ltx-2-mlx:

* Q/K RMSNorm epsilon is 1e-6 from the embedded model config;
* A->V and V->A consume the same pre-cross video/audio states;
* timestep AdaLN is computed with the real top-level checkpoint weights;
* the last two text tokens are masked to exercise mask handling.

Generated fixture files are development artifacts and must not be committed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--comfy-root",
        type=Path,
        default=Path("/Users/james/ComfyUI-Installs/ComfyUI/ComfyUI"),
    )
    parser.add_argument("--video-rows", type=int, default=4)
    parser.add_argument("--audio-rows", type=int, default=4)
    parser.add_argument("--text-rows", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.5)
    return parser.parse_args()


ARGS = parse_args()
sys.path.insert(0, str(ARGS.comfy_root))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

import comfy.model_management  # noqa: E402
import comfy.ops  # noqa: E402
from comfy.ldm.lightricks.av_model import BasicAVTransformerBlock  # noqa: E402
from comfy.ldm.lightricks.model import (  # noqa: E402
    AdaLayerNormSingle,
    freqs_cis_matrix,
    generate_freq_grid_np,
    generate_freqs,
)


VIDEO_DIM = 4096
AUDIO_DIM = 2048
VIDEO_HEADS = 32
AUDIO_HEADS = 32
VIDEO_HEAD_DIM = 128
AUDIO_HEAD_DIM = 64
DTYPE = torch.bfloat16
DEVICE = torch.device("mps")
ROOT_PREFIX = "model.diffusion_model."
BLOCK_PREFIX = ROOT_PREFIX + "transformer_blocks.0."


def load_module(handle, module: torch.nn.Module, prefix: str) -> None:
    state = {
        key.removeprefix(prefix): handle.get_tensor(key)
        for key in handle.keys()
        if key.startswith(prefix)
    }
    if not state:
        raise RuntimeError(f"checkpoint has no tensors under {prefix}")
    incompatible = module.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"load {prefix}: missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )


def make_adaln(operations, dim: int, coefficient: int) -> AdaLayerNormSingle:
    return AdaLayerNormSingle(
        dim,
        embedding_coefficient=coefficient,
        use_additional_conditions=False,
        dtype=DTYPE,
        device=DEVICE,
        operations=operations,
    ).eval()


def timestep_params(
    module: AdaLayerNormSingle, value: float
) -> tuple[torch.Tensor, torch.Tensor]:
    timestep = torch.tensor([value], dtype=torch.float32, device=DEVICE)
    params, embedded = module(
        timestep,
        {"resolution": None, "aspect_ratio": None},
        batch_size=1,
        hidden_dtype=DTYPE,
    )
    return params[:, None, :], embedded


def combine(table: torch.Tensor, params: torch.Tensor,
            rows: int, dim: int) -> torch.Tensor:
    return (
        table[:rows].to(device=DEVICE, dtype=DTYPE)[None, :, :]
        + params.reshape(1, rows, dim)
    )[0]


def rms_norm(value: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.rms_norm(
        value, (value.shape[-1],), weight=None, eps=1e-6
    )


def make_positions(video_rows: int, audio_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    if video_rows != 4:
        raise ValueError("the initial fixture currently requires --video-rows 4")
    video = np.empty((1, 3, video_rows), dtype=np.float32)
    video[:, 0, :] = 0.5 / 24.0
    video[:, 1, :] = np.array([16.0, 16.0, 48.0, 48.0], dtype=np.float32)
    video[:, 2, :] = np.array([16.0, 48.0, 16.0, 48.0], dtype=np.float32)
    token = np.arange(audio_rows, dtype=np.float32)
    start = np.maximum(token * 4.0 + 1.0 - 4.0, 0.0)
    end = np.maximum((token + 1.0) * 4.0 + 1.0 - 4.0, 0.0)
    audio = ((start + end) * 0.5 * (160.0 / 16000.0))[None, None, :]
    return (
        torch.from_numpy(video).to(DEVICE),
        torch.from_numpy(audio).to(DEVICE),
    )


def make_rope(positions: torch.Tensor, inner_dim: int, heads: int,
              max_positions: list[float]) -> tuple[torch.Tensor, bool]:
    axes = positions.shape[1]
    indices = generate_freq_grid_np(10000.0, axes, inner_dim).to(DEVICE)
    frequencies = generate_freqs(indices, positions, max_positions, False)
    used = (inner_dim // (2 * axes)) * axes
    padding = inner_dim // 2 - used
    return freqs_cis_matrix(frequencies, padding, True, heads, DTYPE)


def reference_forward(
    block: BasicAVTransformerBlock,
    video: torch.Tensor,
    audio: torch.Tensor,
    video_context: torch.Tensor,
    audio_context: torch.Tensor,
    text_mask: torch.Tensor,
    video_pe,
    audio_pe,
    video_cross_pe,
    audio_cross_pe,
    video_adaln: torch.Tensor,
    audio_adaln: torch.Tensor,
    video_prompt: torch.Tensor,
    audio_prompt: torch.Tensor,
    av_video: torch.Tensor,
    av_audio: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    options: dict = {}
    vx = video.clone()
    ax = audio.clone()

    norm = rms_norm(vx) * (1.0 + video_adaln[1]) + video_adaln[0]
    vx = vx + block.attn1(norm, pe=video_pe, transformer_options=options) * video_adaln[2]

    norm = rms_norm(ax) * (1.0 + audio_adaln[1]) + audio_adaln[0]
    ax = ax + block.audio_attn1(norm, pe=audio_pe, transformer_options=options) * audio_adaln[2]

    norm = rms_norm(vx) * (1.0 + video_adaln[7]) + video_adaln[6]
    context = video_context * (1.0 + video_prompt[1]) + video_prompt[0]
    vx = vx + block.attn2(
        norm, context=context, mask=text_mask, transformer_options=options
    ) * video_adaln[8]

    norm = rms_norm(ax) * (1.0 + audio_adaln[7]) + audio_adaln[6]
    context = audio_context * (1.0 + audio_prompt[1]) + audio_prompt[0]
    ax = ax + block.audio_attn2(
        norm, context=context, mask=text_mask, transformer_options=options
    ) * audio_adaln[8]

    video_pre_cross = vx
    audio_pre_cross = ax
    video_norm = rms_norm(video_pre_cross)
    audio_norm = rms_norm(audio_pre_cross)
    video_query = video_norm * (1.0 + av_video[0]) + av_video[1]
    audio_key_value = audio_norm * (1.0 + av_audio[0]) + av_audio[1]
    audio_query = audio_norm * (1.0 + av_audio[2]) + av_audio[3]
    video_key_value = video_norm * (1.0 + av_video[2]) + av_video[3]
    a2v = block.audio_to_video_attn(
        video_query,
        context=audio_key_value,
        pe=video_cross_pe,
        k_pe=audio_cross_pe,
        transformer_options=options,
    )
    v2a = block.video_to_audio_attn(
        audio_query,
        context=video_key_value,
        pe=audio_cross_pe,
        k_pe=video_cross_pe,
        transformer_options=options,
    )
    vx = video_pre_cross + a2v * av_video[4]
    ax = audio_pre_cross + v2a * av_audio[4]

    norm = rms_norm(vx) * (1.0 + video_adaln[4]) + video_adaln[3]
    vx = vx + block.ff(norm) * video_adaln[5]
    norm = rms_norm(ax) * (1.0 + audio_adaln[4]) + audio_adaln[3]
    ax = ax + block.audio_ff(norm) * audio_adaln[5]
    return vx, ax


def tensor_entry(path: Path, tensor: torch.Tensor, dtype: str) -> dict:
    value = tensor.detach().contiguous().cpu()
    if dtype == "bf16":
        value.view(torch.uint16).numpy().tofile(path)
    elif dtype == "f32":
        value.to(torch.float32).numpy().tofile(path)
    else:
        raise ValueError(dtype)
    return {"file": path.name, "dtype": dtype, "shape": list(tensor.shape)}


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; run outside the restricted sandbox")
    if ARGS.video_rows != 4 or ARGS.audio_rows <= 0 or ARGS.text_rows < 3:
        raise ValueError("invalid fixture geometry")
    ARGS.output_dir.mkdir(parents=True, exist_ok=True)
    comfy.model_management.in_training = True
    operations = comfy.ops.mixed_precision_ops(
        compute_dtype=DTYPE, full_precision_mm=True
    )

    block = BasicAVTransformerBlock(
        VIDEO_DIM,
        AUDIO_DIM,
        VIDEO_HEADS,
        AUDIO_HEADS,
        VIDEO_HEAD_DIM,
        AUDIO_HEAD_DIM,
        v_context_dim=VIDEO_DIM,
        a_context_dim=AUDIO_DIM,
        apply_gated_attention=True,
        cross_attention_adaln=True,
        ff_bias=False,
        audio_ff_bias=True,
        dtype=DTYPE,
        device=DEVICE,
        operations=operations,
    ).eval()
    adaln_modules = {
        "video": make_adaln(operations, VIDEO_DIM, 9),
        "audio": make_adaln(operations, AUDIO_DIM, 9),
        "video_prompt": make_adaln(operations, VIDEO_DIM, 2),
        "audio_prompt": make_adaln(operations, AUDIO_DIM, 2),
        "av_video": make_adaln(operations, VIDEO_DIM, 4),
        "av_audio": make_adaln(operations, AUDIO_DIM, 4),
        "a2v_gate": make_adaln(operations, VIDEO_DIM, 1),
        "v2a_gate": make_adaln(operations, AUDIO_DIM, 1),
    }
    io_modules = {
        "video_patchify": operations.Linear(
            128, VIDEO_DIM, bias=True, dtype=DTYPE, device=DEVICE
        ).eval(),
        "audio_patchify": operations.Linear(
            128, AUDIO_DIM, bias=True, dtype=DTYPE, device=DEVICE
        ).eval(),
        "video_output": operations.Linear(
            VIDEO_DIM, 128, bias=True, dtype=DTYPE, device=DEVICE
        ).eval(),
        "audio_output": operations.Linear(
            AUDIO_DIM, 128, bias=True, dtype=DTYPE, device=DEVICE
        ).eval(),
    }
    module_prefixes = {
        "video": ROOT_PREFIX + "adaln_single.",
        "audio": ROOT_PREFIX + "audio_adaln_single.",
        "video_prompt": ROOT_PREFIX + "prompt_adaln_single.",
        "audio_prompt": ROOT_PREFIX + "audio_prompt_adaln_single.",
        "av_video": ROOT_PREFIX + "av_ca_video_scale_shift_adaln_single.",
        "av_audio": ROOT_PREFIX + "av_ca_audio_scale_shift_adaln_single.",
        "a2v_gate": ROOT_PREFIX + "av_ca_a2v_gate_adaln_single.",
        "v2a_gate": ROOT_PREFIX + "av_ca_v2a_gate_adaln_single.",
    }
    io_prefixes = {
        "video_patchify": ROOT_PREFIX + "patchify_proj.",
        "audio_patchify": ROOT_PREFIX + "audio_patchify_proj.",
        "video_output": ROOT_PREFIX + "proj_out.",
        "audio_output": ROOT_PREFIX + "audio_proj_out.",
    }
    with safe_open(ARGS.checkpoint, framework="pt", device="cpu") as handle:
        load_module(handle, block, BLOCK_PREFIX)
        for name, module in adaln_modules.items():
            load_module(handle, module, module_prefixes[name])
        for name, module in io_modules.items():
            load_module(handle, module, io_prefixes[name])
        video_output_table = handle.get_tensor(
            ROOT_PREFIX + "scale_shift_table"
        ).to(DEVICE, DTYPE)
        audio_output_table = handle.get_tensor(
            ROOT_PREFIX + "audio_scale_shift_table"
        ).to(DEVICE, DTYPE)

    for attention in (
        block.attn1,
        block.audio_attn1,
        block.attn2,
        block.audio_attn2,
        block.audio_to_video_attn,
        block.video_to_audio_attn,
    ):
        attention.q_norm.eps = 1e-6
        attention.k_norm.eps = 1e-6

    with torch.inference_mode():
        main_timestep = ARGS.sigma * 1000.0
        av_timestep = ARGS.sigma
        video_params, video_embedded = timestep_params(
            adaln_modules["video"], main_timestep
        )
        audio_params, audio_embedded = timestep_params(
            adaln_modules["audio"], main_timestep
        )
        video_prompt_params, video_prompt_embedded = timestep_params(
            adaln_modules["video_prompt"], main_timestep
        )
        audio_prompt_params, audio_prompt_embedded = timestep_params(
            adaln_modules["audio_prompt"], main_timestep
        )
        av_video_params, av_video_embedded = timestep_params(
            adaln_modules["av_video"], av_timestep
        )
        av_audio_params, av_audio_embedded = timestep_params(
            adaln_modules["av_audio"], av_timestep
        )
        a2v_gate_params, a2v_gate_embedded = timestep_params(
            adaln_modules["a2v_gate"], av_timestep
        )
        v2a_gate_params, v2a_gate_embedded = timestep_params(
            adaln_modules["v2a_gate"], av_timestep
        )

        video_adaln = combine(
            block.scale_shift_table, video_params, 9, VIDEO_DIM
        )
        audio_adaln = combine(
            block.audio_scale_shift_table, audio_params, 9, AUDIO_DIM
        )
        video_prompt = combine(
            block.prompt_scale_shift_table, video_prompt_params, 2, VIDEO_DIM
        )
        audio_prompt = combine(
            block.audio_prompt_scale_shift_table,
            audio_prompt_params,
            2,
            AUDIO_DIM,
        )
        av_video = torch.cat(
            [
                combine(
                    block.scale_shift_table_a2v_ca_video,
                    av_video_params,
                    4,
                    VIDEO_DIM,
                ),
                combine(
                    block.scale_shift_table_a2v_ca_video[4:],
                    a2v_gate_params,
                    1,
                    VIDEO_DIM,
                ),
            ],
            dim=0,
        )
        av_audio = torch.cat(
            [
                combine(
                    block.scale_shift_table_a2v_ca_audio,
                    av_audio_params,
                    4,
                    AUDIO_DIM,
                ),
                combine(
                    block.scale_shift_table_a2v_ca_audio[4:],
                    v2a_gate_params,
                    1,
                    AUDIO_DIM,
                ),
            ],
            dim=0,
        )

        rng = np.random.default_rng(42)
        video_patch_input = torch.from_numpy(
            rng.standard_normal(
                (1, ARGS.video_rows, 128), dtype=np.float32
            ) * 0.25
        ).to(DEVICE, DTYPE)
        audio_patch_input = torch.from_numpy(
            rng.standard_normal(
                (1, ARGS.audio_rows, 128), dtype=np.float32
            ) * 0.25
        ).to(DEVICE, DTYPE)
        video_patch_output = io_modules["video_patchify"](video_patch_input)
        audio_patch_output = io_modules["audio_patchify"](audio_patch_input)
        video_context_np = rng.standard_normal(
            (1, ARGS.text_rows, VIDEO_DIM), dtype=np.float32
        ) * 0.1
        audio_context_np = rng.standard_normal(
            (1, ARGS.text_rows, AUDIO_DIM), dtype=np.float32
        ) * 0.1
        video = video_patch_output
        audio = audio_patch_output
        video_context = torch.from_numpy(video_context_np).to(DEVICE, DTYPE)
        audio_context = torch.from_numpy(audio_context_np).to(DEVICE, DTYPE)
        text_mask = torch.zeros(
            (1, 1, 1, ARGS.text_rows), dtype=DTYPE, device=DEVICE
        )
        text_mask[..., -2:] = torch.finfo(DTYPE).min

        video_positions, audio_positions = make_positions(
            ARGS.video_rows, ARGS.audio_rows
        )
        video_pe = make_rope(
            video_positions, VIDEO_DIM, VIDEO_HEADS, [20.0, 2048.0, 2048.0]
        )
        audio_pe = make_rope(
            audio_positions, AUDIO_DIM, AUDIO_HEADS, [20.0]
        )
        video_cross_pe = make_rope(
            video_positions[:, 0:1, :], AUDIO_DIM, AUDIO_HEADS, [20.0]
        )
        audio_cross_pe = make_rope(
            audio_positions, AUDIO_DIM, AUDIO_HEADS, [20.0]
        )
        video_output, audio_output = reference_forward(
            block,
            video,
            audio,
            video_context,
            audio_context,
            text_mask,
            video_pe,
            audio_pe,
            video_cross_pe,
            audio_cross_pe,
            video_adaln,
            audio_adaln,
            video_prompt,
            audio_prompt,
            av_video,
            av_audio,
        )
        video_scale_shift = (
            video_output_table[None, None]
            + video_embedded[:, None, None, :]
        )
        video_shift = video_scale_shift[:, :, 0, :]
        video_scale = video_scale_shift[:, :, 1, :]
        video_head_hidden = torch.nn.functional.layer_norm(
            video_output, (VIDEO_DIM,), eps=1e-6
        )
        video_head_hidden = (
            video_head_hidden * (1.0 + video_scale) + video_shift
        )
        video_head_output = io_modules["video_output"](video_head_hidden)

        audio_scale_shift = (
            audio_output_table[None, None]
            + audio_embedded[:, None, None, :]
        )
        audio_shift = audio_scale_shift[:, :, 0, :]
        audio_scale = audio_scale_shift[:, :, 1, :]
        audio_head_hidden = torch.nn.functional.layer_norm(
            audio_output, (AUDIO_DIM,), eps=1e-6
        )
        audio_head_hidden = (
            audio_head_hidden * (1.0 + audio_scale) + audio_shift
        )
        audio_head_output = io_modules["audio_output"](audio_head_hidden)
        torch.mps.synchronize()

    tensors = {
        "video_input": tensor_entry(
            ARGS.output_dir / "video_input.bf16", video[0], "bf16"
        ),
        "audio_input": tensor_entry(
            ARGS.output_dir / "audio_input.bf16", audio[0], "bf16"
        ),
        "video_patch_input": tensor_entry(
            ARGS.output_dir / "video_patch_input.bf16",
            video_patch_input[0],
            "bf16",
        ),
        "audio_patch_input": tensor_entry(
            ARGS.output_dir / "audio_patch_input.bf16",
            audio_patch_input[0],
            "bf16",
        ),
        "video_patch_output_f32": tensor_entry(
            ARGS.output_dir / "video_patch_output.f32",
            video_patch_output[0],
            "f32",
        ),
        "audio_patch_output_f32": tensor_entry(
            ARGS.output_dir / "audio_patch_output.f32",
            audio_patch_output[0],
            "f32",
        ),
        "video_context": tensor_entry(
            ARGS.output_dir / "video_context.bf16", video_context[0], "bf16"
        ),
        "audio_context": tensor_entry(
            ARGS.output_dir / "audio_context.bf16", audio_context[0], "bf16"
        ),
        "text_mask": tensor_entry(
            ARGS.output_dir / "text_mask.bf16", text_mask[0, 0, 0], "bf16"
        ),
        "video_positions": tensor_entry(
            ARGS.output_dir / "video_positions.f32",
            video_positions[0].transpose(0, 1),
            "f32",
        ),
        "audio_positions": tensor_entry(
            ARGS.output_dir / "audio_positions.f32",
            audio_positions[0].transpose(0, 1),
            "f32",
        ),
        "video_adaln": tensor_entry(
            ARGS.output_dir / "video_adaln.bf16", video_adaln, "bf16"
        ),
        "audio_adaln": tensor_entry(
            ARGS.output_dir / "audio_adaln.bf16", audio_adaln, "bf16"
        ),
        "video_prompt": tensor_entry(
            ARGS.output_dir / "video_prompt.bf16", video_prompt, "bf16"
        ),
        "audio_prompt": tensor_entry(
            ARGS.output_dir / "audio_prompt.bf16", audio_prompt, "bf16"
        ),
        "av_video": tensor_entry(
            ARGS.output_dir / "av_video.bf16", av_video, "bf16"
        ),
        "av_audio": tensor_entry(
            ARGS.output_dir / "av_audio.bf16", av_audio, "bf16"
        ),
        "video_adaln_params": tensor_entry(
            ARGS.output_dir / "video_adaln_params.bf16",
            video_params[0, 0],
            "bf16",
        ),
        "video_embedded_timestep": tensor_entry(
            ARGS.output_dir / "video_embedded_timestep.bf16",
            video_embedded[0],
            "bf16",
        ),
        "audio_adaln_params": tensor_entry(
            ARGS.output_dir / "audio_adaln_params.bf16",
            audio_params[0, 0],
            "bf16",
        ),
        "audio_embedded_timestep": tensor_entry(
            ARGS.output_dir / "audio_embedded_timestep.bf16",
            audio_embedded[0],
            "bf16",
        ),
        "video_prompt_params": tensor_entry(
            ARGS.output_dir / "video_prompt_params.bf16",
            video_prompt_params[0, 0],
            "bf16",
        ),
        "video_prompt_embedded": tensor_entry(
            ARGS.output_dir / "video_prompt_embedded.bf16",
            video_prompt_embedded[0],
            "bf16",
        ),
        "audio_prompt_params": tensor_entry(
            ARGS.output_dir / "audio_prompt_params.bf16",
            audio_prompt_params[0, 0],
            "bf16",
        ),
        "audio_prompt_embedded": tensor_entry(
            ARGS.output_dir / "audio_prompt_embedded.bf16",
            audio_prompt_embedded[0],
            "bf16",
        ),
        "av_video_params": tensor_entry(
            ARGS.output_dir / "av_video_params.bf16",
            av_video_params[0, 0],
            "bf16",
        ),
        "av_video_embedded": tensor_entry(
            ARGS.output_dir / "av_video_embedded.bf16",
            av_video_embedded[0],
            "bf16",
        ),
        "av_audio_params": tensor_entry(
            ARGS.output_dir / "av_audio_params.bf16",
            av_audio_params[0, 0],
            "bf16",
        ),
        "av_audio_embedded": tensor_entry(
            ARGS.output_dir / "av_audio_embedded.bf16",
            av_audio_embedded[0],
            "bf16",
        ),
        "a2v_gate_params": tensor_entry(
            ARGS.output_dir / "a2v_gate_params.bf16",
            a2v_gate_params[0, 0],
            "bf16",
        ),
        "a2v_gate_embedded": tensor_entry(
            ARGS.output_dir / "a2v_gate_embedded.bf16",
            a2v_gate_embedded[0],
            "bf16",
        ),
        "v2a_gate_params": tensor_entry(
            ARGS.output_dir / "v2a_gate_params.bf16",
            v2a_gate_params[0, 0],
            "bf16",
        ),
        "v2a_gate_embedded": tensor_entry(
            ARGS.output_dir / "v2a_gate_embedded.bf16",
            v2a_gate_embedded[0],
            "bf16",
        ),
        "video_output_bf16": tensor_entry(
            ARGS.output_dir / "video_output.bf16", video_output[0], "bf16"
        ),
        "audio_output_bf16": tensor_entry(
            ARGS.output_dir / "audio_output.bf16", audio_output[0], "bf16"
        ),
        "video_output_f32": tensor_entry(
            ARGS.output_dir / "video_output.f32", video_output[0], "f32"
        ),
        "audio_output_f32": tensor_entry(
            ARGS.output_dir / "audio_output.f32", audio_output[0], "f32"
        ),
        "video_head_output_f32": tensor_entry(
            ARGS.output_dir / "video_head_output.f32",
            video_head_output[0],
            "f32",
        ),
        "audio_head_output_f32": tensor_entry(
            ARGS.output_dir / "audio_head_output.f32",
            audio_head_output[0],
            "f32",
        ),
    }
    manifest = {
        "format": "ltx-mac-block-fixture-v2",
        "checkpoint": str(ARGS.checkpoint.resolve()),
        "block": 0,
        "semantics": "ltx-reference-pre-cross",
        "qk_norm_epsilon": 1e-6,
        "sigma": ARGS.sigma,
        "timestep_scale_multiplier": 1000.0,
        "av_ca_timestep_scale_multiplier": 1.0,
        "masked_text_tokens": 2,
        "tensors": tensors,
    }
    (ARGS.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"video mean={video_output.float().mean().item():.8f} "
        f"rms={video_output.float().square().mean().sqrt().item():.8f}"
    )
    print(
        f"audio mean={audio_output.float().mean().item():.8f} "
        f"rms={audio_output.float().square().mean().sqrt().item():.8f}"
    )
    print(f"wrote {ARGS.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
