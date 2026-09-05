#!/usr/bin/env python3
"""Bridge real ComfyUI conditioning and VAE decode to the native runtime.

This is an interim correctness tool.  The native Transformer consumes the
fully processed video/audio contexts produced *after* the LTX embeddings
connectors.  The bridge exports those contexts as raw BF16 files, then can
decode the final native video latent into an MP4 for visual comparison.

The bridge intentionally runs in separate processes from the native engine so
the 14 GB text encoder and 20 GB Transformer checkpoint do not remain resident
while the Metal denoiser runs.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

DEFAULT_COMFY_ROOT = (
    Path(os.environ["COMFYUI_ROOT"])
    if os.environ.get("COMFYUI_ROOT")
    else None
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    root.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    commands = root.add_subparsers(dest="command", required=True)

    encode = commands.add_parser("encode")
    encode.add_argument("text_encoder", type=Path)
    encode.add_argument("transformer", type=Path)
    encode.add_argument("output_dir", type=Path)
    encode.add_argument("--prompt", required=True)
    encode.add_argument(
        "--raw-only",
        action="store_true",
        help="export Gemma dual-projection output and skip slow connectors",
    )

    connect = commands.add_parser("connect-raw")
    connect.add_argument("transformer", type=Path)
    connect.add_argument("raw_dir", type=Path)
    connect.add_argument("output_dir", type=Path)

    compare = commands.add_parser("compare-conditioning")
    compare.add_argument("candidate_dir", type=Path)
    compare.add_argument("reference_dir", type=Path)

    decode = commands.add_parser("decode-video")
    decode.add_argument("video_vae", type=Path)
    decode.add_argument("artifacts_dir", type=Path)
    decode.add_argument("output", type=Path)
    decode.add_argument("--fps", type=int)
    decode.add_argument("--codec", default="libx264")
    decode.add_argument("--crf", type=int, default=18)

    vae_fixture = commands.add_parser("export-video-vae-fixture")
    vae_fixture.add_argument("video_vae", type=Path)
    vae_fixture.add_argument("output_dir", type=Path)
    vae_fixture.add_argument("--frames", type=int, default=2)
    vae_fixture.add_argument("--height", type=int, default=2)
    vae_fixture.add_argument("--width", type=int, default=3)

    native_video = commands.add_parser("encode-native-video")
    native_video.add_argument("pixels", type=Path)
    native_video.add_argument("output", type=Path)
    native_video.add_argument("--frames", type=int, required=True)
    native_video.add_argument("--height", type=int, required=True)
    native_video.add_argument("--width", type=int, required=True)
    native_video.add_argument("--fps", type=int, default=24)
    native_video.add_argument("--codec", default="libx264")
    native_video.add_argument("--crf", type=int, default=18)

    video_compare = commands.add_parser("compare-videos")
    video_compare.add_argument("candidate", type=Path)
    video_compare.add_argument("reference", type=Path)
    video_compare.add_argument("--contact-sheet", type=Path)
    return root


def import_comfy(comfy_root: Path | None):
    if comfy_root is None:
        raise RuntimeError(
            "set COMFYUI_ROOT or pass --comfy-root before the subcommand"
        )
    if not comfy_root.is_dir():
        raise RuntimeError(f"ComfyUI root does not exist: {comfy_root}")
    sys.path.insert(0, str(comfy_root))
    import av  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    import comfy.model_management as model_management  # noqa: PLC0415
    import comfy.sd as comfy_sd  # noqa: PLC0415
    import comfy.utils as comfy_utils  # noqa: PLC0415

    return av, np, torch, model_management, comfy_sd, comfy_utils


def write_bf16(path: Path, tensor, torch) -> None:
    value = tensor.detach().to(device="cpu", dtype=torch.bfloat16)
    value = value.contiguous().view(torch.uint16).numpy()
    value.tofile(path)


def encode_conditioning(args: argparse.Namespace) -> None:
    _, _, torch, model_management, comfy_sd, _ = import_comfy(
        args.comfy_root
    )
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; run on the host Apple GPU")
    for path in (args.text_encoder, args.transformer):
        if not path.is_file():
            raise RuntimeError(f"checkpoint does not exist: {path}")

    started = time.perf_counter()
    clip = comfy_sd.load_clip(
        ckpt_paths=[str(args.text_encoder)],
        embedding_directory=[],
        clip_type=comfy_sd.CLIPType.LTXV,
    )
    encoded = clip.encode_from_tokens(
        clip.tokenize(args.prompt), return_pooled=True, return_dict=True
    )
    raw_context = encoded["cond"].detach().to(device="cpu")
    unprocessed = bool(encoded.get("unprocessed_ltxav_embeds", False))
    attention_mask = encoded.get("attention_mask")
    text_seconds = time.perf_counter() - started

    if raw_context.ndim != 3 or raw_context.shape[0] != 1 or \
            raw_context.shape[-1] != 4096 + 2048:
        raise RuntimeError(
            f"unexpected raw projected context shape: {tuple(raw_context.shape)}"
        )
    raw_rows = int(raw_context.shape[1])
    raw_video = raw_context[0, :, :4096]
    raw_audio = raw_context[0, :, 4096:]
    raw_mask = torch.zeros(raw_rows, dtype=torch.float32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_bf16(args.output_dir / "raw_video_context.bf16", raw_video, torch)
    write_bf16(args.output_dir / "raw_audio_context.bf16", raw_audio, torch)
    write_bf16(args.output_dir / "raw_text_mask.bf16", raw_mask, torch)
    raw_manifest = {
        "format": "ltx-mac-raw-conditioning-v1",
        "prompt": args.prompt,
        "text_encoder": str(args.text_encoder),
        "shape": list(raw_context.shape),
        "video_shape": list(raw_video.shape),
        "audio_shape": list(raw_audio.shape),
        "all_input_rows_valid": True,
        "connector_minimum_rows": 1024,
        "connector_register_period": 128,
        "expected_processed_rows": (
            (max(1024, raw_rows) + 127) // 128
        ) * 128,
        "text_encode_seconds": text_seconds,
        "requires_embeddings_connector": True,
    }
    (args.output_dir / "raw_conditioning.json").write_text(
        json.dumps(raw_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("raw_conditioning=" + json.dumps(raw_manifest, ensure_ascii=False))
    if args.raw_only:
        return

    del encoded
    del clip
    gc.collect()
    model_management.soft_empty_cache()

    connector_started = time.perf_counter()
    patcher = comfy_sd.load_diffusion_model(str(args.transformer))
    model_management.load_models_gpu([patcher], force_full_load=True)
    diffusion = patcher.model.diffusion_model
    execution_device = patcher.load_device
    inference_dtype = patcher.model.get_dtype_inference()
    with torch.inference_mode():
        processed = diffusion.preprocess_text_embeds(
            raw_context.to(device=execution_device, dtype=inference_dtype),
            unprocessed=unprocessed,
        ).to(device="cpu")
    connector_seconds = time.perf_counter() - connector_started

    if processed.ndim != 3 or processed.shape[0] != 1:
        raise RuntimeError(
            f"unexpected processed context shape: {tuple(processed.shape)}"
        )
    if processed.shape[-1] != 4096 + 2048:
        raise RuntimeError(
            "processed LTX context must contain 4096 video and 2048 audio "
            f"features, got {processed.shape[-1]}"
        )
    rows = int(processed.shape[1])
    video_context = processed[0, :, :4096]
    audio_context = processed[0, :, 4096:]

    if attention_mask is not None and attention_mask.shape[-1] == rows:
        valid = attention_mask.detach().to(device="cpu", dtype=torch.bool)
        while valid.ndim > 1:
            valid = valid[0]
        additive_mask = torch.zeros(rows, dtype=torch.float32)
        additive_mask.masked_fill_(~valid, torch.finfo(torch.bfloat16).min)
        mask_source = "text-encoder-attention-mask"
    else:
        additive_mask = torch.zeros(rows, dtype=torch.float32)
        mask_source = "all-exported-tokens-valid"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_bf16(args.output_dir / "video_context.bf16", video_context, torch)
    write_bf16(args.output_dir / "audio_context.bf16", audio_context, torch)
    write_bf16(args.output_dir / "text_mask.bf16", additive_mask, torch)
    manifest = {
        "format": "ltx-mac-conditioning-v1",
        "prompt": args.prompt,
        "text_encoder": str(args.text_encoder),
        "transformer": str(args.transformer),
        "raw_shape": list(raw_context.shape),
        "processed_shape": list(processed.shape),
        "video_shape": list(video_context.shape),
        "audio_shape": list(audio_context.shape),
        "mask_shape": list(additive_mask.shape),
        "mask_source": mask_source,
        "unprocessed_input": unprocessed,
        "text_encode_seconds": text_seconds,
        "connector_seconds": connector_seconds,
    }
    (args.output_dir / "conditioning.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def read_bf16(path: Path, shape: list[int], np, torch):
    elements = 1
    for dimension in shape:
        elements *= dimension
    bits = np.fromfile(path, dtype=np.uint16)
    if bits.size != elements:
        raise RuntimeError(
            f"{path} contains {bits.size} BF16 values, expected {elements}"
        )
    return torch.from_numpy(bits.copy()).view(torch.bfloat16).reshape(shape)


def connect_raw_conditioning(args: argparse.Namespace) -> None:
    _, np, torch, model_management, comfy_sd, _ = import_comfy(
        args.comfy_root
    )
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; run on the host Apple GPU")
    if not args.transformer.is_file():
        raise RuntimeError(f"checkpoint does not exist: {args.transformer}")
    metadata = json.loads(
        (args.raw_dir / "raw_conditioning.json").read_text(encoding="utf-8")
    )
    video_shape = [int(value) for value in metadata["video_shape"]]
    audio_shape = [int(value) for value in metadata["audio_shape"]]
    if len(video_shape) != 2 or len(audio_shape) != 2 or \
            video_shape[0] != audio_shape[0] or \
            video_shape[1] != 4096 or audio_shape[1] != 2048:
        raise RuntimeError(
            f"invalid raw conditioning shapes: {video_shape}, {audio_shape}"
        )
    raw_video = read_bf16(
        args.raw_dir / "raw_video_context.bf16", video_shape, np, torch
    )
    raw_audio = read_bf16(
        args.raw_dir / "raw_audio_context.bf16", audio_shape, np, torch
    )
    raw_context = torch.cat((raw_video, raw_audio), dim=-1).unsqueeze(0)

    load_started = time.perf_counter()
    patcher = comfy_sd.load_diffusion_model(str(args.transformer))
    model_management.load_models_gpu([patcher], force_full_load=True)
    diffusion = patcher.model.diffusion_model
    execution_device = patcher.load_device
    inference_dtype = patcher.model.get_dtype_inference()
    load_seconds = time.perf_counter() - load_started
    connector_started = time.perf_counter()
    with torch.inference_mode():
        processed = diffusion.preprocess_text_embeds(
            raw_context.to(device=execution_device, dtype=inference_dtype),
            unprocessed=True,
        ).to(device="cpu")
    connector_seconds = time.perf_counter() - connector_started
    if processed.ndim != 3 or processed.shape[0] != 1 or \
            processed.shape[-1] != 4096 + 2048:
        raise RuntimeError(
            f"unexpected processed context shape: {tuple(processed.shape)}"
        )

    rows = int(processed.shape[1])
    video_context = processed[0, :, :4096]
    audio_context = processed[0, :, 4096:]
    additive_mask = torch.zeros(rows, dtype=torch.bfloat16)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_bf16(args.output_dir / "video_context.bf16", video_context, torch)
    write_bf16(args.output_dir / "audio_context.bf16", audio_context, torch)
    write_bf16(args.output_dir / "text_mask.bf16", additive_mask, torch)
    manifest = {
        "format": "ltx-mac-conditioning-reference-v1",
        "source": "ComfyUI preprocess_text_embeds",
        "transformer": str(args.transformer),
        "raw_directory": str(args.raw_dir),
        "raw_rows": video_shape[0],
        "processed_rows": rows,
        "video_shape": list(video_context.shape),
        "audio_shape": list(audio_context.shape),
        "transformer_load_seconds": load_seconds,
        "connector_seconds": connector_seconds,
    }
    (args.output_dir / "conditioning.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


def compare_conditioning(args: argparse.Namespace) -> None:
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    def load(directory: Path, name: str):
        path = directory / name
        bits = np.fromfile(path, dtype=np.uint16)
        return torch.from_numpy(bits.copy()).view(torch.bfloat16).float()

    reference_metadata = json.loads(
        (args.reference_dir / "conditioning.json").read_text(
            encoding="utf-8"
        )
    )
    raw_rows = int(reference_metadata.get("raw_rows", 0))
    report = {
        "format": "ltx-mac-conditioning-comparison-v2",
        "prompt_rows": raw_rows,
    }

    def metrics(candidate, reference):
        candidate_finite = torch.isfinite(candidate)
        reference_finite = torch.isfinite(reference)
        if not bool(torch.all(candidate_finite & reference_finite)):
            return {
                "elements": candidate.numel(),
                "candidate_non_finite": int((~candidate_finite).sum()),
                "reference_non_finite": int((~reference_finite).sum()),
            }
        candidate64 = candidate.double()
        reference64 = reference.double()
        difference64 = candidate64 - reference64
        reference_norm = torch.linalg.vector_norm(reference64)
        candidate_norm = torch.linalg.vector_norm(candidate64)
        difference_norm = torch.linalg.vector_norm(difference64)
        denominator = float(reference_norm)
        cosine_denominator = float(reference_norm * candidate_norm)
        cosine = (
            float(torch.dot(candidate64, reference64)) / cosine_denominator
            if cosine_denominator else 0.0
        )
        return {
            "elements": candidate.numel(),
            "rel_l2": (
                float(difference_norm) / denominator
                if denominator else float("inf")
            ),
            "cosine": max(-1.0, min(1.0, cosine)),
            "max_abs": float(torch.max(torch.abs(difference64))),
            "candidate_rms": float(torch.sqrt(torch.mean(candidate64.square()))),
            "reference_rms": float(torch.sqrt(torch.mean(reference64.square()))),
            "candidate_non_finite": 0,
            "reference_non_finite": 0,
        }

    for modality in ("video", "audio"):
        name = f"{modality}_context.bf16"
        candidate = load(args.candidate_dir, name)
        reference = load(args.reference_dir, name)
        if candidate.shape != reference.shape:
            raise RuntimeError(
                f"{modality} shapes differ: {tuple(candidate.shape)} vs "
                f"{tuple(reference.shape)}"
            )
        dim = 4096 if modality == "video" else 2048
        rows = candidate.numel() // dim
        candidate = candidate.reshape(rows, dim)
        reference = reference.reshape(rows, dim)
        modality_report = {
            "all_rows": metrics(candidate.flatten(), reference.flatten())
        }
        if 0 < raw_rows <= rows:
            modality_report["prompt_rows"] = metrics(
                candidate[:raw_rows].flatten(),
                reference[:raw_rows].flatten(),
            )
            if raw_rows < rows:
                modality_report["register_rows"] = metrics(
                    candidate[raw_rows:].flatten(),
                    reference[raw_rows:].flatten(),
                )
        report[modality] = modality_report
    print(json.dumps(report, indent=2))


def save_mp4(images, output: Path, fps: int, codec: str, crf: int,
             av, torch) -> None:
    if images.ndim == 5:
        if images.shape[0] != 1:
            raise RuntimeError(
                f"only batch size 1 is supported, got {tuple(images.shape)}"
            )
        images = images[0]
    if images.ndim != 4 or images.shape[-1] < 3:
        raise RuntimeError(f"unexpected decoded video shape: {tuple(images.shape)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    height = int(images.shape[1])
    width = int(images.shape[2])
    container = av.open(str(output), mode="w")
    stream = container.add_stream(codec, rate=Fraction(fps, 1))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(crf)}
    try:
        for image in images:
            rgb = image[..., :3].mul(255.0).clamp(0.0, 255.0)
            rgb = rgb.to(device="cpu", dtype=torch.uint8).numpy()
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def decode_video(args: argparse.Namespace) -> None:
    av, np, torch, _, comfy_sd, comfy_utils = import_comfy(args.comfy_root)
    if not args.video_vae.is_file():
        raise RuntimeError(f"VAE checkpoint does not exist: {args.video_vae}")
    metadata_path = args.artifacts_dir / "generation.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    video_info = metadata["video"]
    token_shape = [int(value) for value in video_info["shape"]]
    if len(token_shape) != 5 or token_shape[0] != 1:
        raise RuntimeError(f"unsupported video latent shape: {token_shape}")
    token_major = read_bf16(
        args.artifacts_dir / video_info["file"], token_shape, np, torch
    )
    latent = token_major.permute(0, 4, 1, 2, 3).contiguous()

    vae_started = time.perf_counter()
    vae_sd, vae_metadata = comfy_utils.load_torch_file(
        str(args.video_vae), safe_load=True, return_metadata=True
    )
    vae = comfy_sd.VAE(sd=vae_sd, metadata=vae_metadata)
    load_seconds = time.perf_counter() - vae_started
    decode_started = time.perf_counter()
    with torch.inference_mode():
        images = vae.decode(latent)
    decode_seconds = time.perf_counter() - decode_started

    fps = args.fps or int(metadata["decoded_video"]["fps"])
    save_started = time.perf_counter()
    save_mp4(images, args.output, fps, args.codec, args.crf, av, torch)
    save_seconds = time.perf_counter() - save_started
    report = {
        "format": "ltx-mac-video-decode-v1",
        "artifacts": str(args.artifacts_dir),
        "video_vae": str(args.video_vae),
        "latent_shape_bcfhw": list(latent.shape),
        "decoded_shape": list(images.shape),
        "fps": fps,
        "output": str(args.output),
        "vae_load_seconds": load_seconds,
        "decode_seconds": decode_seconds,
        "encode_seconds": save_seconds,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def export_video_vae_fixture(args: argparse.Namespace) -> None:
    _, _, torch, model_management, comfy_sd, comfy_utils = import_comfy(
        args.comfy_root
    )
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; run on the host Apple GPU")
    if not args.video_vae.is_file():
        raise RuntimeError(f"VAE checkpoint does not exist: {args.video_vae}")
    if args.frames <= 0 or args.height <= 0 or args.width <= 0:
        raise RuntimeError("fixture dimensions must be positive")

    token_shape = [1, args.frames, args.height, args.width, 128]
    elements = 1
    for dimension in token_shape:
        elements *= dimension
    token_major = (
        (torch.arange(elements, dtype=torch.int64) % 257) - 128
    ).to(torch.float32).div_(256.0).reshape(token_shape).to(torch.bfloat16)
    latent = token_major.permute(0, 4, 1, 2, 3).contiguous()

    load_started = time.perf_counter()
    state_dict, metadata = comfy_utils.load_torch_file(
        str(args.video_vae), safe_load=True, return_metadata=True
    )
    vae = comfy_sd.VAE(sd=state_dict, metadata=metadata)
    model_management.load_models_gpu([vae.patcher], force_full_load=True)
    load_seconds = time.perf_counter() - load_started

    decode_started = time.perf_counter()
    with torch.inference_mode():
        raw_output = vae.first_stage_model.decode(
            latent.to(device=vae.device, dtype=vae.vae_dtype)
        ).to(device="cpu", dtype=torch.float32)
    decode_seconds = time.perf_counter() - decode_started

    expected_shape = [
        1,
        3,
        args.frames * 8 - 7,
        args.height * 32,
        args.width * 32,
    ]
    if list(raw_output.shape) != expected_shape:
        raise RuntimeError(
            f"unexpected raw decoder shape: {list(raw_output.shape)}, "
            f"expected {expected_shape}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_bf16(args.output_dir / "input_tokens.bf16", token_major, torch)
    raw_output.contiguous().numpy().tofile(args.output_dir / "output_f32.bin")
    report = {
        "format": "ltx-mac-video-vae-fixture-v1",
        "video_vae": str(args.video_vae),
        "input_layout": "BFHWC",
        "input_shape": token_shape,
        "input_dtype": "BF16",
        "output_layout": "BCFHW",
        "output_shape": expected_shape,
        "output_dtype": "F32",
        "vae_dtype": str(vae.vae_dtype),
        "load_seconds": load_seconds,
        "decode_seconds": decode_seconds,
    }
    (args.output_dir / "fixture.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def encode_native_video(args: argparse.Namespace) -> None:
    import av  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if args.frames <= 0 or args.height <= 0 or args.width <= 0:
        raise RuntimeError("video dimensions must be positive")
    shape = [1, 3, args.frames, args.height, args.width]
    raw = read_bf16(args.pixels, shape, np, torch).float()
    non_finite = int((~torch.isfinite(raw)).sum())
    if non_finite:
        raise RuntimeError(f"native video contains {non_finite} non-finite values")
    images = raw.add(1.0).div(2.0).clamp(0.0, 1.0).movedim(1, -1)
    started = time.perf_counter()
    save_mp4(images, args.output, args.fps, args.codec, args.crf, av, torch)
    encode_seconds = time.perf_counter() - started
    report = {
        "format": "ltx-mac-native-video-encode-v1",
        "pixels": str(args.pixels),
        "pixel_layout": "BCFHW",
        "pixel_shape": shape,
        "pixel_dtype": "BF16",
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "raw_mean": float(raw.mean()),
        "raw_rms": float(torch.sqrt(torch.mean(raw.square()))),
        "non_finite": non_finite,
        "fps": args.fps,
        "output": str(args.output),
        "encode_seconds": encode_seconds,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def compare_videos(args: argparse.Namespace) -> None:
    import av  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    def load(path: Path):
        if not path.is_file():
            raise RuntimeError(f"video does not exist: {path}")
        container = av.open(str(path))
        try:
            frames = [
                frame.to_ndarray(format="rgb24")
                for frame in container.decode(video=0)
            ]
        finally:
            container.close()
        if not frames:
            raise RuntimeError(f"video has no decoded frames: {path}")
        return np.stack(frames)

    candidate = load(args.candidate)
    reference = load(args.reference)
    if candidate.shape != reference.shape:
        raise RuntimeError(
            f"video shapes differ: {candidate.shape} vs {reference.shape}"
        )
    candidate64 = candidate.astype(np.float64)
    reference64 = reference.astype(np.float64)
    difference = candidate64 - reference64
    difference_norm = float(np.linalg.norm(difference.ravel()))
    candidate_norm = float(np.linalg.norm(candidate64.ravel()))
    reference_norm = float(np.linalg.norm(reference64.ravel()))
    mse = float(np.mean(difference * difference))
    report = {
        "format": "ltx-mac-video-comparison-v1",
        "candidate": str(args.candidate),
        "reference": str(args.reference),
        "shape": list(candidate.shape),
        "rel_l2": difference_norm / reference_norm if reference_norm else None,
        "cosine": (
            float(np.dot(candidate64.ravel(), reference64.ravel())) /
            (candidate_norm * reference_norm)
            if candidate_norm and reference_norm else None
        ),
        "mae_8bit": float(np.mean(np.abs(difference))),
        "rmse_8bit": mse ** 0.5,
        "psnr_db": (
            float("inf") if mse == 0.0 else 10.0 * np.log10(255.0**2 / mse)
        ),
        "max_abs_8bit": float(np.max(np.abs(difference))),
    }
    try:
        from skimage.metrics import structural_similarity  # noqa: PLC0415

        frame_ssim = [
            structural_similarity(
                reference[index], candidate[index],
                channel_axis=-1, data_range=255
            )
            for index in range(reference.shape[0])
        ]
        report["ssim_mean"] = float(np.mean(frame_ssim))
        report["ssim_min"] = float(np.min(frame_ssim))
    except ImportError:
        report["ssim"] = "skipped (scikit-image unavailable)"

    if args.contact_sheet:
        from PIL import Image, ImageDraw  # noqa: PLC0415

        count = min(5, reference.shape[0])
        indices = np.linspace(0, reference.shape[0] - 1, count).astype(int)
        frame_height, frame_width = reference.shape[1:3]
        label_height = 28
        sheet = Image.new(
            "RGB", (frame_width * count, label_height + frame_height * 2),
            color=(0, 0, 0)
        )
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 7), "reference (top) / native (bottom)", fill=(255, 255, 255))
        for column, frame_index in enumerate(indices):
            x = column * frame_width
            sheet.paste(
                Image.fromarray(reference[frame_index]), (x, label_height)
            )
            sheet.paste(
                Image.fromarray(candidate[frame_index]),
                (x, label_height + frame_height)
            )
            draw.text((x + 8, label_height + 7), f"frame {frame_index}",
                      fill=(255, 255, 255))
        args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(args.contact_sheet)
        report["contact_sheet"] = str(args.contact_sheet)
    print(json.dumps(report, indent=2))


def main() -> None:
    args = parser().parse_args()
    if args.command == "encode":
        encode_conditioning(args)
    elif args.command == "connect-raw":
        connect_raw_conditioning(args)
    elif args.command == "compare-conditioning":
        compare_conditioning(args)
    elif args.command == "decode-video":
        decode_video(args)
    elif args.command == "export-video-vae-fixture":
        export_video_vae_fixture(args)
    elif args.command == "encode-native-video":
        encode_native_video(args)
    elif args.command == "compare-videos":
        compare_videos(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
