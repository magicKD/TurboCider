#!/usr/bin/env python3
"""Header/stat-only local base-model inventory. Never hashes model payloads."""
import argparse
from collections import Counter
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]


def inspect_weight(path):
    status = path.stat()
    if not path.is_file() or status.st_size <= 0:
        raise ValueError("missing regular weight")
    result = dict(path=str(path.relative_to(ROOT)), bytes=status.st_size)
    with path.open("rb") as source:
        prefix = source.read(8)
        if path.suffix == ".gguf":
            if len(prefix) != 8 or prefix[:4] != b"GGUF" or struct.unpack("<I", prefix[4:])[0] != 3:
                raise ValueError("unsupported GGUF header")
            result.update(format="gguf-v3", content_verified=False)
        else:
            if len(prefix) != 8:
                raise ValueError("truncated safetensors")
            length = struct.unpack("<Q", prefix)[0]
            if not 0 < length <= min(16 << 20, status.st_size - 8):
                raise ValueError("invalid bounded safetensors header")
            header = json.loads(source.read(length))
            tensors = {key: value for key, value in header.items() if key != "__metadata__"}
            if not tensors:
                raise ValueError("empty tensor header")
            for tensor in tensors.values():
                begin, end = tensor["data_offsets"]
                if not 0 <= begin <= end <= status.st_size - 8 - length:
                    raise ValueError("tensor range exceeds file")
            result.update(format="safetensors", tensor_count=len(tensors),
                          dtypes=dict(Counter(value["dtype"] for value in tensors.values())),
                          content_verified=False)
    return result


def specification(model, kind, components, note=""):
    return dict(model=model, kind=kind, components=components, note=note)


def inventory():
    comfy = "models/Comfy-Org-z_image_turbo"
    z_common = dict(encoder=[comfy + "/split_files/text_encoders/qwen_3_4b.safetensors"],
                    vae=[comfy + "/split_files/vae/ae.safetensors"],
                    tokenizer=[comfy + "/tokenizer/tokenizer.json"])
    qwen = "models/Comfy-Org-Qwen-Image-2.1"
    qwen_common = dict(vae=[qwen + "/vae/qwen_image_2.1_vae_bf16.safetensors"],
                       tokenizer=[qwen + "/processor/tokenizer.json"])
    specs = [
        specification("Qwen Image 2.1 BF16", "base", dict(qwen_common,
            dit=[qwen + "/diffusion_models/qwen_image_2.1_bf16.safetensors"],
            encoder=[qwen + "/text_encoders/qwen3vl_8b_bf16.safetensors"])),
        specification("Qwen Image 2.1 original Q4_K_M", "base", dict(qwen_common,
            dit=["models/unsloth-qwen21-q4-k-m/diffusion_models/qwen-image-2.1-Q4_K_M.gguf"],
            encoder=["models/unsloth-qwen21-q4-k-m/text_encoders/Qwen3-VL-8B-Instruct-Q4_K_M.gguf"])),
        specification("Z-Image Turbo BF16", "base", dict(z_common,
            dit=[comfy + "/split_files/diffusion_models/z_image_turbo_bf16.safetensors"])),
        specification("Z-Image Turbo NVFP4", "base-format-variant", dict(z_common,
            dit=[comfy + "/split_files/diffusion_models/z_image_turbo_nvfp4.safetensors"])),
        specification("Z-Image Turbo INT8 ConvRot", "base-format-variant", dict(z_common,
            dit=[comfy + "/split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors"])),
        specification("Z-Image Turbo GGUF Q4_0", "base-format-variant", dict(z_common,
            dit=["models/z-image-runtime-gguf-q4-main/z_image_turbo-Q4_0.gguf"])),
        specification("Z-Image Turbo GGUF Q8_0", "base-format-variant", dict(z_common,
            dit=["models/z-image-runtime-gguf-q8/z_image_turbo-Q8_0.gguf"])),
        specification("FLUX.2 Klein 9B BF16", "base", dict(
            dit=["models/FLUX.2-klein-9B/transformer/diffusion_pytorch_model-*.safetensors"],
            encoder=["models/FLUX.2-klein-9B/text_encoder/model-*.safetensors"],
            vae=["models/FLUX.2-klein-9B/vae/diffusion_pytorch_model.safetensors"],
            tokenizer=["models/FLUX.2-klein-9B/tokenizer/tokenizer.json"])),
        specification("LTX 2.5 distilled INT8 ConvRot", "standalone-distilled-base", dict(
            dit=["models/LTX-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"],
            encoder=["models/LTX-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"],
            connector=["models/LTX-2.5/connector.safetensors"],
            upsampler=["models/LTX-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"],
            video_vae=["models/LTX-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors"],
            audio_vae=["models/LTX-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors"],
            tokenizer=["models/LTX-2.5/gemma4-12b-ltx-v1/tokenizer.json"])),
        specification("MiniMax H3 original BF16 source", "source-components", dict(
            dit=["models/MiniMax-H3-ModelScope/FL2VA/transformer/model-*.safetensors"],
            video_vae=["models/MiniMax-H3-ModelScope/FL2VA/video_vae/*.safetensors"],
            audio_vae=["models/MiniMax-H3-ModelScope/FL2VA/audio_vae/model.safetensors"]),
            "Source components alone do not prove an executable native model root."),
        specification("FastH3 standalone four-step INT6", "standalone-distilled-base", dict(
            dit=["models/FastH3-MLX/int6/mlx_h3_dit.safetensors"],
            encoder=["models/FastH3-ModelScope/text_encoder/model-*.safetensors"],
            video_vae=["models/FastH3-ModelScope/vae/diffusion_pytorch_model-*.safetensors"],
            audio_vae=["models/FastH3-ModelScope/audio_vae/diffusion_pytorch_model.safetensors"],
            tokenizer=["models/FastH3-ModelScope/tokenizer/tokenizer.json"])),
        specification("Qwen3 4B original GGUF Q4_K_M", "encoder-component", dict(
            encoder=["models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"])),
        specification("Qwen3 4B original GGUF Q8_0", "encoder-component", dict(
            encoder=["models/Qwen3-4B-GGUF/Qwen3-4B-Q8_0.gguf"])),
        specification("Qwen3 4B TurboCider affine Q4", "encoder-component", dict(
            encoder=["models/qwen3-4b-tc-q4/text_encoder/*.safetensors"])),
        specification("Qwen3 4B TurboCider affine Q8", "encoder-component", dict(
            encoder=["models/qwen3-4b-tc-q8/text_encoder/*.safetensors"])),
    ]
    rows = []
    for spec in specs:
        components, missing, errors = {}, [], []
        for name, patterns in spec["components"].items():
            paths = [path for pattern in patterns for path in ROOT.glob(pattern) if path.is_file()]
            if not paths:
                missing.append(name)
            files = []
            for path in paths:
                try:
                    files.append(inspect_weight(path) if path.suffix in (".gguf", ".safetensors") else
                                 dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size))
                except (OSError, ValueError, KeyError, TypeError) as error:
                    errors.append(dict(path=str(path.relative_to(ROOT)), reason=str(error)))
            components[name] = files
            # Sharded checkpoints need every index-referenced shard, not merely
            # one matching file. Flat native checkpoints do not consume indexes.
            for parent in {path.parent for path in paths if "-000" in path.name}:
                for index in parent.glob("*.safetensors.index.json"):
                    try:
                        weight_map = json.loads(index.read_text())["weight_map"]
                        if not weight_map:
                            raise ValueError("empty shard index")
                        absent = [shard for shard in sorted(set(weight_map.values()))
                                  if not (parent / shard).is_file()]
                        if absent:
                            raise ValueError("index-referenced shards missing: " + ", ".join(absent))
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        errors.append(dict(path=str(index.relative_to(ROOT)), reason=str(error)))
        rows.append(dict(model=spec["model"], kind=spec["kind"], components=components,
                         missing_components=missing, header_errors=errors, note=spec["note"],
                         assets_present=not missing and not errors, runtime_support_verified=False,
                         logical_weight_bytes=sum(item["bytes"] for files in components.values()
                             for item in files if item["path"].endswith((".gguf", ".safetensors")))))
    return dict(schema="tc-base-model-header-inventory-v1", date="2026-10-11", rows=rows,
                scope="header/stat availability, not payload integrity, runtime admission or inference qualification; shared logical bytes are not disk/RAM usage",
                exclusions=["PE-T2I/PE-I2I/Viggle adapter or student overlays", "VDN stage-DMD Turbo adapter",
                            "Core ML graphs, app-bindings, pe-library and alias-only model roots",
                            "Comfy distill_patch_lora"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("fresh inventory output required")
    result = inventory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as target:
        json.dump(result, target, indent=2, allow_nan=False); target.write("\n")
    for row in result["rows"]:
        print(row["model"], "present" if row["assets_present"] else "missing/invalid", row["missing_components"])


if __name__ == "__main__":
    main()
