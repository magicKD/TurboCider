#!/usr/bin/env python3
"""Lossless local Comfy-to-reference layouts, never edits original weights."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import time

from flux9_bf16_framework_screen import generation, check_generations, write_json
from runtime_ane_common import sha256_file

ROOT = Path(__file__).resolve().parents[2]
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4}


def read_header(path):
    with path.open("rb") as source:
        prefix = source.read(8)
        length = struct.unpack("<Q", prefix)[0]
        if not 0 < length <= 16 << 20:
            raise ValueError("bounded safetensors header required")
        raw = source.read(length)
    header = json.loads(raw)
    return header, prefix + raw


def qwen_vae_key(key):
    if key.startswith("conv1."):
        key = "quant_conv" + key[5:]
    elif key.startswith("conv2."):
        key = "post_quant_conv" + key[5:]
    substitutions = (
        (r"^(encoder|decoder)\.conv1\.", r"\1.conv_in."),
        (r"^(encoder|decoder)\.head\.0\.", r"\1.norm_out."),
        (r"^(encoder|decoder)\.head\.2\.", r"\1.conv_out."),
        (r"\.middle\.0\.", ".mid_block.resnets.0."),
        (r"\.middle\.1\.", ".mid_block.attentions.0."),
        (r"\.middle\.2\.", ".mid_block.resnets.1."),
        (r"decoder\.upsamples\.([0-4])\.upsamples\.3\.", r"decoder.up_blocks.\1.upsampler."),
        (r"decoder\.upsamples\.([0-4])\.upsamples\.([0-2])\.", r"decoder.up_blocks.\1.resnets.\2."),
        (r"encoder\.downsamples\.([0-4])\.downsamples\.2\.", r"encoder.down_blocks.\1.downsampler."),
        (r"encoder\.downsamples\.([0-4])\.downsamples\.([0-1])\.", r"encoder.down_blocks.\1.resnets.\2."),
        (r"\.residual\.0\.", ".norm1."), (r"\.residual\.2\.", ".conv1."),
        (r"\.residual\.3\.", ".norm2."), (r"\.residual\.6\.", ".conv2."),
        (r"\.shortcut\.", ".conv_shortcut."),
    )
    for pattern, replacement in substitutions:
        key = re.sub(pattern, replacement, key)
    return key


def z_vae_key(key):
    for side in ("encoder", "decoder"):
        key = key.replace(f"{side}.mid.block_1.", f"{side}.mid_block.resnets.0.")
        key = key.replace(f"{side}.mid.block_2.", f"{side}.mid_block.resnets.1.")
        for old, new in (("norm", "group_norm"), ("q", "to_q"), ("k", "to_k"),
                         ("v", "to_v"), ("proj_out", "to_out.0")):
            key = key.replace(f"{side}.mid.attn_1.{old}.", f"{side}.mid_block.attentions.0.{new}.")
        key = key.replace(f"{side}.norm_out.", f"{side}.conv_norm_out.")
    match = re.match(r"^(encoder\.down|decoder\.up)\.([0-3])\.(.*)$", key)
    if match:
        side, index, tail = match.groups()
        block = int(index) if side == "encoder.down" else 3 - int(index)
        tail = re.sub(r"^block\.([0-9]+)\.", r"resnets.\1.", tail)
        tail = tail.replace("nin_shortcut.", "conv_shortcut.")
        tail = tail.replace("downsample.conv.", "downsamplers.0.conv.")
        tail = tail.replace("upsample.conv.", "upsamplers.0.conv.")
        key = f'{"encoder.down_blocks" if side == "encoder.down" else "decoder.up_blocks"}.{block}.{tail}'
    return key


def mapped_header(header, kind):
    output = {"__metadata__": dict(header.get("__metadata__", {}), tc_layout="lossless-reference-view-v1")}
    bindings = []
    for name, value in header.items():
        if name == "__metadata__":
            continue
        value = dict(value, shape=list(value["shape"]), data_offsets=list(value["data_offsets"]))
        if value["dtype"] not in DTYPE_BYTES or math.prod(value["shape"]) * DTYPE_BYTES[value["dtype"]] != value["data_offsets"][1] - value["data_offsets"][0]:
            raise ValueError("unsupported dtype or inconsistent original tensor extent")
        names = [(name, value)]
        if kind == "qwen-text" and (name.startswith("model.layers.") or name in ("model.embed_tokens.weight", "model.norm.weight")):
            names = [("model.language_model." + name[6:], value)]
        elif kind == "qwen-vae":
            if len(value["shape"]) == 5 and "time_conv" not in name:
                if value["shape"][2] != 1:
                    raise ValueError("single-frame spatial kernel required; never discard temporal values")
                value["shape"].pop(2)
            names = [(qwen_vae_key(name), value)]
        elif kind == "z-vae":
            target = z_vae_key(name)
            if ".attentions.0." in target and target.endswith(".weight") and len(value["shape"]) == 4:
                if value["shape"][2:] != [1, 1]:
                    raise ValueError("linear attention must have exact singleton spatial axes")
                value["shape"] = value["shape"][:2]
            names = [(target, value)]
        elif kind == "z-dit":
            target = name.replace("x_embedder.", "all_x_embedder.2-1.").replace("final_layer.", "all_final_layer.2-1.")
            target = target.replace(".attention.out.", ".attention.to_out.0.")
            target = target.replace(".attention.q_norm.", ".attention.norm_q.").replace(".attention.k_norm.", ".attention.norm_k.")
            if target.endswith(".attention.qkv.weight"):
                rows = value["shape"][0]
                if len(value["shape"]) != 2 or rows % 3:
                    raise ValueError("exact contiguous three-way QKV split required")
                start, end = value["data_offsets"]
                span = (end - start) // 3
                names = [(target.replace("qkv.weight", f"to_{part}.weight"),
                          dict(value, shape=[rows // 3, value["shape"][1]],
                               data_offsets=[start + index * span, start + (index + 1) * span]))
                         for index, part in enumerate(("q", "k", "v"))]
            else:
                names = [(target, value)]
        elif kind not in ("qwen-text", "identity"):
            raise ValueError("unknown layout recipe")
        for target, tensor in names:
            if target in output:
                raise ValueError("layout rename collision")
            output[target] = tensor
            bindings.append(dict(original=name, mapped=target, dtype=tensor["dtype"], shape=tensor["shape"],
                                 original_offsets=value["data_offsets"], mapped_offsets=tensor["data_offsets"]))
    extents = sorted(value["data_offsets"] for name, value in output.items() if name != "__metadata__")
    if extents[0][0] != 0 or any(a[1] != b[0] for a, b in zip(extents, extents[1:])):
        raise ValueError("mapped tensors must cover the entire original payload exactly once")
    return output, bindings


def rewrite(source, destination, kind):
    before = generation(source)
    header, original_prefix = read_header(source)
    mapped, bindings = mapped_header(header, kind)
    if max(value["data_offsets"][1] for name, value in mapped.items() if name != "__metadata__") != before["bytes"] - len(original_prefix):
        raise ValueError("tensor extents must cover the entire physical payload")
    encoded = json.dumps(mapped, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    source_sha, payload_sha = hashlib.sha256(original_prefix), hashlib.sha256()
    start = time.perf_counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as original, destination.open("xb") as converted:
        original.seek(len(original_prefix))
        converted.write(struct.pack("<Q", len(encoded)))
        converted.write(encoded)
        while block := original.read(4 << 20):
            source_sha.update(block)
            payload_sha.update(block)
            converted.write(block)
    check_generations({str(source): before})
    actual_payload = hashlib.sha256()
    converted_header, converted_prefix = read_header(destination)
    with destination.open("rb") as converted:
        converted.seek(len(converted_prefix))
        while block := converted.read(4 << 20):
            actual_payload.update(block)
    if converted_header != mapped or payload_sha.digest() != actual_payload.digest():
        raise ValueError("full derived payload verification failed")
    destination.chmod(0o400)
    return dict(source=str(source), source_generation=before, source_sha256=source_sha.hexdigest(),
                derived=str(destination), derived_generation=generation(destination), derived_sha256=sha256_file(destination),
                payload_sha256=payload_sha.hexdigest(), full_payload_equal=True, kind=kind,
                preparation_seconds=time.perf_counter() - start, tensor_bindings=bindings)


def prepare(family, destination):
    if destination.exists():
        raise ValueError("fresh layout destination required; use its verified receipt for later sweeps")
    original = ROOT / ("models/Comfy-Org-Qwen-Image-2.1" if family == "qwen21" else "models/Comfy-Org-z_image_turbo")
    specifications = (
        [(original / "text_encoders/qwen3vl_8b_bf16.safetensors", "text_encoder/model.safetensors", "qwen-text"),
         (original / "vae/qwen_image_2.1_vae_bf16.safetensors", "vae/diffusion_pytorch_model.safetensors", "qwen-vae")]
        if family == "qwen21" else
        [(original / "split_files/diffusion_models/z_image_turbo_bf16.safetensors", "transformer/diffusion_pytorch_model.safetensors", "z-dit"),
         (original / "split_files/vae/ae.safetensors", "vae/diffusion_pytorch_model.safetensors", "z-vae")])
    if shutil.disk_usage(ROOT).free < sum(source.stat().st_size for source, _, _ in specifications) + (2 << 30):
        raise ValueError("insufficient space for derived layouts plus 2GiB reserve; no original files deleted")
    destination.mkdir(parents=True)
    report = dict(schema="tc-lossless-reference-layout-v1", family=family, status="preparing", qualification_passed=False,
                  files=[], links=[], scope="header-only key/singleton-shape/QKV-range mapping; complete payload bytes unchanged; no quantization or modified reference code")
    receipt = destination / "layout-receipt.json"
    write_json(receipt, report)
    try:
        for source, relative, kind in specifications:
            report["files"].append(rewrite(source, destination / relative, kind))
            write_json(receipt, report)
        if family == "qwen21":
            links = [(original / "diffusion_models/qwen_image_2.1_bf16.safetensors", "transformer/diffusion_pytorch_model.safetensors"),
                     (original / "processor/tokenizer.json", "processor/tokenizer.json")]
            header, _ = read_header(destination / "text_encoder/model.safetensors")
            write_json(destination / "text_encoder/model.safetensors.index.json",
                       dict(weight_map={name: "model.safetensors" for name in header if name != "__metadata__"}))
        else:
            links = [(original / "split_files/text_encoders/qwen_3_4b.safetensors", "text_encoder/model.safetensors")]
            (destination / "tokenizer").symlink_to(original / "tokenizer", target_is_directory=True)
        for source, relative in links:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)
            report["links"].append(dict(path=str(target), source=str(source), generation=generation(source)))
        if family == "qwen21":
            write_json(destination / "processor/tokenizer_config.json",
                       dict(tokenizer_class="Qwen2Tokenizer", model_max_length=131072, eos_token="<|im_end|>", pad_token="<|endoftext|>"))
        report["status"] = "complete_layout_only"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(receipt, report)
    return report


def validate_layout(destination):
    report = json.loads((destination / "layout-receipt.json").read_text())
    if report["status"] != "complete_layout_only" or not report["files"]:
        raise ValueError("complete independently payload-verified layout receipt required")
    for record in report["files"]:
        if record["full_payload_equal"] is not True:
            raise ValueError("lossless conversion proof missing")
        check_generations({record["source"]: record["source_generation"], record["derived"]: record["derived_generation"]})
    for link in report["links"]:
        check_generations({link["source"]: link["generation"]})
        if Path(link["path"]).resolve(strict=True) != Path(link["source"]).resolve(strict=True):
            raise ValueError("layout symlink retargeted")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("qwen21", "z"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.family, args.output.absolute())
    print(json.dumps(dict(status=result["status"], files=len(result["files"]))))


if __name__ == "__main__":
    main()
