#!/usr/bin/env python3
"""Lossless native layout of upstream FLUX4B saved Q4/g64, never edits sources."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import urllib.request

from flux4_restore_assets import REPO, REVISION
from flux9_bf16_framework_screen import generation, check_generations, registered_hashes, write_json
from flux9_affine_pack import native_key
from mflux_original_layout import read_header
from runtime_ane_common import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def affine_header(header, inverse, component, source_header=None):
    source_header = header if source_header is None else source_header
    mapped = {"__metadata__": dict(header.get("__metadata__", {}),
        tc_affine_group_size="64", tc_affine_bits="4", tc_layout="flux4-upstream-saved-native-v1")}
    for name, value in header.items():
        if name == "__metadata__":
            continue
        if name == "rotary_emb.inv_freq" and component == "text_encoder":
            if value["dtype"] != "F32" or len(value["shape"]) != 1:
                raise ValueError("stored F32 architecture constant must be preserved")
            target = name
        else:
            target = native_key(name, inverse)
        if target in mapped:
            raise ValueError("native key collision")
        mapped[target] = value
        if value["dtype"] == "U32" and (not name.endswith(".weight") or
                name[:-7] + ".scales" not in source_header or name[:-7] + ".biases" not in source_header):
            raise ValueError("every packed weight requires complete affine planes")
        if name.endswith(".scales"):
            prefix = name[:-7]
            weight, bias = source_header[prefix + ".weight"], source_header[prefix + ".biases"]
            if (weight["dtype"] != "U32" or value["dtype"] != "BF16" or bias["dtype"] != "BF16" or
                    len(weight["shape"]) != 2 or len(value["shape"]) != 2 or
                    value["shape"] != bias["shape"] or weight["shape"][0] != value["shape"][0] or
                    weight["shape"][1] != value["shape"][1] * 8):
                raise ValueError("exact upstream Q4/g64 affine planes required")
    return mapped


def rewrite_affine(source, destination, mapped):
    original, prefix = read_header(source)
    before = generation(source)
    if sorted(json.dumps(v, sort_keys=True) for k, v in original.items() if k != "__metadata__") != sorted(
            json.dumps(v, sort_keys=True) for k, v in mapped.items() if k != "__metadata__"):
        raise ValueError("tensor descriptors must remain byte-identical")
    encoded = json.dumps(mapped, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    digest = hashlib.sha256()
    with source.open("rb") as reader, destination.open("xb") as writer:
        reader.seek(len(prefix))
        writer.write(struct.pack("<Q", len(encoded)))
        writer.write(encoded)
        while block := reader.read(4 << 20):
            digest.update(block)
            writer.write(block)
    _, actual_prefix = read_header(destination)
    actual = hashlib.sha256()
    with destination.open("rb") as reader:
        reader.seek(len(actual_prefix))
        while block := reader.read(4 << 20):
            actual.update(block)
    check_generations({str(source): before})
    if digest.digest() != actual.digest() or source.stat().st_size - len(prefix) != destination.stat().st_size - len(actual_prefix):
        raise ValueError("complete affine payload must be unchanged")
    return dict(full_payload_equal=True, payload_sha256=digest.hexdigest(),
                bytes=destination.stat().st_size, sha256=sha256_file(destination))


def inverse_mapping(definitions, mapper):
    inverse, transforms = {}, {}
    for source, targets in mapper._build_flat_mapping(definitions, 20, 36).items():
        for mapped, transform in targets:
            if mapped in inverse and inverse[mapped] != source:
                raise ValueError("ambiguous source mapping")
            inverse[mapped] = source
            transforms[mapped] = None if transform is None else transform.__name__
    return inverse, transforms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "mflux", "asset-receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.model = args.model.resolve(strict=True)
    args.mflux = args.mflux.resolve(strict=True)
    args.asset_receipt = args.asset_receipt.resolve(strict=True)
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error("fresh native view required")
    receipt = json.loads(args.asset_receipt.read_text())
    if (receipt.get("status") != "complete_assets_only" or receipt.get("repo") != "mflux-community/flux2-klein-4b-mflux-q4" or
            receipt.get("revision") != "794cd159538149ad9830848508c31f0ea7088e58"):
        parser.error("pinned upstream Q4 asset receipt required")
    files = [args.model / row["path"] for row in receipt["files"]]
    identities = {str(path): generation(path) for path in files}
    hashes = registered_hashes(files, ROOT / "outputs/benchmark-source-registrations/flux4-q4.json")
    if any(path.stat().st_size != row["bytes"] or hashes[str(path)] != row["sha256"]
           for path, row in zip(files, receipt["files"])):
        raise ValueError("source generation differs from asset receipt")
    required = sum(p.stat().st_size for p in files) + (2 << 30) + (256 << 20)
    if shutil.disk_usage(ROOT).free < required:
        raise ValueError("native view would violate2GiB reserve; no originals deleted")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.mflux.resolve() / "src"))
    import mlx.core as mx
    from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition
    from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
    mx.set_default_device(mx.cpu)
    args.output.mkdir(parents=True)
    data = dict(status="preparing", schema="tc-flux4-native-saved-view-v1", runtime_verified=False,
        performance_verified=False, source_repo=receipt["repo"], source_revision=receipt["revision"],
        bits=4, group_size=64, components={}, configs={}, source_sha256={str(p.relative_to(ROOT)): hashes[str(p)] for p in files},
        scope="same saved affine payload incl quantized embedding/VAE attention; key remap and bit-exact inverse convolution transpose; not requantization or originalBF16")
    target = args.output / "layout-receipt.json"
    write_json(target, data)
    try:
        for definition in Flux2KleinWeightDefinition.get_components():
            name = definition.name
            destination = args.output / name
            destination.mkdir()
            inverse, transforms = inverse_mapping(definition.mapping_getter(), WeightMapper)
            records, combined = [], {}
            paths = sorted((args.model / name).glob("*.safetensors"))
            source_header = {}
            for path in paths:
                header, _ = read_header(path)
                for key, tensor in header.items():
                    if key != "__metadata__":
                        if key in source_header:
                            raise ValueError("duplicate source tensor across shards")
                        source_header[key] = tensor
            for path in paths:
                header, _ = read_header(path)
                mapped = affine_header(header, inverse, name, source_header)
                actual = destination / path.name
                if name != "vae":
                    record = rewrite_affine(path, actual, mapped)
                else:
                    arrays = mx.load(str(path))
                    converted = {}
                    for key, value in arrays.items():
                        native = native_key(key, inverse)
                        transform = transforms.get(key)
                        if transform == "transpose_conv2d_weight":
                            if value.ndim != 4 or value.dtype != mx.bfloat16:
                                raise ValueError("known BF162D convolution inverse required")
                            value = mx.transpose(value, (0, 3, 1, 2))
                        elif transform is not None:
                            raise ValueError("unverified VAE inverse transform")
                        converted[native] = value
                    mx.eval(list(converted.values()))
                    mx.save_safetensors(str(actual), converted, mapped["__metadata__"])
                    reloaded = mx.load(str(actual))
                    for key, value in arrays.items():
                        other = reloaded[native_key(key, inverse)]
                        if transforms.get(key) == "transpose_conv2d_weight":
                            other = mx.transpose(other, (0, 2, 3, 1))
                        if value.dtype != other.dtype or value.shape != other.shape or not mx.all(mx.view(value, mx.uint8) == mx.view(other, mx.uint8)).item():
                            raise ValueError("entire VAE tensor bitpattern differs")
                    record = dict(full_tensor_bitpattern_equivalent=True, bytes=actual.stat().st_size, sha256=sha256_file(actual))
                    del arrays, converted, reloaded
                actual_header, _ = read_header(actual)
                for key in actual_header:
                    if key != "__metadata__":
                        if key in combined:
                            raise ValueError("duplicate native source tensor")
                        combined[key] = path.name
                records.append(dict(file=path.name, stored_tensors=len(actual_header)-1,
                    quantized_matrices=sum(k.endswith(".scales") for k in actual_header), **record))
                write_json(target, data)
            if not records:
                raise ValueError("complete saved source component required")
            write_json(destination / "model.safetensors.index.json", dict(weight_map=combined,
                metadata=dict(quantization_level="4", group_size="64", tc_layout="flux4-native-saved-v1")))
            data["components"][name] = records
            # Configs come from the pinned original4B checkpoint, not9B or
            # guessed geometry. Weights remain the declared saved Q4 source.
            request = urllib.request.Request(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}/config.json")
            with urllib.request.urlopen(request, timeout=45) as response:
                raw = response.read((1 << 20) + 1)
            if len(raw) > 1 << 20:
                raise ValueError("bounded original config required")
            config = json.loads(raw)
            write_json(destination / "config.json", config)
            data["configs"][name] = dict(repo=REPO, revision=REVISION, remote_sha256=hashlib.sha256(raw).hexdigest(),
                sha256=sha256_file(destination / "config.json"))
            write_json(target, data)
        (args.output / "tokenizer").symlink_to(args.model.resolve() / "tokenizer", target_is_directory=True)
        check_generations(identities)
        data.update(status="complete_layout_only", final_free_bytes=shutil.disk_usage(ROOT).free)
        if data["final_free_bytes"] < 2 << 30:
            raise ValueError("post-conversion reserve violated")
    except BaseException as error:
        data.update(status="failed", error=str(error))
        raise
    finally:
        write_json(target, data)
    print(json.dumps(dict(status=data["status"], components=data["components"])))


if __name__ == "__main__":
    main()
