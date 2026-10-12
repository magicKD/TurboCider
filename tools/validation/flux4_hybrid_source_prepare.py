#!/usr/bin/env python3
"""Prepare a source-bound mixed Q4/BF16 view for the existing FLUX4 hybrid graph.

Only forty single-block projections are decoded. This is not original BF16,
bit-exact packed inference, an activation-INT8 recipe, or a performance result.
Original files are never changed; encoder/VAE/tokenizer are read-only links.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys

from flux9_bf16_framework_screen import check_generations, generation, registered_hashes, write_json
from mflux_original_layout import read_header
from runtime_ane_common import sha256_file

ROOT = Path(__file__).resolve().parents[2]
RESERVE = 2 << 30
ANE_WEIGHT_BYTES = 20 * 3 * 3072 * 6144
DTYPE_BYTES = {"U32": 4, "BF16": 2, "F16": 2, "F32": 4}


def prefixes():
    return [f"single_transformer_blocks.{block}.attn.{name}"
            for block in range(20) for name in ("to_qkv_mlp_proj", "to_out")]


def tensor_bytes(item):
    width = DTYPE_BYTES[item["dtype"]]
    for dimension in item["shape"]:
        if type(dimension) is not int or dimension <= 0:
            raise ValueError("positive integer tensor geometry required")
        width *= dimension
    start, end = item["data_offsets"]
    if type(start) is not int or type(end) is not int or start < 0 or end - start != width:
        raise ValueError("complete tensor payload required")
    return width


def plan(headers, free_bytes):
    selected = prefixes()
    if len(headers) != 387:
        raise ValueError("complete current387-tensor native transformer required")
    original = sum(tensor_bytes(item) for item in headers.values())
    removed, dense = 0, 0
    for prefix in selected:
        weight, scales, biases = (headers[prefix + suffix] for suffix in (".weight", ".scales", ".biases"))
        expected = [27648, 384] if prefix.endswith("to_qkv_mlp_proj") else [3072, 1536]
        if (weight["dtype"] != "U32" or weight["shape"] != expected or
                scales["dtype"] != "BF16" or biases["dtype"] != "BF16" or
                scales["shape"] != [expected[0], expected[1] // 8] or scales["shape"] != biases["shape"]):
            raise ValueError("exact current Q4/g64 single projection geometry required")
        removed += sum(tensor_bytes(item) for item in (weight, scales, biases))
        dense += expected[0] * expected[1] * 8 * 2
    derived = original - removed + dense
    # Source packages + managed compiled cache + allowance for additional NE
    # compilation copies, plus bounded temporary/conversion space. No cleanup
    # of original models, build products, or generated media is attempted.
    budget = derived + 3 * ANE_WEIGHT_BYTES + (1 << 30)
    result = dict(original_payload_bytes=original, replaced_original_bytes=removed,
                  dense_replacements_bytes=dense, derived_payload_bytes=derived,
                  ane_int8_source_weights_bytes=ANE_WEIGHT_BYTES,
                  conservative_total_budget_bytes=budget, minimum_reserve_bytes=RESERVE,
                  free_bytes=free_bytes, source_tensors=387, derived_tensors=307,
                  decoded_projections=40, unchanged_tensors=267,
                  remaining_quantized_matrices=69)
    if free_bytes < budget + RESERVE:
        raise ValueError("derived source/export/cache would violate2GiB reserve; no originals deleted")
    return result


def prepare(model, output, plan_only=False):
    model = model.resolve(strict=True)
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("fresh non-symlink derived view required")
    if not model.is_relative_to(ROOT) or not output.is_relative_to(ROOT):
        raise ValueError("workspace-local source and dedicated output required")
    layout_path = model / "layout-receipt.json"
    layout = json.loads(layout_path.read_text())
    if (layout.get("status") != "complete_layout_only" or (layout.get("bits"), layout.get("group_size")) != (4, 64) or
            layout.get("source_repo") != "mflux-community/flux2-klein-4b-mflux-q4" or
            layout.get("source_revision") != "794cd159538149ad9830848508c31f0ea7088e58"):
        raise ValueError("current full-payload-verified saved-Q4 native view required")
    transformer = model / "transformer"
    headers, files, metadata = {}, [], None
    for record in layout["components"]["transformer"]:
        path = transformer / record["file"]
        header, _ = read_header(path)
        current_metadata = header.get("__metadata__", {})
        if metadata is not None and current_metadata != metadata:
            raise ValueError("shards have different source metadata")
        metadata = current_metadata
        for name, item in header.items():
            if name != "__metadata__":
                if name in headers:
                    raise ValueError("duplicate source tensor")
                headers[name] = item
        files.append(path)
    index = json.loads((transformer / "model.safetensors.index.json").read_text())
    if set(index["weight_map"]) != set(headers):
        raise ValueError("complete source index required")
    budget = plan(headers, shutil.disk_usage(ROOT).free)
    if plan_only:
        return dict(status="preflight_only", performance_verified=False, budget=budget)

    source_files = [model / component / record["file"]
                    for component, records in layout["components"].items() for record in records]
    source_files += [layout_path, *sorted(model.glob("*/config.json")),
                     *sorted(model.glob("*/model.safetensors.index.json"))]
    identities = {str(path): generation(path) for path in source_files}
    hashes = registered_hashes(source_files, ROOT / "outputs/benchmark-source-registrations/flux4-native-q4.json")
    for component, records in layout["components"].items():
        for record in records:
            if hashes[str(model / component / record["file"])] != record["sha256"]:
                raise ValueError("source differs from original full-payload proof")
    check_generations(identities)
    import mlx.core as mx
    import numpy as np
    output.mkdir(parents=True)
    target = output / "derived-receipt.json"
    receipt = dict(schema="tc-flux4-hybrid-derived-source-v1", status="preparing",
        runtime_verified=False, performance_verified=False, recipe="single40-q4-g64-to-bf16-v1",
        source_view=str(model.relative_to(ROOT)), source_repo=layout["source_repo"],
        source_revision=layout["source_revision"], mlx_version=version("mlx"),
        source_sha256={str(path.relative_to(ROOT)): hashes[str(path)] for path in source_files},
        upstream_source_sha256=layout["source_sha256"], budget=budget,
        decoded_projections=prefixes(), conversions=[], unchanged_tensor_bitpatterns_verified=False,
        scope="derived from savedQ4 coefficients with BF16 rounding; only40single projections decoded; other267DiT tensors and encoder/VAE/tokenizer unchanged; not originalBF16/packedQ4 parity/activationINT8/performance")
    write_json(target, receipt)
    try:
        values = {}
        for path in files:
            arrays = mx.load(str(path))
            if any(name in values for name in arrays):
                raise ValueError("duplicate loaded source tensor")
            values.update(arrays)
        original = dict(values)
        for index, prefix in enumerate(prefixes()):
            value = mx.dequantize(values[prefix + ".weight"], values[prefix + ".scales"],
                                  values[prefix + ".biases"], group_size=64, bits=4, dtype=mx.bfloat16)
            mx.eval(value)
            if value.dtype != mx.bfloat16 or list(value.shape) != [headers[prefix + ".weight"]["shape"][0],
                                                                  headers[prefix + ".weight"]["shape"][1] * 8]:
                raise ValueError("official BF16 dequantization returned wrong geometry")
            values[prefix + ".weight"] = value
            del values[prefix + ".scales"], values[prefix + ".biases"]
            receipt["conversions"].append(dict(prefix=prefix, input_dtype="U32", bits=4, group_size=64,
                output_dtype="BF16", output_shape=list(value.shape), output_bytes=value.nbytes))
            write_json(target, receipt)
            print(f"decoded projection {index + 1}/40", flush=True)
        destination = output / "transformer"
        destination.mkdir()
        weights = destination / "diffusion_pytorch_model.safetensors"
        mx.save_safetensors(str(weights), values, dict(metadata, tc_layout="flux4-single40-derived-bf16-v1"))
        reloaded = mx.load(str(weights))
        if set(reloaded) != set(values) or len(reloaded) != budget["derived_tensors"]:
            raise ValueError("derived tensor coverage differs")
        decoded = {prefix + ".weight" for prefix in prefixes()}
        for name, value in reloaded.items():
            expected = values[name] if name in decoded else original[name]
            if value.dtype != expected.dtype or value.shape != expected.shape or not mx.all(
                    mx.view(value, mx.uint8) == mx.view(expected, mx.uint8)).item():
                raise ValueError("full saved tensor bitpattern differs: " + name)
        if sum(name.endswith(".scales") for name in reloaded) != budget["remaining_quantized_matrices"]:
            raise ValueError("untouched packed projection coverage differs")
        shutil.copyfile(transformer / "config.json", destination / "config.json")
        for component in ("text_encoder", "vae", "tokenizer"):
            (output / component).symlink_to(model / component, target_is_directory=True)
        check_generations(identities)
        receipt.update(status="complete_derived_source_only", unchanged_tensor_bitpatterns_verified=True,
            all_saved_tensor_bitpatterns_verified=True, source_generations_unchanged=True,
            checkpoint=str(weights.relative_to(ROOT)), checkpoint_bytes=weights.stat().st_size,
            checkpoint_sha256=sha256_file(weights), final_free_bytes=shutil.disk_usage(ROOT).free)
        if receipt["final_free_bytes"] < RESERVE:
            raise ValueError("post-conversion2GiB reserve violated")
    except BaseException as error:
        receipt.update(status="failed", error=str(error))
        raise
    finally:
        write_json(target, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.model, args.output, args.plan_only), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
