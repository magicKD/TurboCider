#!/usr/bin/env python3
"""Same affine g32 tensor payload in mflux/native key layouts; original stays RO."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile

from flux9_bf16_framework_screen import generation, check_generations, write_json
from mflux_original_layout import read_header
from runtime_ane_common import sha256_file

ROOT = Path(__file__).resolve().parents[2]


def native_key(name, inverse):
    if name in inverse:
        return inverse[name]
    for suffix in (".scales", ".biases"):
        if name.endswith(suffix):
            weight = name[:-len(suffix)] + ".weight"
            if weight in inverse:
                return inverse[weight][:-7] + suffix
    raise ValueError("saved learned tensor lacks a native source key: " + name)


def mirrored_header(header, inverse):
    result = {"__metadata__": dict(header.get("__metadata__", {}), tc_layout="same-affine-g32-native-v1")}
    for name, value in header.items():
        if name != "__metadata__":
            target = native_key(name, inverse)
            if target in result:
                raise ValueError("native layout key collision")
            result[target] = value
    return result


def stream_component(arrays, destination, inverse, mx, bits):
    # Each temporary contains just one actual tensor, never a full dense
    # checkpoint. Native header clones reserve room without shifting payload.
    items = sorted(arrays.items())
    header, size = {}, 0
    dtype_names = {str(mx.bfloat16): "BF16", str(mx.float16): "F16", str(mx.float32): "F32", str(mx.uint32): "U32"}
    for name, value in items:
        dtype = dtype_names.get(str(value.dtype))
        if dtype is None:
            raise ValueError("unsupported saved parameter dtype: " + str(value.dtype))
        header[name] = dict(dtype=dtype, shape=list(value.shape), data_offsets=[size, size + value.nbytes])
        size += value.nbytes
    header["__metadata__"] = dict(quantization_level=str(bits), mflux_version="0.19.2",
                                 group_size="32", tc_scope="matched-linear-only; embeddings and VAE original BF16")
    native = mirrored_header(header, inverse)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    other = json.dumps(native, separators=(",", ":")).encode()
    capacity = ((max(len(encoded), len(other)) + 65535) // 65536) * 65536
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output, tempfile.TemporaryDirectory(prefix="tensor-piece-", dir=destination.parent) as raw:
        output.write(struct.pack("<Q", capacity))
        output.write(encoded.ljust(capacity, b" "))
        piece = Path(raw) / "piece.safetensors"
        for name, value in items:
            mx.eval(value)
            mx.save_safetensors(str(piece), {"tensor": value})
            actual, prefix = read_header(piece)
            item = actual["tensor"]
            if item["dtype"] != header[name]["dtype"] or item["shape"] != header[name]["shape"]:
                raise ValueError("temporary tensor serialization changed dtype or shape")
            with piece.open("rb") as source:
                source.seek(len(prefix))
                shutil.copyfileobj(source, output, 4 << 20)
            piece.unlink()
        output.flush()
        os.fsync(output.fileno())
    if destination.stat().st_size != size + capacity + 8:
        raise ValueError("complete original quantized payload extent required")
    return header, native, capacity


def clone_native(source, destination, header, capacity):
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    clone = library.clonefile
    clone.argtypes, clone.restype = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int], ctypes.c_int
    destination.parent.mkdir(parents=True, exist_ok=True)
    if clone(str(source).encode(), str(destination).encode(), 0):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))
    encoded = json.dumps(header, separators=(",", ":")).encode()
    if len(encoded) > capacity:
        raise ValueError("native view exceeds fixed reserved header")
    with destination.open("r+b") as output:
        output.seek(8)
        output.write(encoded.ljust(capacity, b" "))
        output.flush()
        os.fsync(output.fileno())
    source_payload, native_payload = hashlib.sha256(), hashlib.sha256()
    with source.open("rb") as a, destination.open("rb") as b:
        a.seek(8 + capacity); b.seek(8 + capacity)
        while chunk := a.read(4 << 20):
            peer = b.read(len(chunk))
            if peer != chunk:
                raise ValueError("entire shared compressed tensor payload differs")
            source_payload.update(chunk); native_payload.update(peer)
        if b.read(1):
            raise ValueError("extra native payload")
    destination.chmod(0o400)
    source.chmod(0o400)
    return dict(full_compressed_payload_equal=True, payload_sha256=source_payload.hexdigest(),
                mflux_sha256=sha256_file(source), native_sha256=sha256_file(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mflux", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", type=int, choices=(4, 8), required=True)
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(args.mflux.resolve() / "src"))
    import mlx.core as mx
    from mlx import nn
    from mlx.utils import tree_flatten
    from mflux.models.flux2.variants import Flux2Klein
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition
    from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
    from mflux_weight_coverage import inspect_reference_coverage
    mx.set_default_device(mx.cpu)
    model_root, output = args.model.resolve(strict=True), args.output.absolute()
    if output.exists():
        parser.error("fresh matched quantized pack destination required")
    estimate = (10 if args.bits == 4 else 18) << 30
    if shutil.disk_usage(ROOT).free < estimate + (2 << 30):
        parser.error("bounded matched pack plus2GiB reserve unavailable; do not delete originals")
    sources = sorted([*model_root.rglob("*.safetensors"), *model_root.rglob("*.json"), *model_root.rglob("*.jinja")])
    identities = {str(path): generation(path) for path in sources}
    hashes = {str(path): sha256_file(path) for path in sources}
    output.mkdir(parents=True)
    report = dict(schema="tc-flux9-matched-affine-pack-v1", status="preparing", bits=args.bits, group_size=32,
        source_sha256=hashes, source_generations=identities, components={},
        scope="same g32 compressed planes for native and configured mflux; stock defaultg64 is separately measured; original embeddings/VAE unchanged")
    target = output / "conversion.json"
    write_json(target, report)
    try:
        model = Flux2Klein(model_path=str(model_root), model_config=ModelConfig.flux2_klein_9b(), quantize=None)
        definitions = Flux2KleinWeightDefinition.get_components()
        report["required_coverage"] = inspect_reference_coverage(model_root, Flux2KleinWeightDefinition,
            dict(vae=model.vae, transformer=model.transformer, text_encoder=model.text_encoder))
        for name in ("transformer", "text_encoder"):
            module = getattr(model, name)
            nn.quantize(module, group_size=32, bits=args.bits,
                        class_predicate=lambda _, layer: isinstance(layer, nn.Linear))
            component = next(value for value in definitions if value.name == name)
            parameter_names = {key: None for key, _ in tree_flatten(module.parameters())}
            layers = WeightMapper._detect_num_layers(parameter_names)
            flat = WeightMapper._build_flat_mapping(component.mapping_getter(), 24, layers)
            inverse = {}
            for source, mappings in flat.items():
                for mapped, _ in mappings:
                    if mapped in inverse and inverse[mapped] != source:
                        raise ValueError("ambiguous native source mapping")
                    inverse[mapped] = source
            file = output / "mflux" / name / "0.safetensors"
            native_file = output / "native" / name / "0.safetensors"
            header, native, capacity = stream_component(dict(tree_flatten(module.parameters())), file, inverse, mx, args.bits)
            report["components"][name] = clone_native(file, native_file, native, capacity)
            write_json(target, report)
        for root in (output / "native", output / "mflux"):
            (root / "tokenizer").symlink_to(model_root / "tokenizer", target_is_directory=True)
            (root / "vae").symlink_to(model_root / "vae", target_is_directory=True)
        for name in ("transformer", "text_encoder"):
            (output / "native" / name / "config.json").symlink_to(model_root / name / "config.json")
        check_generations(identities)
        report["status"] = "complete_conversion_only"
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(target, report)
    print(json.dumps(dict(status=report["status"], bits=args.bits)))


if __name__ == "__main__":
    main()
