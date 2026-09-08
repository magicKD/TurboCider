"""Offline Z-Image-Turbo FFN partition exporter.

The exporter is intentionally repository-local and does not import ComfyUI or
diffusers.  It reads the official ComfyUI split-file, a diffusers single-file
transformer checkpoint, a Hugging Face/ModelScope safetensors index with
multiple shards, or a native MLX-readable Q8_0/Q4_0/Q4_1 GGUF transformer.  It
emits one Core ML INT8 per-channel artifact per transformer block and publishes
a provenance-bound manifest consumed by the native runtime.  GGUF weights are
dequantized one tensor at a time and are never written as a merged BF16
checkpoint.

The native implementation computes the suffix of each gated MLP on the GPU;
Core ML owns the contiguous prefix.  The model ABI is therefore fixed to
``x/y = [1, 3840, 1, bucket]`` while the partition width is recorded in the
manifest.
"""

import argparse
import fcntl
import gc
import hashlib
import json
import mmap
import os
import signal
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora import apply as apply_lora
from lora import load as load_loras
from lora import provenance as lora_provenance


HIDDEN = 3840
MLP_WIDTH = 10240
DEFAULT_BLOCKS = [f"noise_refiner.{i}" for i in range(2)] + [f"layers.{i}" for i in range(30)]


def atom(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


_CONVROT_HADAMARD = {}


def convrot_hadamard(size, np):
    cached = _CONVROT_HADAMARD.get(size)
    if cached is not None:
        return cached
    if size < 4 or size & (size - 1) or (size.bit_length() - 1) % 2:
        raise ValueError(f"ConvRot group must be a power of four: {size}")
    h4 = np.array([[1, 1, 1, -1], [1, 1, -1, 1],
                   [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float32)
    value = h4
    width = 4
    while width < size:
        value = np.kron(value, h4)
        width *= 4
    value = np.ascontiguousarray(value / np.sqrt(np.float32(size)), dtype=np.float32)
    _CONVROT_HADAMARD[size] = value
    return value


def rotate_convrot_matrix(weight, group, np):
    """Apply the symmetric block-Hadamard on a matrix's input axis."""
    if weight.ndim != 2 or weight.shape[1] % group:
        raise ValueError(f"weight shape {weight.shape} is not ConvRot-{group} compatible")
    rotated = weight.astype(np.float32, copy=False).reshape(-1, group) @ convrot_hadamard(group, np)
    return np.ascontiguousarray(rotated.reshape(weight.shape))


def convrot_activation_weight(channels, group, np):
    if channels % group:
        raise ValueError(f"activation channels {channels} are not ConvRot-{group} aligned")
    tiled = np.tile(convrot_hadamard(group, np), (channels // group, 1))
    return np.ascontiguousarray(tiled.astype(np.float16)[:, :, None, None])


def apply_export_lora(name, base, bundles, np, convrot_group=None):
    if convrot_group is None:
        return apply_lora(name, base, bundles, np)
    # ConvRot checkpoints store W@H, while ordinary LoRA files describe a
    # delta in the original W domain. Rotate only the delta before adding it
    # to the already-rotated base, preserving the independent LoRA file.
    delta, applied = apply_lora(name, np.zeros(base.shape, dtype=np.float32), bundles, np)
    if not applied:
        return base, 0
    merged = base.astype(np.float32) + rotate_convrot_matrix(delta, convrot_group, np)
    return merged.astype(base.dtype), applied


class SafetensorsSource:
    """One safetensors file or an index-bound set of immutable shards."""

    def __init__(self, checkpoint: Path, weight_map=None, convrot_mode="native"):
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise ValueError(f"checkpoint must be a regular non-symlink file: {checkpoint}")
        if convrot_mode not in {"native", "derotate"}:
            raise ValueError("convrot_mode must be native or derotate")
        self.checkpoint = checkpoint.resolve()
        self.weight_map = weight_map
        self.convrot_mode = convrot_mode
        self.shards = []
        if weight_map is not None:
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("safetensors index weight_map must be a nonempty object")
            names = set()
            for tensor_name, shard_name in weight_map.items():
                if not isinstance(tensor_name, str) or not tensor_name or not isinstance(shard_name, str):
                    raise ValueError("safetensors index contains an invalid weight mapping")
                relative = Path(shard_name)
                if relative.is_absolute() or relative.name != shard_name or shard_name in {".", ".."}:
                    raise ValueError(f"safetensors shard must be a sibling filename: {shard_name}")
                names.add(shard_name)
            for name in sorted(names):
                shard = self.checkpoint.parent / name
                if shard.is_symlink() or not shard.is_file() or shard.suffix != ".safetensors":
                    raise ValueError(f"indexed safetensors shard missing or invalid: {shard}")
                self.shards.append(shard.resolve())
        else:
            self.shards = [self.checkpoint]
        self._streams = {}
        self._memories = {}
        self._headers = {}
        self._data_offsets = {}
        self.has_convrot = False
        for path in self.shards:
            with path.open("rb") as stream:
                _, header = load_header(stream)
            if any(name.endswith(".comfy_quant") for name in header):
                self.has_convrot = True

    @classmethod
    def from_model(cls, model: Path, convrot_mode="native"):
        if model.is_file() and model.suffix.lower() == ".safetensors":
            return cls(model, convrot_mode=convrot_mode)
        index = model / "transformer/diffusion_pytorch_model.safetensors.index.json"
        if index.is_file() and not index.is_symlink():
            value = json.loads(index.read_text())
            if not isinstance(value, dict):
                raise ValueError("safetensors index must be a JSON object")
            return cls(index, value.get("weight_map"), convrot_mode=convrot_mode)
        single_candidates = [
            model / "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
            model / "transformer/diffusion_pytorch_model.safetensors",
        ]
        for candidate in single_candidates:
            if candidate.is_file() and not candidate.is_symlink():
                return cls(candidate, convrot_mode=convrot_mode)
        raise ValueError("Z-Image transformer checkpoint or safetensors index not found under model root")

    def provenance(self):
        result = {
            "checkpoint": str(self.checkpoint),
            "checkpoint_bytes": self.checkpoint.stat().st_size,
            "checkpoint_sha256": sha(self.checkpoint),
        }
        if self.weight_map is not None:
            result["checkpoint_shards"] = {
                shard.name: {"bytes": shard.stat().st_size, "sha256": sha(shard)}
                for shard in self.shards
            }
        return result

    def validate_unchanged(self, provenance) -> None:
        if (self.checkpoint.stat().st_size != provenance["checkpoint_bytes"] or
                sha(self.checkpoint) != provenance["checkpoint_sha256"]):
            raise ValueError("checkpoint changed during export")
        expected = provenance.get("checkpoint_shards", {})
        if self.weight_map is not None and set(expected) != {path.name for path in self.shards}:
            raise ValueError("checkpoint shard set changed during export")
        for shard in self.shards:
            if self.weight_map is None:
                continue
            value = expected[shard.name]
            if shard.stat().st_size != value["bytes"] or sha(shard) != value["sha256"]:
                raise ValueError(f"checkpoint shard changed during export: {shard.name}")

    def __enter__(self):
        try:
            for path in self.shards:
                stream = path.open("rb")
                memory = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
                header_size, header = load_header(stream)
                if 8 + header_size > len(memory) or not isinstance(header, dict):
                    raise ValueError(f"invalid safetensors header: {path}")
                self._streams[path] = stream
                self._memories[path] = memory
                self._headers[path] = header
                self._data_offsets[path] = 8 + header_size
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        for memory in self._memories.values():
            memory.close()
        for stream in self._streams.values():
            stream.close()
        self._streams.clear()
        self._memories.clear()
        self._headers.clear()
        self._data_offsets.clear()

    def _tensor_record(self, name):
        path = self.tensor_path(name)
        header = self._headers.get(path)
        memory = self._memories.get(path)
        if header is None or memory is None:
            raise ValueError(f"safetensors shard is not open: {path}")
        meta = header.get(name)
        if not isinstance(meta, dict):
            raise ValueError(f"tensor missing from safetensors checkpoint: {name}")
        return path, memory, meta, self._data_offsets[path]

    def _array(self, name, np):
        _, memory, meta, data_offset = self._tensor_record(name)
        shape = tuple(int(value) for value in meta.get("shape", []))
        dtype = meta.get("dtype")
        start, end = meta["data_offsets"]
        count = int(np.prod(shape))
        dtypes = {"F32": ("<f4", 4), "F16": ("<f2", 2),
                  "I8": ("i1", 1), "U8": ("u1", 1)}
        if dtype not in dtypes:
            raise ValueError(f"unsupported raw safetensors dtype for {name}: {dtype}")
        numpy_dtype, itemsize = dtypes[dtype]
        if start < 0 or end - start != count * itemsize or data_offset + end > len(memory):
            raise ValueError(f"invalid safetensors offsets: {name}")
        return np.frombuffer(memory, dtype=numpy_dtype, count=count,
                             offset=data_offset + start).copy().reshape(shape)

    def convrot_group(self, name, np):
        if not name.endswith(".weight"):
            return None
        prefix = name[:-len(".weight")]
        marker_name = prefix + ".comfy_quant"
        try:
            marker = self._array(marker_name, np)
        except ValueError:
            return None
        try:
            descriptor = json.loads(marker.astype(np.uint8, copy=False).tobytes().decode())
        except Exception as error:
            raise ValueError(f"invalid ConvRot marker for {name}: {error}") from error
        group = descriptor.get("convrot_groupsize")
        if (descriptor.get("format") != "int8_tensorwise" or
                descriptor.get("convrot") is not True or
                not isinstance(group, int) or group < 4 or group & (group - 1) or
                (group.bit_length() - 1) % 2):
            raise ValueError(f"unsupported ConvRot descriptor for {name}: {descriptor}")
        return group

    def tensor(self, name, shape, np):
        _, memory, meta, data_offset = self._tensor_record(name)
        if meta.get("shape") != list(shape):
            raise ValueError(f"unexpected tensor shape: {name}")
        if meta.get("dtype") == "I8":
            group = self.convrot_group(name, np)
            if group is None or len(shape) != 2 or shape[1] % group:
                raise ValueError(f"unsupported INT8 tensor without valid ConvRot metadata: {name}")
            quantized = self._array(name, np).astype(np.float32)
            scale_name = name[:-len(".weight")] + ".weight_scale"
            scale = self._array(scale_name, np)
            if scale.dtype != np.float32 or scale.shape != (shape[0], 1):
                raise ValueError(f"invalid ConvRot scale tensor: {scale_name}")
            value = quantized * scale
            if self.convrot_mode == "derotate":
                value = rotate_convrot_matrix(value, group, np)
            return value.astype(np.float16)
        if meta.get("dtype") not in {"BF16", "F16", "F32"}:
            raise ValueError(f"unexpected tensor dtype: {name}")
        start, end = meta["data_offsets"]
        count = int(np.prod(shape))
        itemsize = 4 if meta["dtype"] == "F32" else 2
        if start < 0 or end - start != count * itemsize or data_offset + end > len(memory):
            raise ValueError(f"invalid safetensors offsets: {name}")
        if meta["dtype"] == "BF16":
            words = np.frombuffer(memory, dtype="<u2", count=count, offset=data_offset + start)
            return (words.astype(np.uint32) << 16).view(np.float32).astype(np.float16).reshape(shape)
        dtype = "<f4" if meta["dtype"] == "F32" else "<f2"
        return np.frombuffer(memory, dtype=dtype, count=count,
                             offset=data_offset + start).copy().astype(np.float16).reshape(shape)

    def tensor_path(self, name):
        if self.weight_map is None:
            return self.shards[0]
        shard_name = self.weight_map.get(name)
        if not isinstance(shard_name, str):
            raise ValueError(f"tensor missing from safetensors index: {name}")
        return (self.checkpoint.parent / shard_name).resolve()


class GGUFSource:
    """MLX-backed GGUF source for affine Q4/Q8 tensors."""

    FILE_TYPES = {2: "q4_0", 3: "q4_1", 7: "q8_0"}

    def __init__(self, checkpoint: Path):
        if checkpoint.is_symlink() or not checkpoint.is_file() or checkpoint.suffix.lower() != ".gguf":
            raise ValueError(f"GGUF checkpoint must be a regular non-symlink file: {checkpoint}")
        self.checkpoint = checkpoint.resolve()
        self.weight_map = None
        self.shards = [self.checkpoint]
        self._arrays = None
        self._mx = None
        self.quantization = None

    def provenance(self):
        return {
            "checkpoint": str(self.checkpoint),
            "checkpoint_bytes": self.checkpoint.stat().st_size,
            "checkpoint_sha256": sha(self.checkpoint),
            "checkpoint_format": "gguf",
        }

    def validate_unchanged(self, provenance) -> None:
        if (self.checkpoint.stat().st_size != provenance["checkpoint_bytes"] or
                sha(self.checkpoint) != provenance["checkpoint_sha256"]):
            raise ValueError("GGUF checkpoint changed during export")

    def __enter__(self):
        import mlx.core as mx
        arrays, metadata = mx.load(self.checkpoint, return_metadata=True, stream=mx.cpu)
        if not isinstance(arrays, dict) or not arrays:
            raise ValueError("GGUF checkpoint contains no MLX-readable tensors")
        file_type = metadata.get("general.file_type") if isinstance(metadata, dict) else None
        if file_type is None:
            raise ValueError("GGUF checkpoint has no general.file_type metadata")
        mx.eval(file_type)
        file_type = int(file_type.item())
        if file_type not in self.FILE_TYPES:
            raise ValueError(
                f"native Core ML GGUF export supports Q4_0/Q4_1/Q8_0 only; "
                f"general.file_type={file_type}"
            )
        self.quantization = self.FILE_TYPES[file_type]
        self._mx = mx
        self._arrays = arrays
        return self

    def __exit__(self, *_):
        if self._arrays is not None:
            self._arrays.clear()
        self._arrays = None
        if self._mx is not None:
            self._mx.clear_cache()
        self._mx = None

    def tensor(self, name, shape, np):
        if self._arrays is None or self._mx is None:
            raise ValueError("GGUF source is not open")
        value = self._arrays.get(name)
        if value is None:
            raise ValueError(f"tensor missing from GGUF checkpoint: {name}")
        mx = self._mx
        if value.dtype == mx.uint32:
            prefix = name.removesuffix(".weight")
            scales = self._arrays.get(prefix + ".scales")
            biases = self._arrays.get(prefix + ".biases")
            if scales is None:
                raise ValueError(f"quantized GGUF tensor has no scales: {name}")
            if len(shape) != 2 or tuple(value.shape)[0] != shape[0] or tuple(scales.shape)[0] != shape[0]:
                raise ValueError(f"unexpected GGUF tensor geometry: {name}")
            logical_input = shape[1]
            if logical_input % int(scales.shape[1]) != 0 or (int(value.shape[1]) * 32) % logical_input != 0:
                raise ValueError(f"invalid GGUF affine geometry: {name}")
            group_size = logical_input // int(scales.shape[1])
            bits = int(value.shape[1]) * 32 // logical_input
            if group_size != 32 or bits not in {4, 8}:
                raise ValueError(
                    f"unsupported native Core ML GGUF quantization for {name}: "
                    f"group_size={group_size}, bits={bits}; expected Q4_0/Q4_1/Q8_0"
                )
            expected_bits = 8 if self.quantization == "q8_0" else 4
            if bits != expected_bits:
                raise ValueError("GGUF tensor geometry does not match general.file_type")
            dense = mx.dequantize(value, scales, biases, group_size=group_size,
                                  bits=bits, mode="affine", dtype=mx.float16,
                                  stream=mx.cpu)
        else:
            if tuple(value.shape) != tuple(shape) or value.dtype not in {mx.float16, mx.bfloat16, mx.float32}:
                raise ValueError(f"unexpected floating GGUF tensor shape/dtype: {name}")
            dense = mx.astype(value, mx.float16, stream=mx.cpu)
        mx.eval(dense)
        result = np.array(dense, dtype=np.float16, copy=True).reshape(shape)
        del dense
        mx.clear_cache()
        return result


def source_from_model(model: Path, convrot_mode="native"):
    if model.is_file() and model.suffix.lower() == ".gguf":
        return GGUFSource(model)
    return SafetensorsSource.from_model(model, convrot_mode=convrot_mode)


def checkpoint_for(model: Path) -> Path:
    """Compatibility helper used by external tooling."""
    return source_from_model(model).checkpoint


def parse_blocks(value: str, blocks=DEFAULT_BLOCKS):
    if value == "all":
        return list(range(len(blocks)))
    result = []
    for item in value.split(","):
        number = int(item)
        if number < 0 or number >= len(blocks):
            raise ValueError("block index must be 0...31")
        if number not in result:
            result.append(number)
    if not result:
        raise ValueError("at least one block is required")
    return result


def load_header(stream):
    raw = stream.read(8)
    if len(raw) != 8:
        raise ValueError("invalid safetensors header length")
    header_size = struct.unpack("<Q", raw)[0]
    if header_size > 100_000_000:
        raise ValueError("safetensors header is unreasonably large")
    header = json.loads(stream.read(header_size))
    return header_size, header


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-kind", choices=["z-image", "llada-image"], default="z-image",
                        help="transformer family sharing the 3840/10240 gated-FFN ABI")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bucket", type=int, default=4608)
    parser.add_argument("--ane-mlp-width", type=int, default=7680)
    parser.add_argument("--activation-scale", type=float, default=8.0)
    parser.add_argument("--output-scale", type=float, default=32.0)
    parser.add_argument("--variant", choices=["int8_pc", "fp16"], default="int8_pc")
    parser.add_argument("--convrot-mode", choices=["native", "derotate"], default="native",
                        help="keep ConvRot weights rotated and emit online FWHT, or derotate offline")
    parser.add_argument("--blocks", default="all", help="all or comma-separated block indexes")
    parser.add_argument("--lora", action="append", default=[],
                        help="independent transformer LoRA file; repeat for multiple adapters")
    parser.add_argument("--lora-strength", action="append", default=[],
                        help="strength matching each --lora (default 1.0)")
    parser.add_argument("--lora-role", default="transformer")
    args = parser.parse_args()
    if not 64 <= args.bucket <= 8192:
        raise ValueError("bucket must be 64...8192")
    if args.ane_mlp_width <= 0 or args.ane_mlp_width >= MLP_WIDTH:
        raise ValueError("ane-mlp-width must be in 1...10239")
    if not 1.0 <= args.activation_scale <= 64.0:
        raise ValueError("activation-scale must be 1...64")
    if not 1.0 <= args.output_scale <= 256.0:
        raise ValueError("output-scale must be 1...256")
    blocks = list(DEFAULT_BLOCKS)
    block_indexes = parse_blocks(args.blocks, blocks)

    import numpy as np
    import coremltools as ct
    import coremltools.optimize.coreml as optimize
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    checkpoint_source = source_from_model(args.model, args.convrot_mode)
    checkpoint = checkpoint_source.checkpoint
    if args.lora_strength and len(args.lora_strength) != len(args.lora):
        raise ValueError("--lora-strength must be repeated once per --lora")
    lora_strengths = [float(value) for value in args.lora_strength] or [1.0] * len(args.lora)
    lora_roles = [args.lora_role] * len(args.lora)
    lora_records = lora_provenance(args.lora, lora_strengths, lora_roles)
    lora_bundles = load_loras(args.lora, lora_strengths, lora_roles, np)
    output = args.output.absolute()
    if output.is_symlink():
        raise ValueError("symlink output unsupported")
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".export.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        provenance = checkpoint_source.provenance()
        identity = {
            "owner": ("turbocider.z_image.coreml.v1" if args.model_kind == "z-image"
                      else "turbocider.llada_image.coreml.v1"),
            **provenance,
            "bucket": args.bucket,
            "hidden": HIDDEN,
            "mlp_width": MLP_WIDTH,
            "ane_mlp_start": 0,
            "ane_mlp_end": args.ane_mlp_width,
            "activation_scale": args.activation_scale,
            "output_scale": args.output_scale,
            "blocks": block_indexes,
            "variant": args.variant,
            "coremltools": ct.__version__,
            "numpy": np.__version__,
            "checkpoint_format": "gguf" if isinstance(checkpoint_source, GGUFSource)
                                 else ("convrot_int8" if getattr(checkpoint_source, "has_convrot", False)
                                       else "safetensors"),
            "convrot_mode": args.convrot_mode if getattr(checkpoint_source, "has_convrot", False) else "none",
            "recipe": 4 if isinstance(checkpoint_source, GGUFSource)
                      else (6 if getattr(checkpoint_source, "has_convrot", False) and args.convrot_mode == "native"
                            else (5 if getattr(checkpoint_source, "has_convrot", False) else
                                  (2 if checkpoint_source.weight_map is not None else 1))),
        }
        if args.model_kind != "z-image":
            identity["model_kind"] = args.model_kind
        if lora_records:
            identity["loras"] = lora_records
            identity["recipe"] = 3
        marker = output / ".turbocider-export.json"
        if marker.exists():
            if marker.is_symlink() or json.loads(marker.read_text()) != identity:
                raise ValueError("export identity changed; select a new output directory")
        else:
            allowed = {".export.lock", "export.log", "progress.json"}
            if any(item.name not in allowed for item in output.iterdir()):
                raise ValueError("refusing to adopt a nonempty export directory")
            atom(marker, identity)

        with checkpoint_source as reader:
            artifacts = {}
            checksums = {}
            convrot_groups = set()
            for ordinal in block_indexes:
                prefix = blocks[ordinal]
                name = f"block{ordinal}_mlp_branch.int8_pc.mlpackage"
                destination = output / name
                receipt = output / f"block{ordinal}.json"
                if destination.is_symlink() or receipt.is_symlink():
                    raise ValueError("symlink artifact unsupported")
                if destination.exists() and receipt.exists():
                    saved = json.loads(receipt.read_text())
                    valid = all((destination / relative).is_file() and sha(destination / relative) == digest
                                for relative, digest in saved.items())
                    if not valid:
                        raise ValueError(f"existing export corrupted: block {ordinal}")
                    checksums[str(ordinal)] = saved
                else:
                    if destination.exists():
                        raise ValueError(f"incomplete artifact must be removed: block {ordinal}")
                    w1_name = f"{prefix}.feed_forward.w1.weight"
                    w2_name = f"{prefix}.feed_forward.w2.weight"
                    w3_name = f"{prefix}.feed_forward.w3.weight"
                    groups = [reader.convrot_group(name, np)
                              if isinstance(reader, SafetensorsSource) else None
                              for name in (w1_name, w2_name, w3_name)]
                    if any(group is not None for group in groups) and len(set(groups)) != 1:
                        raise ValueError(f"inconsistent ConvRot metadata in {prefix}: {groups}")
                    native_group = (groups[0] if groups[0] is not None and
                                    checkpoint_source.convrot_mode == "native" else None)
                    if native_group is not None:
                        if HIDDEN % native_group or args.ane_mlp_width % native_group:
                            raise ValueError("native ConvRot ANE split must align to its Hadamard group")
                        convrot_groups.add(native_group)
                    w1 = reader.tensor(w1_name, (MLP_WIDTH, HIDDEN), np)
                    w2 = reader.tensor(w2_name, (HIDDEN, MLP_WIDTH), np)
                    w3 = reader.tensor(w3_name, (MLP_WIDTH, HIDDEN), np)
                    w1, _ = apply_export_lora(w1_name, w1, lora_bundles, np, native_group)
                    w2, _ = apply_export_lora(w2_name, w2, lora_bundles, np, native_group)
                    w3, _ = apply_export_lora(w3_name, w3, lora_bundles, np, native_group)
                    width = args.ane_mlp_width
                    first = np.ascontiguousarray(np.concatenate((w1[:width], w3[:width]), axis=0)[:, :, None, None])
                    # Z-Image's BF16 gated activations can exceed FP16 range at
                    # the elementwise product even though the final projection
                    # is finite.  Scale both multiplicands before the product
                    # and compensate exactly in w2: (a/s)*(b/s)*(w2*s^2).
                    # Per-channel symmetric quantization is invariant to this
                    # uniform row scale apart from normal FP16 rounding.
                    scale = np.float32(args.activation_scale)
                    last = np.ascontiguousarray(
                        (w2[:, :width].astype(np.float32) * (scale * scale / np.float32(args.output_scale))).astype(np.float16)[:, :, None, None]
                    )
                    del w1, w2, w3
                    rotation_hidden = (convrot_activation_weight(HIDDEN, native_group, np)
                                       if native_group is not None else None)
                    rotation_width = (convrot_activation_weight(width, native_group, np)
                                      if native_group is not None else None)

                    @mb.program(
                        input_specs=[mb.TensorSpec(shape=(1, HIDDEN, 1, args.bucket), dtype=types.fp16)],
                        opset_version=ct.target.macOS15,
                    )
                    def branch(x):
                        projected_input = x
                        if native_group is not None:
                            projected_input = mb.conv(
                                x=x, weight=rotation_hidden, groups=HIDDEN // native_group,
                                pad_type="valid", name="convrot_input")
                        projected = mb.conv(x=projected_input, weight=first,
                                            pad_type="valid", name="projected")
                        gate, up = mb.split(x=projected, num_splits=2, axis=1)
                        reciprocal = np.float16(1.0 / args.activation_scale)
                        gated = mb.mul(x=mb.silu(x=gate), y=reciprocal)
                        raised = mb.mul(x=up, y=reciprocal)
                        last_input = mb.mul(x=gated, y=raised)
                        if native_group is not None:
                            last_input = mb.conv(
                                x=last_input, weight=rotation_width,
                                groups=width // native_group,
                                pad_type="valid", name="convrot_mlp")
                        return mb.conv(x=last_input, weight=last,
                                       pad_type="valid", name="y")

                    model = ct.convert(
                        branch,
                        convert_to="mlprogram",
                        minimum_deployment_target=ct.target.macOS15,
                        compute_precision=ct.precision.FLOAT16,
                        skip_model_load=True,
                    )
                    if args.variant == "int8_pc":
                        quantizer = optimize.OptimizationConfig(
                            global_config=optimize.OpLinearQuantizerConfig(
                                mode="linear_symmetric", dtype="int8", granularity="per_channel",
                                block_size=32, weight_threshold=0
                            )
                        )
                        compressed = optimize.linear_quantize_weights(model, quantizer)
                    else:
                        compressed = model
                    with tempfile.TemporaryDirectory(prefix=".export-", dir=output) as temporary:
                        package = Path(temporary) / name
                        compressed.save(package)
                        os.replace(package, destination)
                    saved = {str(path.relative_to(destination)): sha(path)
                             for path in sorted(destination.rglob("*")) if path.is_file()}
                    atom(receipt, saved)
                    checksums[str(ordinal)] = saved
                    del first, last, rotation_hidden, rotation_width, branch, model, compressed
                    gc.collect()
                artifacts[str(ordinal)] = {"int8_pc": name}
                atom(output / "progress.json", {"completed": len(artifacts), "total": len(block_indexes)})
                print(f"partition {len(artifacts)}/{len(block_indexes)} ready", flush=True)

        checkpoint_source.validate_unchanged(provenance)
        if lora_records != lora_provenance(args.lora, lora_strengths, lora_roles):
            raise ValueError("LoRA changed during export")
        for bundle in lora_bundles:
            if bundle["role"] == "transformer" and bundle["applied"] == 0:
                raise ValueError(f"LoRA did not match an exported Z-Image FFN: {bundle['path']}")
        source_manifest = {**provenance, "blocks": block_indexes}
        if args.model_kind != "z-image":
            source_manifest["model_kind"] = args.model_kind
        if isinstance(checkpoint_source, GGUFSource):
            source_manifest["gguf_quantization"] = checkpoint_source.quantization
        if getattr(checkpoint_source, "has_convrot", False):
            source_manifest["convrot"] = {
                "mode": checkpoint_source.convrot_mode,
                "groups": sorted(convrot_groups),
            }
        if lora_records:
            source_manifest["loras"] = lora_records
        atom(output / "manifest.json", {
            "schema_version": 2,
            "source": source_manifest,
            "shape": {
                "K": HIDDEN,
                "N": HIDDEN,
                "mlp_width": MLP_WIDTH,
                "ane_mlp_start": 0,
                "ane_mlp_end": args.ane_mlp_width,
                "activation_scale": args.activation_scale,
                "output_scale": args.output_scale,
                "buckets": [args.bucket],
            },
            "functions": {str(args.bucket): "main"},
            "artifacts": artifacts,
            "artifact_sha256": checksums,
            "export_identity": identity,
        })
        print(json.dumps({"source_manifest": str(output / "manifest.json"),
                          "checkpoint": str(checkpoint), "blocks": block_indexes}), flush=True)


if __name__ == "__main__":
    main()
