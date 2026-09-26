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
from z_image_smoothquant import (load_calibration_union,
                                 smooth_partial_ffn,
                                 calibrate_input_a8, calibrate_hidden_a8,
                                 calibrate_adaptive_hidden_a8,
                                 calibrate_hidden_channel_groups, route_channel_indices)


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
    parser.add_argument("--shape-mode", choices=["fixed", "enumerated", "range"], default="fixed")
    parser.add_argument("--min-bucket", type=int, default=1056,
                        help="minimum and default row count for flexible exports")
    parser.add_argument("--bucket-step", type=int, default=32)
    parser.add_argument("--ane-mlp-width", type=int, default=7680)
    parser.add_argument("--row-split-probe", action="store_true",
                        help="single-block W8A8 probe: full FFN width on a prefix of image-token rows; "
                             "not accepted by the native hybrid runtime")
    parser.add_argument("--activation-scale", type=float, default=8.0)
    parser.add_argument("--output-scale", type=float, default=32.0)
    parser.add_argument("--variant", choices=["int8_pc", "fp16"], default="int8_pc")
    parser.add_argument("--activation-precision", choices=["fp16", "int8"], default="fp16",
                        help="int8 requires real, per-block FFN input calibration")
    parser.add_argument("--calibration-dir", type=Path,
                        help="block0/ ... block31/ of .npy samples, each [bucket, 3840] FP16")
    parser.add_argument("--extra-calibration-dir", type=Path, action="append", default=[],
                        help="additional independent block directories; digest binds ordered union")
    parser.add_argument("--calibration-source-rows", type=int,
                        help="read larger captured rows but calibrate only the exported prefix; "
                             "1024-row image-only probe from 1056-row Z-Image captures")
    parser.add_argument("--sq-alpha1", type=float, default=0.5)
    parser.add_argument("--sq-alpha2", type=float, default=0.5)
    parser.add_argument("--sq-hidden-rows", type=int, default=32)
    parser.add_argument("--input-a8-search", action="store_true",
                        help="search input A8 clipping by calibrated partial-FFN output error")
    parser.add_argument("--outlier-rows", action="store_true",
                        help="include high-amplitude FFN input rows in calibration statistics")
    parser.add_argument("--region-image-rows", type=int,
                        help="experimental image/context split with independent A8 scales")
    parser.add_argument("--shared-region-input-qdq", action="store_true",
                        help="one input Q/DQ for image/caption via fixed per-row pre/post scales")
    parser.add_argument("--sq-hidden-image-only", action="store_true",
                        help="derive hidden-channel SmoothQuant scales from image rows only")
    parser.add_argument("--hidden-a8-channel-groups", type=int,
                        help="experimental image-row hidden A8 scales per channel group; "
                             "1024-row image-only export needs no separate caption region")
    parser.add_argument("--adaptive-hidden-a8-bins", type=int,
                        help="experimental fixed hidden A8 scale bank selected per image token")
    parser.add_argument("--adaptive-hidden-a8-shared-qdq", action="store_true",
                        help="experimental one-Q/DQ bank via per-token pre/post scale")
    parser.add_argument("--image-hidden-sq-alpha2", type=float,
                        help="experimental image-only hidden SQ after SwiGLU; preserve "
                             "the original W8 projections and caption path")
    parser.add_argument("--image-hidden-sq-blocks",
                        help="comma-separated subset for image-only hidden SQ; default all exported blocks")
    parser.add_argument("--adaptive-caption-a8-bins", type=int,
                        help="experimental fixed hidden A8 scale bank per caption token")
    parser.add_argument("--adaptive-caption-a8-blocks",
                        help="comma-separated subset for caption A8; default exported non-refiner blocks")
    parser.add_argument("--channel-routing", type=Path,
                        help="training-only ANE channel groups; GPU packs the exact complement")
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
    if args.shape_mode != "fixed" and (not 64 <= args.min_bucket <= args.bucket or
            args.bucket_step <= 0 or (args.bucket - args.min_bucket) % args.bucket_step):
        raise ValueError("flexible buckets must form an aligned bounded range")
    buckets = ([args.bucket] if args.shape_mode == "fixed" else
               list(range(args.min_bucket, args.bucket + 1, args.bucket_step)))
    if args.shape_mode == "enumerated" and not 2 <= len(buckets) <= 128:
        raise ValueError("enumerated export requires 2...128 shapes")
    if args.ane_mlp_width <= 0 or args.ane_mlp_width > MLP_WIDTH or (
            args.ane_mlp_width == MLP_WIDTH and not args.row_split_probe):
        raise ValueError("ane-mlp-width must be in 1...10239 outside the full-width row probe")
    if args.row_split_probe and not (
            args.ane_mlp_width == MLP_WIDTH and args.bucket in (256, 384, 416, 512, 544, 768, 1024) and
            args.calibration_source_rows == 1056 and args.activation_precision == "int8" and
            args.shape_mode == "fixed" and args.region_image_rows is None and
            args.hidden_a8_channel_groups == 4 and args.blocks != "all"):
        raise ValueError("row-split probe needs one image-row bucket, full FFN width, "
                         "four hidden A8 groups and 1056-row real captures")
    if not 1.0 <= args.activation_scale <= 64.0:
        raise ValueError("activation-scale must be 1...64")
    if not 1.0 <= args.output_scale <= 256.0:
        raise ValueError("output-scale must be 1...256")
    if args.activation_precision == "int8":
        if (args.variant != "int8_pc" or args.calibration_dir is None or
                args.shape_mode != "fixed" or args.sq_hidden_rows < 1 or
                not 0 <= args.sq_alpha1 <= 1 or not 0 <= args.sq_alpha2 <= 1):
            raise ValueError("W8A8 requires INT8 weights, fixed bucket and real SmoothQuant calibration")
        if any(path.is_symlink() or not path.is_dir() for path in
               [args.calibration_dir, *args.extra_calibration_dir]):
            raise ValueError("calibration directories must be real directories")
    elif args.calibration_dir is not None or args.extra_calibration_dir:
        raise ValueError("calibration directories require --activation-precision int8")
    if args.calibration_source_rows is not None and not (
            args.activation_precision == "int8" and
            (args.bucket == 1024 or args.row_split_probe) and
            args.calibration_source_rows == 1056 and args.region_image_rows is None):
        raise ValueError("calibration source row trimming requires an image-only W8A8 export")
    if (args.input_a8_search or args.outlier_rows) and args.activation_precision != "int8":
        raise ValueError("A8 calibration options require INT8 activation calibration")
    if args.region_image_rows is not None and (args.activation_precision != "int8" or
            not args.outlier_rows or not 0 < args.region_image_rows < args.bucket or
            args.region_image_rows % 32):
        raise ValueError("region-image-rows needs outlier-aware W8A8 and a 32-aligned split")
    if args.shared_region_input_qdq and args.region_image_rows is None:
        raise ValueError("shared input Q/DQ requires image/caption regions")
    if args.sq_hidden_image_only and args.region_image_rows is None:
        raise ValueError("image-only hidden SmoothQuant requires region-image-rows")
    if args.hidden_a8_channel_groups is not None and (
            not (args.sq_hidden_image_only or
                 ((args.bucket == 1024 or args.row_split_probe) and args.region_image_rows is None and
                  args.activation_precision == "int8" and args.outlier_rows)) or
            args.hidden_a8_channel_groups < 2 or
            args.ane_mlp_width % args.hidden_a8_channel_groups or
            (args.ane_mlp_width // args.hidden_a8_channel_groups) % 32):
        raise ValueError("hidden A8 channel groups need image-only calibration and 32-aligned groups")
    if args.adaptive_hidden_a8_bins is not None and (
            not args.sq_hidden_image_only or args.adaptive_hidden_a8_bins < 2 or
            args.adaptive_hidden_a8_bins > 8 or args.hidden_a8_channel_groups is not None):
        raise ValueError("adaptive hidden A8 requires image-only SQ, 2...8 bins, no channel groups")
    if args.adaptive_hidden_a8_shared_qdq and args.adaptive_hidden_a8_bins is None:
        raise ValueError("shared Q/DQ requires an adaptive hidden A8 scale bank")
    if args.image_hidden_sq_alpha2 is not None and (
            not args.adaptive_hidden_a8_shared_qdq or
            not args.sq_hidden_image_only or
            not 0 <= args.image_hidden_sq_alpha2 <= 1):
        raise ValueError("image-only hidden SQ requires a shared adaptive A8 graph and alpha2 in [0, 1]")
    if args.adaptive_caption_a8_bins is not None and (
            not args.adaptive_hidden_a8_shared_qdq or
            args.region_image_rows is None or
            not 2 <= args.adaptive_caption_a8_bins <= 8):
        raise ValueError("adaptive caption A8 needs a shared image bank and 2...8 bins")
    if args.channel_routing is not None and (
            args.activation_precision != "int8" or args.region_image_rows is None or
            args.lora or args.blocks != "all"):
        raise ValueError("channel routing requires all W8A8 blocks and no LoRA")
    blocks = list(DEFAULT_BLOCKS)
    block_indexes = parse_blocks(args.blocks, blocks)
    if args.row_split_probe and len(block_indexes) != 1:
        raise ValueError("full-width row-split probe exports exactly one block")
    if args.image_hidden_sq_blocks is not None and args.image_hidden_sq_alpha2 is None:
        raise ValueError("image-only hidden SQ blocks require an alpha2 override")
    if args.adaptive_caption_a8_blocks is not None and args.adaptive_caption_a8_bins is None:
        raise ValueError("adaptive caption A8 blocks require a scale bank")
    image_hidden_sq_blocks = (parse_blocks(args.image_hidden_sq_blocks, blocks)
                              if args.image_hidden_sq_blocks is not None else
                              block_indexes if args.image_hidden_sq_alpha2 is not None else [])
    if any(block not in block_indexes for block in image_hidden_sq_blocks):
        raise ValueError("image-only hidden SQ blocks must be exported")
    adaptive_caption_a8_blocks = (parse_blocks(args.adaptive_caption_a8_blocks, blocks)
                                  if args.adaptive_caption_a8_blocks is not None else
                                  [block for block in block_indexes if block >= 2]
                                  if args.adaptive_caption_a8_bins is not None else [])
    if any(block not in block_indexes for block in adaptive_caption_a8_blocks):
        raise ValueError("adaptive caption A8 blocks must be exported")
    if any(block < 2 for block in adaptive_caption_a8_blocks):
        raise ValueError("noise-refiner blocks have no real caption rows for adaptive A8")

    import numpy as np
    import coremltools as ct
    import coremltools.optimize.coreml as optimize
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types, get_new_symbol
    from z_image_smoothquant import verify_w8a8

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    checkpoint_source = source_from_model(args.model, args.convrot_mode)
    if (args.activation_precision == "int8" and
            getattr(checkpoint_source, "has_convrot", False) and args.convrot_mode == "native"):
        raise ValueError("SmoothQuant on ConvRot weights requires --convrot-mode derotate")
    checkpoint = checkpoint_source.checkpoint
    if args.lora_strength and len(args.lora_strength) != len(args.lora):
        raise ValueError("--lora-strength must be repeated once per --lora")
    lora_strengths = [float(value) for value in args.lora_strength] or [1.0] * len(args.lora)
    lora_roles = [args.lora_role] * len(args.lora)
    lora_records = lora_provenance(args.lora, lora_strengths, lora_roles)
    lora_bundles = load_loras(args.lora, lora_strengths, lora_roles, np)
    calibration = {}
    calibration_dirs = [args.calibration_dir, *args.extra_calibration_dir] if args.calibration_dir else []
    if args.activation_precision == "int8":
        for ordinal in block_indexes:
            samples, digest = load_calibration_union(
                [path / f"block{ordinal}" for path in calibration_dirs],
                args.bucket, HIDDEN, np, pad_rows=ordinal < 2,
                source_rows=args.calibration_source_rows)
            calibration[str(ordinal)] = {"sha256": digest, "samples": len(samples)}
    channel_routing = None
    if args.channel_routing is not None:
        if args.channel_routing.is_symlink() or not args.channel_routing.is_file():
            raise ValueError("channel routing must be a regular JSON file")
        route = json.loads(args.channel_routing.read_text())
        group = route.get("group_width")
        groups = route.get("ane_group_indexes")
        if (route.get("owner") != "turbocider.z_image.w8a8.channel_routing.v1" or
                not isinstance(group, int) or group < 32 or group % 32 or
                MLP_WIDTH % group or args.ane_mlp_width % group or
                route.get("mlp_width") != MLP_WIDTH or
                route.get("ane_mlp_width") != args.ane_mlp_width or
                not isinstance(groups, dict) or set(groups) != set(calibration) or
                route.get("calibration_sha256") != {block: item["sha256"]
                                                     for block, item in calibration.items()}):
            raise ValueError("channel routing geometry or calibration differs from export")
        for key, selected in groups.items():
            if (not isinstance(selected, list) or
                    len(selected) != args.ane_mlp_width // group or
                    any(not isinstance(index, int) or isinstance(index, bool) or
                        index < 0 or index >= MLP_WIDTH // group for index in selected) or
                    len(set(selected)) != len(selected) or selected != sorted(selected)):
                raise ValueError(f"invalid ANE channel group selection for block {key}")
        channel_routing = {"group_width": group, "ane_group_indexes": groups}
    output = args.output.absolute()
    if output.is_symlink():
        raise ValueError("symlink output unsupported")
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".export.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        provenance = checkpoint_source.provenance()
        if channel_routing is not None and route["checkpoint_sha256"] != provenance["checkpoint_sha256"]:
            raise ValueError("channel routing checkpoint differs from export")
        identity = {
            "owner": ("turbocider.z_image.coreml.v1" if args.model_kind == "z-image"
                      else "turbocider.llada_image.coreml.v1"),
            **provenance,
            "bucket": args.bucket,
            "hidden": HIDDEN,
            "mlp_width": MLP_WIDTH,
            "ane_mlp_start": 0,
            "ane_mlp_end": args.ane_mlp_width,
            **({"row_split_probe": True} if args.row_split_probe else {}),
            "activation_scale": args.activation_scale,
            "output_scale": args.output_scale,
            "blocks": block_indexes,
            "variant": args.variant,
            "activation_precision": args.activation_precision,
            **({"calibration": calibration, "sq_alpha1": args.sq_alpha1,
                "sq_alpha2": args.sq_alpha2, "sq_hidden_rows": args.sq_hidden_rows,
                "a8_graph": ("explicit_dual_qdq_shared_region_input_v13"
                             if args.shared_region_input_qdq else
                             "explicit_dual_qdq_image_only_channel_groups_v12"
                             if (args.bucket == 1024 or args.row_split_probe) and
                                args.region_image_rows is None and
                                args.hidden_a8_channel_groups is not None else
                             (("explicit_dual_qdq_regions_image_sq_adaptive_shared_vector_v11"
                              if args.adaptive_hidden_a8_shared_qdq else
                              "explicit_dual_qdq_regions_image_sq_adaptive_bins_v10"
                              if args.adaptive_hidden_a8_bins is not None else
                              "explicit_dual_qdq_regions_image_sq_channel_groups_v9"
                              if args.hidden_a8_channel_groups is not None else
                              "explicit_dual_qdq_regions_image_sq_v8" if args.sq_hidden_image_only
                              else "explicit_dual_qdq_regions_input_search_v7" if args.input_a8_search
                              else "explicit_dual_qdq_regions_v6")
                             if args.region_image_rows is not None else
                             "explicit_dual_qdq_output_nmse_outlier_rows_v5_" +
                             ("input_search" if args.input_a8_search else "max_input")
                             if args.outlier_rows else
                             ("explicit_dual_qdq_output_nmse_input_search_v4"
                              if args.input_a8_search else "explicit_dual_qdq_output_nmse_v3"))),
                **({"region_image_rows": args.region_image_rows}
                   if args.region_image_rows is not None else {}),
                **({"shared_region_input_qdq": True}
                   if args.shared_region_input_qdq else {}),
                **({"hidden_a8_channel_groups": args.hidden_a8_channel_groups}
                   if args.hidden_a8_channel_groups is not None else {}),
                **({"adaptive_hidden_a8_bins": args.adaptive_hidden_a8_bins}
                   if args.adaptive_hidden_a8_bins is not None else {}),
                **({"adaptive_hidden_a8_shared_qdq": True}
                   if args.adaptive_hidden_a8_shared_qdq else {}),
                **({"image_hidden_sq_alpha2": args.image_hidden_sq_alpha2}
                   if args.image_hidden_sq_alpha2 is not None else {}),
                **({"image_hidden_sq_blocks": image_hidden_sq_blocks}
                   if args.image_hidden_sq_blocks is not None else {}),
                **({"adaptive_caption_a8_bins": args.adaptive_caption_a8_bins}
                   if args.adaptive_caption_a8_bins is not None else {}),
                **({"adaptive_caption_a8_blocks": adaptive_caption_a8_blocks}
                   if args.adaptive_caption_a8_blocks is not None else {}),
                **({"calibration_directory_count": len(calibration_dirs)}
                   if len(calibration_dirs) > 1 else {}),
                **({"calibration_source_rows": args.calibration_source_rows}
                   if args.calibration_source_rows is not None else {}),
                **({"image_only_token_rows": 1024}
                   if not args.row_split_probe and args.bucket == 1024 and
                      args.region_image_rows is None and
                      args.hidden_a8_channel_groups is not None else {}),
                **({"channel_routing": channel_routing} if channel_routing is not None else {})}
               if calibration else {}),
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
        if args.shape_mode != "fixed":
            identity["input_shapes"] = {"mode": args.shape_mode, "buckets": buckets,
                                        "default": args.min_bucket}
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
            activation_receipts = {}
            convrot_groups = set()
            for ordinal in block_indexes:
                prefix = blocks[ordinal]
                name = f"block{ordinal}_mlp_branch.int8_pc.mlpackage"
                destination = output / name
                receipt = output / f"block{ordinal}.json"
                activation_receipt = output / f"block{ordinal}.activation.json"
                if destination.is_symlink() or receipt.is_symlink():
                    raise ValueError("symlink artifact unsupported")
                if destination.exists() and receipt.exists():
                    saved = json.loads(receipt.read_text())
                    valid = all((destination / relative).is_file() and sha(destination / relative) == digest
                                for relative, digest in saved.items())
                    if not valid:
                        raise ValueError(f"existing export corrupted: block {ordinal}")
                    checksums[str(ordinal)] = saved
                    if calibration:
                        if activation_receipt.is_symlink() or not activation_receipt.is_file():
                            raise ValueError(f"missing activation receipt for block {ordinal}")
                        activation_receipts[str(ordinal)] = json.loads(activation_receipt.read_text())
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
                    if channel_routing is not None:
                        indices, _ = route_channel_indices(
                            channel_routing["ane_group_indexes"][str(ordinal)],
                            channel_routing["group_width"], MLP_WIDTH,
                            args.ane_mlp_width, np)
                        w1, w3, w2 = w1[indices], w3[indices], w2[:, indices]
                    width = args.ane_mlp_width
                    inverse_input_scale = None
                    input_a8_scale = None
                    hidden_a8_scale = None
                    region_input_scales = None
                    region_hidden_scales = None
                    region_samples = None
                    image_hidden_channel_scales = None
                    adaptive_hidden_scales = None
                    adaptive_caption_scales = None
                    image_resmooth_ratio = None
                    image_resmooth_inverse = None
                    if calibration:
                        samples, digest = load_calibration_union(
                            [path / f"block{ordinal}" for path in calibration_dirs],
                            args.bucket, HIDDEN, np, pad_rows=ordinal < 2,
                            source_rows=args.calibration_source_rows)
                        if digest != calibration[str(ordinal)]["sha256"]:
                            raise ValueError("calibration changed during export")
                        gate, up, down, s1, s2 = smooth_partial_ffn(
                            w1[:width], w3[:width], w2[:, :width], samples,
                            args.sq_alpha1, args.sq_alpha2, np, args.sq_hidden_rows,
                            outlier_rows=args.outlier_rows,
                            hidden_region_rows=(args.region_image_rows if args.sq_hidden_image_only
                                                else None))
                        if ordinal in image_hidden_sq_blocks:
                            _, _, _, _, image_s2 = smooth_partial_ffn(
                                w1[:width], w3[:width], w2[:, :width], samples,
                                args.sq_alpha1, args.image_hidden_sq_alpha2, np,
                                args.sq_hidden_rows, outlier_rows=args.outlier_rows,
                                hidden_region_rows=args.region_image_rows)
                            ratio = s2 / image_s2
                            image_resmooth_ratio = np.ascontiguousarray(
                                ratio.astype(np.float16).reshape(1, width, 1, 1))
                            image_resmooth_inverse = np.ascontiguousarray(
                                (1 / ratio).astype(np.float16).reshape(1, width, 1, 1))
                        inverse_input_scale = np.ascontiguousarray(
                            (1 / s1).astype(np.float16).reshape(1, HIDDEN, 1, 1))
                        if args.input_a8_search:
                            input_a8_scale, input_trials = calibrate_input_a8(
                                gate, up, down, s1, samples, np,
                                outlier_rows=args.outlier_rows)
                        else:
                            maximum = max(float(np.max(np.abs(sample.astype(np.float32) / s1)))
                                          for sample in samples)
                            input_a8_scale = np.float16(max(maximum / 127, 1e-6))
                        if args.region_image_rows is not None:
                            regions = [(sample[:args.region_image_rows] for sample in samples),
                                       (sample[args.region_image_rows:] for sample in samples)]
                            region_samples = [list(group) for group in regions]
                            region_input_scales = [np.float16(max(
                                max(float(np.max(np.abs(sample.astype(np.float32) / s1)))
                                    for sample in group) / 127, 1e-6))
                                for group in region_samples]
                        first = np.ascontiguousarray(
                            np.concatenate((gate, up), axis=0).astype(np.float16)[:, :, None, None])
                    else:
                        first = np.ascontiguousarray(
                            np.concatenate((w1[:width], w3[:width]), axis=0)[:, :, None, None])
                    # Z-Image's BF16 gated activations can exceed FP16 range at
                    # the elementwise product even though the final projection
                    # is finite.  Scale both multiplicands before the product
                    # and compensate exactly in w2: (a/s)*(b/s)*(w2*s^2).
                    # Per-channel symmetric quantization is invariant to this
                    # uniform row scale apart from normal FP16 rounding.
                    scale = np.float32(args.activation_scale)
                    if calibration:
                        hidden_a8_scale, trials = calibrate_hidden_a8(
                            gate, up, down, s1, samples, float(input_a8_scale),
                            float(scale), np, outlier_rows=args.outlier_rows)
                        if region_input_scales is not None:
                            region_hidden_scales = [calibrate_hidden_a8(
                                gate, up, down, s1, group, float(region_input_scales[index]),
                                float(scale), np, outlier_rows=args.outlier_rows)[0]
                                for index, group in enumerate(region_samples)]
                        if args.hidden_a8_channel_groups is not None:
                            image_hidden_channel_scales = calibrate_hidden_channel_groups(
                                gate, up, down, s1,
                                region_samples[0] if region_samples is not None else samples,
                                float(region_input_scales[0] if region_input_scales is not None
                                      else input_a8_scale), float(scale),
                                args.hidden_a8_channel_groups, np)
                        if args.adaptive_hidden_a8_bins is not None:
                            adaptive_hidden_scales = calibrate_adaptive_hidden_a8(
                                gate, (up * image_resmooth_ratio.reshape(-1, 1)
                                       if image_resmooth_ratio is not None else up),
                                s1, region_samples[0],
                                float(region_input_scales[0]), float(scale),
                                args.region_image_rows, args.adaptive_hidden_a8_bins, np)
                        if ordinal in adaptive_caption_a8_blocks:
                            adaptive_caption_scales = calibrate_adaptive_hidden_a8(
                                gate, up, s1, region_samples[1],
                                float(region_input_scales[1]), float(scale),
                                args.bucket - args.region_image_rows,
                                args.adaptive_caption_a8_bins, np)
                        activation_receipts[str(ordinal)] = {
                            "input_a8_scale": float(input_a8_scale),
                            "hidden_a8_scale": float(hidden_a8_scale),
                            "threshold_search": trials,
                            **({"input_threshold_search": input_trials}
                               if args.input_a8_search else {}),
                            **({"region_image_rows": args.region_image_rows,
                                "region_input_a8_scales": list(map(float, region_input_scales)),
                                "region_hidden_a8_scales": list(map(float, region_hidden_scales))}
                               if region_input_scales is not None else {}),
                            **({"image_hidden_channel_a8_scales": list(map(
                                float, image_hidden_channel_scales))}
                               if image_hidden_channel_scales is not None else {}),
                            **({"adaptive_hidden_a8_scales": list(map(
                                float, adaptive_hidden_scales))}
                               if adaptive_hidden_scales is not None else {}),
                            **({"image_hidden_sq_alpha2": args.image_hidden_sq_alpha2}
                               if image_resmooth_ratio is not None else {}),
                            **({"adaptive_caption_a8_scales": list(map(
                                float, adaptive_caption_scales))}
                               if adaptive_caption_scales is not None else {}),
                        }
                    last = np.ascontiguousarray(
                        ((down if calibration else w2[:, :width].astype(np.float32)) *
                         (scale * scale / np.float32(args.output_scale))).astype(np.float16)[:, :, None, None]
                    )
                    del w1, w2, w3
                    rotation_hidden = (convrot_activation_weight(HIDDEN, native_group, np)
                                       if native_group is not None else None)
                    rotation_width = (convrot_activation_weight(width, native_group, np)
                                      if native_group is not None else None)

                    convert_inputs = {}
                    rows = args.bucket
                    if args.shape_mode != "fixed":
                        rows = get_new_symbol()
                        shape = (ct.EnumeratedShapes(
                            shapes=[(1, HIDDEN, 1, n) for n in buckets],
                            default=(1, HIDDEN, 1, args.min_bucket))
                            if args.shape_mode == "enumerated" else ct.Shape(
                                (1, HIDDEN, 1, ct.RangeDim(args.min_bucket, args.bucket,
                                                         default=args.min_bucket))))
                        convert_inputs["inputs"] = [ct.TensorType(name="x", shape=shape, dtype=np.float16)]
                    @mb.program(
                        input_specs=[mb.TensorSpec(shape=(1, HIDDEN, 1, rows), dtype=types.fp16)],
                        opset_version=ct.target.macOS15,
                    )
                    def branch(x):
                        def quantize_regions(value, scales, label, channels,
                                             image_channel_scales=None):
                            pieces = []
                            for index, (begin, end) in enumerate(
                                    ((0, args.region_image_rows), (args.region_image_rows, args.bucket))):
                                region = mb.slice_by_index(
                                    x=value, begin=[0, 0, 0, begin], end=[1, channels, 1, end],
                                    name=f"{label}_slice_{index}")
                                if index == 0 and image_channel_scales is not None:
                                    channel_pieces = []
                                    step = channels // len(image_channel_scales)
                                    for group, channel_scale in enumerate(image_channel_scales):
                                        channel = mb.slice_by_index(
                                            x=region, begin=[0, group * step, 0, 0],
                                            end=[1, (group + 1) * step, 1, end - begin],
                                            name=f"{label}_channel_{group}")
                                        integer = mb.quantize(
                                            input=channel, scale=channel_scale,
                                            output_dtype="int8", name=f"{label}_q_0_{group}")
                                        channel_pieces.append(mb.dequantize(
                                            input=integer, scale=channel_scale,
                                            name=f"{label}_dq_0_{group}"))
                                    pieces.append(mb.concat(
                                        values=channel_pieces, axis=1,
                                        name=f"{label}_image_channels"))
                                    continue
                                integer = mb.quantize(input=region, scale=scales[index],
                                                      output_dtype="int8", name=f"{label}_q_{index}")
                                pieces.append(mb.dequantize(input=integer, scale=scales[index],
                                                            name=f"{label}_dq_{index}"))
                            return mb.concat(values=pieces, axis=3, name=f"{label}_concat")

                        projected_input = x
                        if inverse_input_scale is not None:
                            projected_input = mb.mul(x=projected_input, y=inverse_input_scale,
                                                     name="smoothquant_input")
                            if region_input_scales is not None:
                                if args.shared_region_input_qdq:
                                    image_scale, caption_scale = map(float, region_input_scales)
                                    ratio = np.ones((1, 1, 1, args.bucket), dtype=np.float16)
                                    ratio[..., :args.region_image_rows] = caption_scale / image_scale
                                    inverse = np.ones_like(ratio)
                                    inverse[..., :args.region_image_rows] = image_scale / caption_scale
                                    prepared = mb.mul(x=projected_input, y=ratio,
                                                      name="a8_input_region_prepare")
                                    quantized = mb.quantize(
                                        input=prepared, scale=region_input_scales[1],
                                        output_dtype="int8", name="a8_input_region_q_shared")
                                    reconstructed = mb.dequantize(
                                        input=quantized, scale=region_input_scales[1],
                                        name="a8_input_region_dq_shared")
                                    projected_input = mb.mul(
                                        x=reconstructed, y=inverse,
                                        name="a8_input_region_restore")
                                else:
                                    projected_input = quantize_regions(
                                        projected_input, region_input_scales, "a8_input", HIDDEN)
                            else:
                                a8 = mb.quantize(input=projected_input, scale=input_a8_scale,
                                                 output_dtype="int8", name="a8_input")
                                projected_input = mb.dequantize(input=a8, scale=input_a8_scale,
                                                                name="a8_input_dequantized")
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
                        if hidden_a8_scale is not None:
                            if adaptive_hidden_scales is not None:
                                image = mb.slice_by_index(
                                    x=last_input, begin=[0, 0, 0, 0],
                                    end=[1, width, 1, args.region_image_rows],
                                    name="a8_hidden_adaptive_image")
                                if image_resmooth_ratio is not None:
                                    image = mb.mul(x=image, y=image_resmooth_ratio,
                                                   name="a8_hidden_image_resmooth")
                                peak = mb.reduce_max(x=mb.abs(x=image), axes=[1],
                                                     keep_dims=True, name="a8_hidden_peak")
                                chosen = None
                                scale_ratio = None
                                inverse_ratio = None
                                base_scale = np.float16(adaptive_hidden_scales[-1])
                                for index in reversed(range(len(adaptive_hidden_scales))):
                                    value = np.float16(adaptive_hidden_scales[index])
                                    if args.adaptive_hidden_a8_shared_qdq:
                                        # Core ML folds select(dynamic per-token
                                        # predicate, scalar, scalar) to a *single*
                                        # scalar in this shape. Explicit constant
                                        # vectors preserve the token axis.
                                        candidate_ratio = np.full(
                                            (1, 1, 1, args.region_image_rows),
                                            float(base_scale) / float(value),
                                            dtype=np.float16)
                                        candidate_inverse = np.full(
                                            (1, 1, 1, args.region_image_rows),
                                            float(value) / float(base_scale),
                                            dtype=np.float16)
                                        if scale_ratio is None:
                                            scale_ratio = candidate_ratio
                                            inverse_ratio = candidate_inverse
                                    else:
                                        q = mb.quantize(input=image, scale=value,
                                                        output_dtype="int8",
                                                        name=f"a8_hidden_adaptive_q_{index}")
                                        candidate = mb.dequantize(input=q, scale=value,
                                                                 name=f"a8_hidden_adaptive_dq_{index}")
                                        if chosen is None:
                                            chosen = candidate
                                    if index != len(adaptive_hidden_scales) - 1:
                                        selected = mb.less_equal(
                                            x=peak, y=np.float16(float(value) * 127.),
                                            name=f"a8_hidden_adaptive_le_{index}")
                                        if args.adaptive_hidden_a8_shared_qdq:
                                            scale_ratio = mb.select(
                                                cond=selected, a=candidate_ratio, b=scale_ratio,
                                                name=f"a8_hidden_ratio_select_{index}")
                                            inverse_ratio = mb.select(
                                                cond=selected, a=candidate_inverse,
                                                b=inverse_ratio,
                                                name=f"a8_hidden_inverse_select_{index}")
                                        else:
                                            chosen = mb.select(
                                                cond=selected, a=candidate, b=chosen,
                                                name=f"a8_hidden_adaptive_select_{index}")
                                if args.adaptive_hidden_a8_shared_qdq:
                                    prepared = mb.mul(x=image, y=scale_ratio,
                                                      name="a8_hidden_adaptive_prepare")
                                    q = mb.quantize(input=prepared, scale=base_scale,
                                                    output_dtype="int8",
                                                    name="a8_hidden_adaptive_q_shared")
                                    dq = mb.dequantize(input=q, scale=base_scale,
                                                       name="a8_hidden_adaptive_dq_shared")
                                    chosen = mb.mul(x=dq, y=inverse_ratio,
                                                    name="a8_hidden_adaptive_restore")
                                if image_resmooth_inverse is not None:
                                    chosen = mb.mul(x=chosen, y=image_resmooth_inverse,
                                                    name="a8_hidden_image_unresmooth")
                                caption = mb.slice_by_index(
                                    x=last_input, begin=[0, 0, 0, args.region_image_rows],
                                    end=[1, width, 1, args.bucket],
                                    name="a8_hidden_adaptive_caption")
                                if adaptive_caption_scales is not None:
                                    caption_rows = args.bucket - args.region_image_rows
                                    caption_peak = mb.reduce_max(
                                        x=mb.abs(x=caption), axes=[1], keep_dims=True,
                                        name="a8_hidden_caption_peak")
                                    base = np.float16(adaptive_caption_scales[-1])
                                    caption_ratio = caption_inverse = None
                                    for index in reversed(range(len(adaptive_caption_scales))):
                                        value = np.float16(adaptive_caption_scales[index])
                                        candidate_ratio = np.full(
                                            (1, 1, 1, caption_rows), float(base) / float(value),
                                            dtype=np.float16)
                                        candidate_inverse = np.full(
                                            (1, 1, 1, caption_rows), float(value) / float(base),
                                            dtype=np.float16)
                                        if caption_ratio is None:
                                            caption_ratio = candidate_ratio
                                            caption_inverse = candidate_inverse
                                        else:
                                            selected = mb.less_equal(
                                                x=caption_peak,
                                                y=np.float16(float(value) * 127.),
                                                name=f"a8_hidden_caption_le_{index}")
                                            caption_ratio = mb.select(
                                                cond=selected, a=candidate_ratio,
                                                b=caption_ratio,
                                                name=f"a8_hidden_caption_ratio_{index}")
                                            caption_inverse = mb.select(
                                                cond=selected, a=candidate_inverse,
                                                b=caption_inverse,
                                                name=f"a8_hidden_caption_inverse_{index}")
                                    caption_prepared = mb.mul(
                                        x=caption, y=caption_ratio,
                                        name="a8_hidden_caption_prepare")
                                    caption_q = mb.quantize(
                                        input=caption_prepared, scale=base,
                                        output_dtype="int8", name="a8_hidden_caption_q_shared")
                                    caption_dq = mb.dequantize(
                                        input=caption_q, scale=base,
                                        name="a8_hidden_caption_dq_shared")
                                    caption = mb.mul(
                                        x=caption_dq, y=caption_inverse,
                                        name="a8_hidden_caption_restore")
                                else:
                                    value = region_hidden_scales[1]
                                    q = mb.quantize(input=caption, scale=value,
                                                    output_dtype="int8", name="a8_hidden_q_1")
                                    caption = mb.dequantize(input=q, scale=value,
                                                            name="a8_hidden_dq_1")
                                last_input = mb.concat(values=[chosen, caption], axis=3,
                                                       name="a8_hidden_adaptive_concat")
                            elif region_hidden_scales is not None:
                                last_input = quantize_regions(
                                    last_input, region_hidden_scales, "a8_hidden", width,
                                    image_hidden_channel_scales)
                            elif image_hidden_channel_scales is not None:
                                step = width // len(image_hidden_channel_scales)
                                pieces = []
                                for group, channel_scale in enumerate(image_hidden_channel_scales):
                                    channel = mb.slice_by_index(
                                        x=last_input, begin=[0, group * step, 0, 0],
                                        end=[1, (group + 1) * step, 1, args.bucket],
                                        name=f"a8_hidden_channel_{group}")
                                    integer = mb.quantize(input=channel, scale=channel_scale,
                                                          output_dtype="int8",
                                                          name=f"a8_hidden_q_0_{group}")
                                    pieces.append(mb.dequantize(input=integer, scale=channel_scale,
                                                                name=f"a8_hidden_dq_0_{group}"))
                                last_input = mb.concat(values=pieces, axis=1,
                                                       name="a8_hidden_image_channels")
                            else:
                                a8_hidden = mb.quantize(input=last_input, scale=hidden_a8_scale,
                                                        output_dtype="int8", name="a8_hidden")
                                last_input = mb.dequantize(input=a8_hidden, scale=hidden_a8_scale,
                                                           name="a8_hidden_dequantized")
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
                        **convert_inputs,
                    )
                    if calibration:
                        del samples
                    if args.variant == "int8_pc":
                        w8 = optimize.OpLinearQuantizerConfig(
                            mode="linear_symmetric", dtype="int8", granularity="per_channel",
                            block_size=32, weight_threshold=0)
                        quantizer = (optimize.OptimizationConfig(
                            global_config=None, op_name_configs={name: w8 for name in ("projected", "y")})
                            if calibration else optimize.OptimizationConfig(global_config=w8))
                        compressed = optimize.linear_quantize_weights(model, quantizer)
                    else:
                        compressed = model
                    if calibration:
                        verify_w8a8(compressed.get_spec(), ("projected", "y"),
                                    allow_adaptive_scale=args.adaptive_hidden_a8_shared_qdq,
                                    allow_input_region_scale=args.shared_region_input_qdq)
                    with tempfile.TemporaryDirectory(prefix=".export-", dir=output) as temporary:
                        package = Path(temporary) / name
                        compressed.save(package)
                        os.replace(package, destination)
                    saved = {str(path.relative_to(destination)): sha(path)
                             for path in sorted(destination.rglob("*")) if path.is_file()}
                    atom(receipt, saved)
                    if calibration:
                        atom(activation_receipt, activation_receipts[str(ordinal)])
                    checksums[str(ordinal)] = saved
                    del first, last, rotation_hidden, rotation_width, branch, model, compressed
                    gc.collect()
                artifacts[str(ordinal)] = {"int8_pc": name}
                atom(output / "progress.json", {"completed": len(artifacts), "total": len(block_indexes)})
                print(f"partition {len(artifacts)}/{len(block_indexes)} ready", flush=True)

        checkpoint_source.validate_unchanged(provenance)
        if lora_records != lora_provenance(args.lora, lora_strengths, lora_roles):
            raise ValueError("LoRA changed during export")
        for ordinal in block_indexes:
            if calibration and load_calibration_union(
                    [path / f"block{ordinal}" for path in calibration_dirs],
                    args.bucket, HIDDEN, np, pad_rows=ordinal < 2,
                    source_rows=args.calibration_source_rows)[1] != \
                    calibration[str(ordinal)]["sha256"]:
                raise ValueError("calibration changed during export")
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
                "buckets": buckets,
                **({"input_mode": args.shape_mode, "default_bucket": args.min_bucket}
                   if args.shape_mode != "fixed" else {}),
            },
            "functions": {str(bucket): "main" for bucket in buckets},
            "artifacts": artifacts,
            "artifact_sha256": checksums,
            **({"activation_quantization": activation_receipts} if calibration else {}),
            "export_identity": identity,
        })
        print(json.dumps({"source_manifest": str(output / "manifest.json"),
                          "checkpoint": str(checkpoint), "blocks": block_indexes}), flush=True)


if __name__ == "__main__":
    main()
