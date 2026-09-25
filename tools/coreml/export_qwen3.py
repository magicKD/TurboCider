"""Offline gated-MLP Core ML exporter for Qwen3 encoders and Qwen21 DiT.

The exporter accepts a Qwen3/Qwen3-VL text-encoder directory, a single
safetensors file, or an index-bound shard set.  BF16/F16 weights are copied
as FP16 and TurboCider's native Q4/Q8 affine tensors are dequantized one
projection at a time through MLX; model weights are never merged or written
to the artifact directory.  The resulting source manifest is compiled by the
normal ``coreml_resources`` cache workflow before native execution.  Use
``--tensor-prefix model.language_model`` for H3's Qwen3-VL checkpoint.
Use ``--tensor-layout qwen21 --hidden 4096 --mlp-width 12288 --layer-count 32``
with the Comfy DiT checkpoint for experimental Qwen Image 2.1 branches.
"""

import argparse
import fcntl
import gc
import hashlib
import json
import mmap
import os
import re
import shutil
import signal
import struct
import tempfile
import math
from pathlib import Path


HIDDEN = 2560
MLP_WIDTH = 9728
LAYERS = 36


def mlp_weights(source, layer, layout, prefix, hidden, width, np, mx):
    if layout == "qwen21":
        root = f"transformer_blocks.{layer}.img_mlp"
        fused = source.tensor(root + ".gate_up.weight", (width * 2, hidden), np, mx)
        gate, up = np.split(fused, 2, axis=0)
        down = source.tensor(root + ".out.weight", (hidden, width), np, mx)
        return gate, up, down
    root = f"{prefix}.layers.{layer}.mlp"
    return (source.tensor(root + ".gate_proj.weight", (width, hidden), np, mx),
            source.tensor(root + ".up_proj.weight", (width, hidden), np, mx),
            source.tensor(root + ".down_proj.weight", (hidden, width), np, mx))


def atom(path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_header(stream):
    raw = stream.read(8)
    if len(raw) != 8:
        raise ValueError("invalid safetensors header length")
    header_size = struct.unpack("<Q", raw)[0]
    if header_size > 100_000_000:
        raise ValueError("safetensors header is unreasonably large")
    raw_header = stream.read(header_size)
    if len(raw_header) != header_size:
        raise ValueError("safetensors header exceeds file")
    header = json.loads(raw_header)
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    return header_size, header


def shard_name(value):
    if (not isinstance(value, str) or not value or value in (".", "..") or
            "/" in value or "\\" in value or Path(value).name != value or
            not value.endswith(".safetensors")):
        raise ValueError(f"invalid Qwen3 shard name: {value!r}")
    return value


def artifact_receipt(destination, receipt):
    if (destination.is_symlink() or not destination.is_dir() or
            receipt.is_symlink() or not receipt.is_file()):
        raise ValueError(f"invalid existing Qwen3 artifact: {destination.name}")
    if any(path.is_symlink() for path in destination.rglob("*")):
        raise ValueError(f"existing Qwen3 artifact contains a symlink: {destination.name}")
    saved = json.loads(receipt.read_text())
    if not isinstance(saved, dict) or not saved:
        raise ValueError(f"invalid Qwen3 artifact receipt: {receipt.name}")
    for relative, digest in saved.items():
        path = Path(relative) if isinstance(relative, str) else Path()
        if (not isinstance(relative, str) or not relative or path.is_absolute() or
                ".." in path.parts or "\\" in relative or not isinstance(digest, str) or
                len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError(f"unsafe Qwen3 artifact receipt: {receipt.name}")
        candidate = destination / path
        if candidate.is_symlink() or not candidate.is_file() or sha(candidate) != digest:
            raise ValueError(f"existing Qwen3 artifact is corrupted: {destination.name}")
    return saved


class QwenSource:
    def __init__(self, model):
        model = Path(model)
        if model.is_symlink():
            raise ValueError("Qwen3 --model path must not be a symlink")
        if model.is_file():
            if model.suffix != ".safetensors":
                raise ValueError("--model file must be a safetensors file")
            self.checkpoint = model.resolve()
            self.weight_map = None
        elif model.is_dir():
            index = model / "model.safetensors.index.json"
            single = model / "model.safetensors"
            if index.is_file():
                if index.is_symlink():
                    raise ValueError("Qwen3 safetensors index must not be a symlink")
                self.checkpoint = index.resolve()
                descriptor = json.loads(index.read_text())
                if not isinstance(descriptor, dict):
                    raise ValueError("Qwen3 safetensors index must be an object")
                self.weight_map = descriptor.get("weight_map")
                if not isinstance(self.weight_map, dict) or not self.weight_map:
                    raise ValueError("Qwen3 safetensors index has no weight_map")
                for tensor, shard in self.weight_map.items():
                    if not isinstance(tensor, str) or not tensor:
                        raise ValueError("Qwen3 safetensors index has an invalid tensor name")
                    shard_name(shard)
                mapped_paths = [model / shard_name(shard)
                                for shard in set(self.weight_map.values())]
                if (not all(path.is_file() and not path.is_symlink()
                            for path in mapped_paths)):
                    # Some published snapshots retain a stale multi-shard
                    # index next to a complete single-file checkpoint. Only
                    # fall back when that file's safetensors header contains
                    # exactly the indexed tensor set; otherwise fail closed.
                    if not single.is_file() or single.is_symlink():
                        raise ValueError("Qwen3 shard set contains a missing or symlinked file")
                    with single.open("rb") as stream:
                        _, header = load_header(stream)
                    header_tensors = set(header) - {"__metadata__"}
                    if header_tensors != set(self.weight_map):
                        raise ValueError("Qwen3 shard set contains a missing or symlinked file")
                    self.checkpoint = single.resolve()
                    self.weight_map = None
            elif single.is_file():
                if single.is_symlink():
                    raise ValueError("Qwen3 safetensors checkpoint must not be a symlink")
                self.checkpoint = single.resolve()
                self.weight_map = None
            else:
                raise ValueError("Qwen3 safetensors/index missing under --model")
        else:
            raise ValueError("Qwen3 --model path does not exist")
        unresolved = ({self.checkpoint} if self.weight_map is None else {
            self.checkpoint.parent / shard_name(name)
            for name in set(self.weight_map.values())
        })
        if any(not path.is_file() or path.is_symlink() for path in unresolved):
            raise ValueError("Qwen3 shard set contains a missing or symlinked file")
        self.shards = {path.resolve() for path in unresolved}
        self.streams = {}
        self.memories = {}
        self.headers = {}
        self.offsets = {}

    def __enter__(self):
        try:
            for path in sorted(self.shards):
                stream = path.open("rb")
                header_size, header = load_header(stream)
                memory = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
                if 8 + header_size > len(memory):
                    memory.close()
                    stream.close()
                    raise ValueError("safetensors header exceeds file")
                self.streams[path] = stream
                self.memories[path] = memory
                self.headers[path] = header
                self.offsets[path] = 8 + header_size
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_):
        for memory in self.memories.values():
            memory.close()
        for stream in self.streams.values():
            stream.close()
        self.streams.clear()
        self.memories.clear()
        self.headers.clear()
        self.offsets.clear()

    def path_for(self, name):
        if self.weight_map is None:
            return self.checkpoint
        shard = self.weight_map.get(name)
        try:
            shard = shard_name(shard)
        except ValueError:
            raise ValueError(f"invalid Qwen3 shard mapping for {name}")
        return (self.checkpoint.parent / shard).resolve()

    def raw(self, name, expected_shape=None):
        path = self.path_for(name)
        meta = self.headers.get(path, {}).get(name)
        if not isinstance(meta, dict):
            raise ValueError(f"Qwen3 tensor missing: {name}")
        raw_shape = meta.get("shape")
        if (not isinstance(raw_shape, list) or not raw_shape or
                any(type(value) is not int or value < 0 or value > 65536
                    for value in raw_shape)):
            raise ValueError(f"invalid Qwen3 tensor shape for {name}")
        shape = tuple(raw_shape)
        if expected_shape is not None and shape != tuple(expected_shape):
            raise ValueError(f"unexpected Qwen3 tensor shape for {name}: {shape}")
        dtype = meta.get("dtype")
        itemsize = {"F32": 4, "F16": 2, "BF16": 2, "U32": 4}.get(dtype)
        if itemsize is None:
            raise ValueError(f"unsupported Qwen3 tensor dtype {dtype}: {name}")
        offsets = meta.get("data_offsets")
        if (not isinstance(offsets, list) or len(offsets) != 2 or
                any(type(value) is not int for value in offsets)):
            raise ValueError(f"invalid Qwen3 tensor offsets: {name}")
        start, end = offsets
        count = 1
        for value in shape:
            count *= value
        memory = self.memories[path]
        offset = self.offsets[path]
        if start < 0 or end - start != count * itemsize or offset + end > len(memory):
            raise ValueError(f"invalid Qwen3 tensor offsets: {name}")
        if dtype == "BF16":
            import numpy as np
            words = np.frombuffer(memory, dtype="<u2", count=count,
                                  offset=offset + start)
            return (words.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        import numpy as np
        numpy_dtype = {"F32": "<f4", "F16": "<f2", "U32": "<u4"}[dtype]
        return np.frombuffer(memory, dtype=numpy_dtype, count=count,
                             offset=offset + start).copy().reshape(shape)

    def tensor(self, name, expected_shape, np, mx):
        value = self.raw(name)
        if value.dtype == np.uint32:
            prefix = name.removesuffix(".weight")
            scales = self.raw(prefix + ".scales")
            biases = self.raw(prefix + ".biases") if self._has(prefix + ".biases") else None
            if (value.ndim != 2 or value.shape[0] != expected_shape[0] or
                    scales.ndim != 2 or scales.shape[0] != expected_shape[0] or
                    scales.shape[1] * 32 != expected_shape[1]):
                raise ValueError(f"invalid Qwen3 affine geometry for {name}")
            bits = value.shape[1] * 32 // expected_shape[1]
            if bits not in (4, 8):
                raise ValueError(f"unsupported Qwen3 affine bit depth {bits}: {name}")
            q = mx.array(value, dtype=mx.uint32)
            s = mx.array(scales)
            b = mx.array(biases) if biases is not None else None
            dense = mx.dequantize(q, s, b, group_size=32, bits=bits,
                                  mode="affine", dtype=mx.float16, stream=mx.cpu)
            mx.eval(dense)
            result = np.array(dense, dtype=np.float16, copy=True)
            del q, s, b, dense
            mx.clear_cache()
            return result
        if value.shape != tuple(expected_shape):
            raise ValueError(f"unexpected Qwen3 tensor shape for {name}: {value.shape}")
        return value.astype(np.float16, copy=False)

    def _has(self, name):
        path = self.path_for(name)
        return name in self.headers.get(path, {})

    def provenance(self):
        result = {"checkpoint": str(self.checkpoint),
                  "checkpoint_bytes": self.checkpoint.stat().st_size,
                  "checkpoint_sha256": sha(self.checkpoint)}
        if self.weight_map is not None:
            result["checkpoint_shards"] = {
                path.name: {"bytes": path.stat().st_size, "sha256": sha(path)}
                for path in sorted(self.shards)
            }
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True,
                        help="Qwen3 text_encoder directory/index, or single Qwen3/Qwen21 checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bucket", type=int,
                        help="one fixed token bucket")
    parser.add_argument("--buckets",
                        help="comma-separated enumerated token buckets")
    parser.add_argument(
        "--min-profitable-rows",
        help=("comma-separated BUCKET:MIN_ACTUAL_ROWS crossovers; omitted "
              "buckets allow exact execution only"),
    )
    parser.add_argument("--ane-mlp-width", type=int, default=4864)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--mlp-width", type=int, default=MLP_WIDTH)
    parser.add_argument("--layer-count", type=int, default=LAYERS)
    parser.add_argument("--tensor-prefix", default="model",
                        help="checkpoint prefix before .layers (for example model.language_model)")
    parser.add_argument("--tensor-layout", choices=["qwen3", "qwen21"], default="qwen3",
                        help="qwen21 reads Comfy fused gate_up and img_mlp.out weights")
    parser.add_argument("--variant", choices=["int8_pc", "fp16"], default="int8_pc")
    parser.add_argument("--activation-precision", choices=["fp16", "int8"], default="fp16",
                        help="explicit dual activation Q/DQ with SmoothQuant (Qwen21 only)")
    parser.add_argument("--calibration-dir", type=Path,
                        help="real decode FFN inputs: block0/step1.npy ... block31/step1.npy")
    parser.add_argument("--sq-alpha1", type=float, default=0.5)
    parser.add_argument("--sq-alpha2", type=float, default=0.5)
    parser.add_argument("--activation-scale", type=float, default=8.0)
    parser.add_argument("--a8-boundary", choices=["both", "input", "hidden"], default="both",
                        help="research-only isolation of an A8 boundary; only both is W8A8")
    parser.add_argument("--down-weight-granularity", choices=["per_channel", "per_tensor", "per_block", "fp16"],
                        default="per_channel", help="research-only ANE down-projection W8 layout")
    parser.add_argument("--projected-weight-precision", choices=["int8", "fp16"], default="int8",
                        help="research-only upper projection precision (int8_pc variant only)")
    parser.add_argument("--tap-hidden", action="store_true",
                        help="research-only output after SwiGLU/optional hidden A8; excludes down projection")
    parser.add_argument("--down-op", choices=["conv", "linear"], default="conv",
                        help="research-only Core ML down-projection lowering")
    parser.add_argument("--output-scale", type=float, default=1.0,
                        help="divide the ANE branch before FP16 down projection")
    args = parser.parse_args()
    if bool(args.bucket) == bool(args.buckets):
        raise ValueError("select exactly one of --bucket or --buckets")
    buckets = ([args.bucket] if args.bucket else
               [int(value) for value in args.buckets.split(",")])
    if (not buckets or buckets != sorted(set(buckets)) or
            any(value < 64 or value > 8192 for value in buckets)):
        raise ValueError("buckets must be unique increasing integers in 64...8192")
    if len(buckets) > 128:
        raise ValueError("at most 128 enumerated buckets are supported")
    minimum_profitable_rows = {}
    if args.min_profitable_rows:
        for entry in args.min_profitable_rows.split(","):
            try:
                bucket_text, minimum_text = entry.split(":", 1)
                bucket, minimum = int(bucket_text), int(minimum_text)
            except ValueError as error:
                raise ValueError(
                    "min-profitable-rows must contain BUCKET:MIN pairs"
                ) from error
            if (bucket not in buckets or str(bucket) in minimum_profitable_rows or
                    minimum < 1 or minimum > bucket):
                raise ValueError(
                    "each minimum-profitable-row policy must name one selected "
                    "bucket and stay within 1...BUCKET"
                )
            minimum_profitable_rows[str(bucket)] = minimum
    if args.hidden <= 0 or args.hidden > 8192 or args.hidden % 32:
        raise ValueError("hidden must be a positive multiple of 32 up to 8192")
    if args.mlp_width <= 0 or args.mlp_width > 65536 or args.mlp_width % 32:
        raise ValueError("mlp-width must be a positive multiple of 32 up to 65536")
    if args.layer_count <= 0 or args.layer_count > 64:
        raise ValueError("layer-count must be 1...64")
    if args.tensor_layout == "qwen21" and (args.hidden != 4096 or args.mlp_width != 12288 or args.layer_count > 32):
        raise ValueError("Qwen21 DiT requires --hidden 4096 --mlp-width 12288 and at most 32 layers")
    if (not re.fullmatch(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*",
                         args.tensor_prefix) or
            args.tensor_prefix.startswith(".") or args.tensor_prefix.endswith(".")):
        raise ValueError("tensor-prefix must be a dotted tensor identifier")
    if (args.ane_mlp_width <= 0 or args.ane_mlp_width >= args.mlp_width or
            args.ane_mlp_width % 32):
        raise ValueError("ane-mlp-width must be a nonzero multiple of 32 below mlp-width")
    if (not math.isfinite(args.output_scale) or args.output_scale < 1.0 or
            args.output_scale > 256.0):
        raise ValueError("output-scale must be finite and in 1...256")
    if args.activation_precision == "int8":
        if (args.tensor_layout != "qwen21" or
                (args.variant != "int8_pc" and
                 not (args.variant == "fp16" and args.a8_boundary == "hidden")) or
                args.calibration_dir is None or len(buckets) != 1 or buckets[0] != 1024 or
                args.output_scale != 1 or
                not all(math.isfinite(v) and 0 <= v <= 1 for v in (args.sq_alpha1, args.sq_alpha2)) or
                not math.isfinite(args.activation_scale) or not 0.125 <= args.activation_scale <= 64):
            raise ValueError("Qwen21 W8A8 requires fixed 1024 rows, real calibration, INT8 weights and valid scales")
    elif args.calibration_dir is not None:
        raise ValueError("calibration requires --activation-precision int8")
    elif args.a8_boundary != "both":
        raise ValueError("isolated A8 boundaries require Qwen21 calibration")
    if args.down_weight_granularity != "per_channel" and args.activation_precision != "int8":
        raise ValueError("alternate W8 granularity requires Qwen21 A8 calibration")
    if args.projected_weight_precision != "int8" and (
            args.variant != "int8_pc" or args.activation_precision != "int8"):
        raise ValueError("FP16 upper projection requires Qwen21 A8 calibration")
    if (args.projected_weight_precision == "fp16" and
            args.down_weight_granularity == "fp16" and not args.tap_hidden):
        raise ValueError("int8_pc diagnostic requires at least one W8 projection")
    if args.tap_hidden and (args.activation_precision != "int8" or
                            args.variant != "int8_pc" or args.layer_count != 1 or
                            args.ane_mlp_width != args.hidden or
                            len(buckets) != 1):
        raise ValueError("hidden tap requires a single fixed-shape Qwen21 A8 layer and matching width")
    if args.down_op != "conv" and (args.activation_precision != "int8" or len(buckets) != 1):
        raise ValueError("linear down projection requires fixed-shape Qwen21 A8 calibration")

    import numpy as np
    import mlx.core as mx
    import coremltools as ct
    import coremltools.optimize.coreml as optimize
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import get_new_symbol, types
    if args.activation_precision == "int8":
        from z_image_smoothquant import (load_calibration, smooth_partial_ffn,
                                          calibrate_hidden_a8, verify_w8a8)

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    source = QwenSource(args.model)
    calibration = {}
    if args.activation_precision == "int8":
        for layer in range(args.layer_count):
            _, digest = load_calibration(args.calibration_dir / f"block{layer}",
                                         buckets[0], args.hidden, np)
            calibration[str(layer)] = digest
    output = args.output.absolute()
    if output.is_symlink():
        raise ValueError("symlink output unsupported")
    output.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output / ".export.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        provenance = source.provenance()
        identity = {
            "owner": ("turbocider.qwen21.dit.coreml.v1" if args.tensor_layout == "qwen21"
                      else "turbocider.qwen3.encoder.coreml.v1"),
            **({"tensor_layout": "qwen21"} if args.tensor_layout == "qwen21" else {}),
            **provenance,
            "buckets": buckets,
            "hidden": args.hidden,
            "mlp_width": args.mlp_width,
            "ane_mlp_start": 0,
            "ane_mlp_end": args.ane_mlp_width,
            "output_scale": args.output_scale,
            "variant": args.variant,
            **({"activation_precision": "int8", "calibration_sha256": calibration,
                "sq_alpha1": args.sq_alpha1, "sq_alpha2": args.sq_alpha2,
                "activation_scale": args.activation_scale, "a8_graph": "sq_v1_" + args.a8_boundary}
               if calibration else {}),
            **({"down_weight_granularity": args.down_weight_granularity}
               if calibration and args.down_weight_granularity != "per_channel" else {}),
            **({"down_op": args.down_op} if args.down_op != "conv" else {}),
            **({"projected_weight_precision": args.projected_weight_precision}
               if args.projected_weight_precision != "int8" else {}),
            **({"tap_hidden": True} if args.tap_hidden else {}),
            "tensor_prefix": args.tensor_prefix,
            **({"minimum_profitable_rows": minimum_profitable_rows}
               if minimum_profitable_rows else {}),
            "coremltools": ct.__version__,
            "numpy": np.__version__,
            "recipe": 1 if source.weight_map is None else 2,
        }
        marker = output / ".turbocider-export.json"
        if marker.exists() or marker.is_symlink():
            if (marker.is_symlink() or not marker.is_file() or
                    json.loads(marker.read_text()) != identity):
                raise ValueError("export identity changed; choose a new output directory")
        else:
            allowed = {".export.lock", "export.log", "progress.json"}
            if any(item.name not in allowed for item in output.iterdir()):
                raise ValueError("refusing to adopt a nonempty export directory")
            atom(marker, identity)

        artifacts, checksums, activation_receipts = {}, {}, {}
        with source:
            for layer in range(args.layer_count):
                name = f"block{layer}_mlp_branch.{args.variant}.mlpackage"
                destination = output / name
                receipt = output / f"block{layer}.json"
                destination_exists = destination.exists() or destination.is_symlink()
                receipt_exists = receipt.exists() or receipt.is_symlink()
                if destination_exists != receipt_exists:
                    raise ValueError(f"incomplete Qwen3 artifact must be removed: block {layer}")
                if destination_exists:
                    saved = artifact_receipt(destination, receipt)
                    checksums[str(layer)] = saved
                    if calibration:
                        activation_receipt = output / f"block{layer}.activation.json"
                        if activation_receipt.is_symlink() or not activation_receipt.is_file():
                            raise ValueError(f"missing Qwen21 activation receipt: block {layer}")
                        activation_receipts[str(layer)] = json.loads(activation_receipt.read_text())
                else:
                    gate, up, down = mlp_weights(source, layer, args.tensor_layout, args.tensor_prefix,
                                                args.hidden, args.mlp_width, np, mx)
                    width = args.ane_mlp_width
                    inverse_s1 = None
                    input_a8_scale = hidden_a8_scale = None
                    if calibration:
                        samples, digest = load_calibration(args.calibration_dir / f"block{layer}",
                                                            buckets[0], args.hidden, np)
                        if digest != calibration[str(layer)]:
                            raise ValueError("Qwen21 calibration changed during export")
                        gate, up, down, s1, _ = smooth_partial_ffn(
                            gate[:width], up[:width], down[:, :width], samples,
                            args.sq_alpha1, args.sq_alpha2, np)
                        inverse_s1 = np.ascontiguousarray(
                            (1 / s1).astype(np.float16).reshape(1, args.hidden, 1, 1))
                        maximum = max(float(np.max(np.abs(sample / s1))) for sample in samples)
                        input_a8_scale = np.float16(max(maximum / 127, 1e-6))
                        search = []
                        if args.a8_boundary != "input":
                            # Uniformly amplifying the hidden activation and
                            # shrinking down weights is algebraically exact.
                            # Search at scale>=1, then transform the chosen
                            # A8 step; the shared Z-Image search API rejects
                            # subunit scales for its own deployment contract.
                            search_scale = max(1., args.activation_scale)
                            hidden_a8_scale, search = calibrate_hidden_a8(
                                gate, up, down, s1, samples, float(input_a8_scale),
                                search_scale, np, outlier_rows=True)
                            hidden_a8_scale = np.float16(
                                float(hidden_a8_scale) * (search_scale / args.activation_scale) ** 2)
                        activation_receipts[str(layer)] = {
                            "input_a8_scale": float(input_a8_scale),
                            "hidden_a8_scale": float(hidden_a8_scale) if hidden_a8_scale is not None else None,
                            "hidden_threshold_search": search,
                            "hidden_search_scale": max(1., args.activation_scale),
                            "calibration_sha256": digest,
                        }
                        if hidden_a8_scale is not None:
                            down = down * np.float32(args.activation_scale ** 2)
                    else:
                        gate, up, down = gate[:width], up[:width], down[:, :width]
                    first = np.ascontiguousarray(np.concatenate((gate, up), axis=0).astype(np.float16)[:, :, None, None])
                    last = np.ascontiguousarray(down.astype(np.float16)[:, :, None, None])
                    del gate, up, down

                    rows = buckets[0] if len(buckets) == 1 else get_new_symbol()
                    convert_inputs = {}
                    if len(buckets) > 1:
                        shape = ct.EnumeratedShapes(
                            shapes=[(1, args.hidden, 1, value) for value in buckets],
                            default=(1, args.hidden, 1, buckets[0]))
                        convert_inputs["inputs"] = [
                            ct.TensorType(name="x", shape=shape, dtype=np.float16)]
                    @mb.program(input_specs=[mb.TensorSpec(shape=(1, args.hidden, 1, rows), dtype=types.fp16)],
                                opset_version=ct.target.macOS15)
                    def branch(x):
                        projected_input = x
                        if inverse_s1 is not None:
                            projected_input = mb.mul(x=x, y=inverse_s1, name="smooth_input")
                            if args.a8_boundary != "hidden":
                                quantized = mb.quantize(input=projected_input, scale=input_a8_scale,
                                                         output_dtype="int8", name="a8_input")
                                projected_input = mb.dequantize(input=quantized, scale=input_a8_scale,
                                                                 name="a8_input_dequantized")
                        projected = mb.conv(x=projected_input, weight=first, pad_type="valid", name="projected")
                        gate_value, up_value = mb.split(x=projected, num_splits=2, axis=1)
                        activation = mb.silu(x=gate_value)
                        if hidden_a8_scale is not None:
                            reciprocal = np.float16(1 / args.activation_scale)
                            activation = mb.mul(x=activation, y=reciprocal)
                            up_value = mb.mul(x=up_value, y=reciprocal)
                        activation = mb.mul(x=activation, y=up_value)
                        if hidden_a8_scale is not None:
                            quantized = mb.quantize(input=activation, scale=hidden_a8_scale,
                                                     output_dtype="int8", name="a8_hidden")
                            activation = mb.dequantize(input=quantized, scale=hidden_a8_scale,
                                                        name="a8_hidden_dequantized")
                        if args.tap_hidden:
                            return mb.reshape(x=activation, shape=(1, args.hidden, 1, rows), name="y")
                        if args.output_scale != 1.0:
                            activation = mb.mul(x=activation,
                                                y=np.float16(1.0 / args.output_scale),
                                                name="output_scale")
                        if args.down_op == "linear":
                            tokens = mb.reshape(
                                x=mb.transpose(x=activation, perm=[0, 3, 2, 1]),
                                shape=(rows, width))
                            projected = mb.linear(x=tokens, weight=last[:, :, 0, 0], name="down_linear")
                            return mb.transpose(
                                x=mb.reshape(x=projected, shape=(1, rows, 1, args.hidden)),
                                perm=[0, 3, 2, 1], name="y")
                        return mb.conv(x=activation, weight=last, pad_type="valid", name="y")

                    model = ct.convert(branch, convert_to="mlprogram",
                                       minimum_deployment_target=ct.target.macOS15,
                                       compute_precision=ct.precision.FLOAT16,
                                       skip_model_load=True, **convert_inputs)
                    compressed = model
                    if args.variant == "int8_pc":
                        w8 = optimize.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8",
                                                               granularity="per_channel", block_size=32,
                                                               weight_threshold=0)
                        down_w8 = (None if args.tap_hidden else
                                   w8 if args.down_weight_granularity == "per_channel" else
                                   None if args.down_weight_granularity == "fp16" else
                                   optimize.OpLinearQuantizerConfig(
                                       mode="linear_symmetric", dtype="int8",
                                       granularity=args.down_weight_granularity, block_size=32,
                                       weight_threshold=0))
                        quantizer = (optimize.OptimizationConfig(global_config=None,
                            op_name_configs={**({"projected": w8}
                                               if args.projected_weight_precision == "int8" else {}),
                                             **({} if down_w8 is None else {
                                                 "down_linear" if args.down_op == "linear" else "y": down_w8})})
                            if calibration else optimize.OptimizationConfig(global_config=w8))
                        compressed = optimize.linear_quantize_weights(model, quantizer)
                    if calibration and args.a8_boundary == "both":
                        expected_w8 = tuple(
                            (["projected"] if args.projected_weight_precision == "int8" else []) +
                            (["down_linear" if args.down_op == "linear" else "y"]
                             if down_w8 is not None else []))
                        if expected_w8:
                            verify_w8a8(compressed.get_spec(), expected_w8)
                    with tempfile.TemporaryDirectory(prefix=".export-", dir=output) as temporary:
                        package = Path(temporary) / name
                        compressed.save(package)
                        os.replace(package, destination)
                    entries = sorted(destination.rglob("*"))
                    if any(path.is_symlink() for path in entries):
                        raise ValueError(f"generated Qwen3 artifact contains a symlink: block {layer}")
                    saved = {str(path.relative_to(destination)): sha(path)
                             for path in entries if path.is_file()}
                    if not saved:
                        raise ValueError(f"generated Qwen3 artifact is empty: block {layer}")
                    atom(receipt, saved)
                    if calibration:
                        atom(output / f"block{layer}.activation.json", activation_receipts[str(layer)])
                    checksums[str(layer)] = saved
                    del first, last, branch, model, compressed
                    gc.collect()
                # The repository's source/compiled manifest ABI names the
                # selectable branch `int8_pc`; the artifact filename and
                # export identity retain whether its weights are FP16 or INT8.
                artifacts[str(layer)] = {"int8_pc": name}
                atom(output / "progress.json", {"completed": layer + 1, "total": args.layer_count})
                print(f"partition {layer + 1}/{args.layer_count} ready", flush=True)

        source_identity = source.provenance()
        if source_identity != provenance:
            raise ValueError("Qwen3 checkpoint changed during export")
        atom(output / "manifest.json", {
            "schema_version": 2,
            "source": {**provenance, "blocks": list(range(args.layer_count))},
            "shape": {"K": args.hidden, "N": args.hidden, "mlp_width": args.mlp_width,
                      "ane_mlp_start": 0, "ane_mlp_end": args.ane_mlp_width,
                      "output_scale": args.output_scale,
                      "buckets": buckets,
                      **({"minimum_profitable_rows": minimum_profitable_rows}
                         if minimum_profitable_rows else {}),
                      **({"input_mode": "enumerated", "default_bucket": buckets[0]}
                         if len(buckets) > 1 else {})},
            "functions": {str(bucket): "main" for bucket in buckets},
            "artifacts": artifacts,
            "artifact_sha256": checksums,
            **({"activation_quantization": activation_receipts} if calibration else {}),
            "export_identity": identity,
        })
        print(json.dumps({"source_manifest": str(output / "manifest.json"),
                          "checkpoint": str(source.checkpoint), "blocks": args.layer_count}), flush=True)


if __name__ == "__main__":
    main()
