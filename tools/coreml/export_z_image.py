"""Offline Z-Image-Turbo FFN partition exporter.

The exporter is intentionally repository-local and does not import ComfyUI or
diffusers.  It reads the official ComfyUI split-file, a diffusers single-file
transformer checkpoint, or a Hugging Face/ModelScope safetensors index with
multiple shards.  It emits one Core ML INT8 per-channel artifact per
transformer block and publishes a provenance-bound manifest consumed by the
native runtime.

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
BLOCKS = [f"noise_refiner.{i}" for i in range(2)] + [f"layers.{i}" for i in range(30)]


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


class SafetensorsSource:
    """One safetensors file or an index-bound set of immutable shards."""

    def __init__(self, checkpoint: Path, weight_map=None):
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise ValueError(f"checkpoint must be a regular non-symlink file: {checkpoint}")
        self.checkpoint = checkpoint.resolve()
        self.weight_map = weight_map
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

    @classmethod
    def from_model(cls, model: Path):
        index = model / "transformer/diffusion_pytorch_model.safetensors.index.json"
        if index.is_file() and not index.is_symlink():
            value = json.loads(index.read_text())
            if not isinstance(value, dict):
                raise ValueError("safetensors index must be a JSON object")
            return cls(index, value.get("weight_map"))
        single_candidates = [
            model / "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
            model / "transformer/diffusion_pytorch_model.safetensors",
        ]
        for candidate in single_candidates:
            if candidate.is_file() and not candidate.is_symlink():
                return cls(candidate)
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

    def tensor(self, name, shape, np):
        path = self.tensor_path(name)
        header = self._headers.get(path)
        memory = self._memories.get(path)
        if header is None or memory is None:
            raise ValueError(f"safetensors shard is not open: {path}")
        meta = header.get(name)
        if not meta or meta.get("shape") != list(shape) or meta.get("dtype") not in {"BF16", "F16", "F32"}:
            raise ValueError(f"unexpected tensor shape/dtype: {name}")
        start, end = meta["data_offsets"]
        count = int(np.prod(shape))
        data_offset = self._data_offsets[path]
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


def checkpoint_for(model: Path) -> Path:
    """Compatibility helper used by external tooling."""
    return SafetensorsSource.from_model(model).checkpoint


def parse_blocks(value: str):
    if value == "all":
        return list(range(len(BLOCKS)))
    result = []
    for item in value.split(","):
        number = int(item)
        if number < 0 or number >= len(BLOCKS):
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bucket", type=int, default=4608)
    parser.add_argument("--shape-mode", choices=["fixed", "enumerated", "range"], default="fixed")
    parser.add_argument("--min-bucket", type=int, default=1056,
                        help="minimum and default row count for flexible exports")
    parser.add_argument("--bucket-step", type=int, default=32)
    parser.add_argument("--ane-mlp-width", type=int, default=7680)
    parser.add_argument("--activation-scale", type=float, default=8.0)
    parser.add_argument("--output-scale", type=float, default=32.0)
    parser.add_argument("--variant", choices=["int8_pc", "fp16"], default="int8_pc")
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
    if args.ane_mlp_width <= 0 or args.ane_mlp_width >= MLP_WIDTH:
        raise ValueError("ane-mlp-width must be in 1...10239")
    if not 1.0 <= args.activation_scale <= 64.0:
        raise ValueError("activation-scale must be 1...64")
    if not 1.0 <= args.output_scale <= 256.0:
        raise ValueError("output-scale must be 1...256")
    block_indexes = parse_blocks(args.blocks)

    import numpy as np
    import coremltools as ct
    import coremltools.optimize.coreml as optimize
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types, get_new_symbol

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    checkpoint_source = SafetensorsSource.from_model(args.model)
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
            "owner": "turbocider.z_image.coreml.v1",
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
            "recipe": 2 if checkpoint_source.weight_map is not None else 1,
        }
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
            for ordinal in block_indexes:
                prefix = BLOCKS[ordinal]
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
                    w1 = reader.tensor(f"{prefix}.feed_forward.w1.weight", (MLP_WIDTH, HIDDEN), np)
                    w2 = reader.tensor(f"{prefix}.feed_forward.w2.weight", (HIDDEN, MLP_WIDTH), np)
                    w3 = reader.tensor(f"{prefix}.feed_forward.w3.weight", (MLP_WIDTH, HIDDEN), np)
                    w1, _ = apply_lora(f"{prefix}.feed_forward.w1.weight", w1, lora_bundles, np)
                    w2, _ = apply_lora(f"{prefix}.feed_forward.w2.weight", w2, lora_bundles, np)
                    w3, _ = apply_lora(f"{prefix}.feed_forward.w3.weight", w3, lora_bundles, np)
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
                        projected = mb.conv(x=x, weight=first, pad_type="valid", name="projected")
                        gate, up = mb.split(x=projected, num_splits=2, axis=1)
                        reciprocal = np.float16(1.0 / args.activation_scale)
                        gated = mb.mul(x=mb.silu(x=gate), y=reciprocal)
                        raised = mb.mul(x=up, y=reciprocal)
                        return mb.conv(x=mb.mul(x=gated, y=raised), weight=last,
                                       pad_type="valid", name="y")

                    model = ct.convert(
                        branch,
                        convert_to="mlprogram",
                        minimum_deployment_target=ct.target.macOS15,
                        compute_precision=ct.precision.FLOAT16,
                        skip_model_load=True,
                        **convert_inputs,
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
                    del first, last, branch, model, compressed
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
            "export_identity": identity,
        })
        print(json.dumps({"source_manifest": str(output / "manifest.json"),
                          "checkpoint": str(checkpoint), "blocks": block_indexes}), flush=True)


if __name__ == "__main__":
    main()
