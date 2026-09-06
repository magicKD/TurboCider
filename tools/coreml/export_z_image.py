"""Offline Z-Image-Turbo FFN partition exporter.

The exporter is intentionally repository-local and does not import ComfyUI or
diffusers.  It reads the official ComfyUI split-file (or the equivalent
diffusers single-file transformer checkpoint), emits one Core ML INT8
per-channel artifact per transformer block, and publishes a provenance-bound
manifest consumed by the native runtime.

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
import tempfile
from pathlib import Path


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


def checkpoint_for(model: Path) -> Path:
    candidates = [
        model / "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
        model / "transformer/diffusion_pytorch_model.safetensors",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError("Z-Image transformer checkpoint not found under model root")


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
    parser.add_argument("--ane-mlp-width", type=int, default=7680)
    parser.add_argument("--activation-scale", type=float, default=8.0)
    parser.add_argument("--output-scale", type=float, default=32.0)
    parser.add_argument("--variant", choices=["int8_pc", "fp16"], default="int8_pc")
    parser.add_argument("--blocks", default="all", help="all or comma-separated block indexes")
    args = parser.parse_args()
    if not 64 <= args.bucket <= 8192:
        raise ValueError("bucket must be 64...8192")
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
    from coremltools.converters.mil.mil import types

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    checkpoint = checkpoint_for(args.model)
    output = args.output.absolute()
    if output.is_symlink():
        raise ValueError("symlink output unsupported")
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".export.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {
            "owner": "turbocider.z_image.coreml.v1",
            "checkpoint": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": sha(checkpoint),
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
            "recipe": 1,
        }
        marker = output / ".turbocider-export.json"
        if marker.exists():
            if marker.is_symlink() or json.loads(marker.read_text()) != identity:
                raise ValueError("export identity changed; select a new output directory")
        else:
            allowed = {".export.lock", "export.log", "progress.json"}
            if any(item.name not in allowed for item in output.iterdir()):
                raise ValueError("refusing to adopt a nonempty export directory")
            atom(marker, identity)

        with checkpoint.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as memory:
            header_size, header = load_header(stream)
            data_offset = 8 + header_size

            def tensor(name, shape):
                meta = header.get(name)
                if not meta or meta.get("shape") != list(shape) or meta.get("dtype") not in {"BF16", "F16"}:
                    raise ValueError(f"unexpected tensor shape/dtype: {name}")
                start, end = meta["data_offsets"]
                count = int(np.prod(shape))
                if start < 0 or end - start != count * 2 or data_offset + end > len(memory):
                    raise ValueError(f"invalid safetensors offsets: {name}")
                if meta["dtype"] == "BF16":
                    words = np.frombuffer(memory, dtype="<u2", count=count, offset=data_offset + start)
                    return (words.astype(np.uint32) << 16).view(np.float32).astype(np.float16).reshape(shape)
                return np.frombuffer(memory, dtype="<f2", count=count, offset=data_offset + start).copy().reshape(shape)

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
                    w1 = tensor(f"{prefix}.feed_forward.w1.weight", (MLP_WIDTH, HIDDEN))
                    w2 = tensor(f"{prefix}.feed_forward.w2.weight", (HIDDEN, MLP_WIDTH))
                    w3 = tensor(f"{prefix}.feed_forward.w3.weight", (MLP_WIDTH, HIDDEN))
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

                    @mb.program(
                        input_specs=[mb.TensorSpec(shape=(1, HIDDEN, 1, args.bucket), dtype=types.fp16)],
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

        if sha(checkpoint) != identity["checkpoint_sha256"]:
            raise ValueError("checkpoint changed during export")
        atom(output / "manifest.json", {
            "schema_version": 2,
            "source": {
                "checkpoint": str(checkpoint),
                "checkpoint_bytes": identity["checkpoint_bytes"],
                "checkpoint_sha256": identity["checkpoint_sha256"],
                "blocks": block_indexes,
            },
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
