"""Check 32 Qwen21 Core ML FFN branches on held-out real decode activations.

This is an offline, single-step layer screen: it does not substitute for
full-image quality, GPU/ANE overlap, or physical Neural Engine occupancy.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import time

import coremltools as ct
import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen21_w8a8_layer_probe import metrics


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_artifact(root, name):
    if not isinstance(name, str) or "\\" in name:
        raise ValueError("invalid Core ML artifact name")
    relative = Path(name)
    if (relative.is_absolute() or len(relative.parts) != 1 or
            not re.fullmatch(r"block[0-9]+_mlp_branch\.int8_pc\.mlpackage", name)):
        raise ValueError("expected a Qwen21 source .mlpackage (coremltools cannot load managed .mlmodelc)")
    path = root / relative
    if not path.is_dir() or path.is_symlink():
        raise ValueError("missing or symlinked Core ML artifact")
    return path


def validate_manifest(manifest, source, checkpoint):
    identity = manifest.get("export_identity", {})
    if (manifest.get("schema_version") != 2 or
            identity.get("tensor_layout") != "qwen21" or
            identity.get("activation_precision") != "int8" or
            identity.get("a8_graph") != "sq_v1_both" or
            identity.get("projected_weight_granularity") != "per_tensor" or
            identity.get("variant") != "int8_pc" or
            manifest.get("shape", {}).get("buckets") != [1024] or
            manifest["shape"].get("ane_mlp_end") != 4096 or
            set(manifest["artifacts"]) != {str(i) for i in range(32)} or
            manifest.get("source", {}).get("checkpoint_sha256") != file_sha256(checkpoint)):
        raise ValueError("expected checkpoint-bound, complete 32-layer Qwen21 W8A8 manifest")
    for block in range(32):
        artifact = manifest["artifacts"][str(block)]
        name = artifact.get("int8_pc") if isinstance(artifact, dict) else None
        if name != f"block{block}_mlp_branch.int8_pc.mlpackage":
            raise ValueError(f"unexpected Core ML source artifact for block {block}")
        safe_artifact(source.parent, name)


def bf16_branch(weights, layer, x):
    prefix = f"transformer_blocks.{layer}.img_mlp."
    gate_up = weights[prefix + "gate_up.weight"]
    down = weights[prefix + "out.weight"]
    value = mx.array(x[None]).astype(mx.bfloat16)
    projected = mx.concatenate([gate_up[:4096], gate_up[12288:16384]])
    gate, up = mx.split(mx.matmul(value, projected.T), 2, axis=-1)
    reference = mx.matmul(gate * mx.sigmoid(gate) * up, down[:, :4096].T)
    mx.eval(reference)
    return np.array(reference.astype(mx.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--heldout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", default=",".join(str(i) for i in range(32)))
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 30:
        parser.error("iterations must be 1...30")
    blocks = [int(part) for part in args.blocks.split(",")]
    if blocks != sorted(set(blocks)) or any(block < 0 or block >= 32 for block in blocks):
        parser.error("blocks must be unique, ordered indices in 0...31")
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest, args.manifest, args.checkpoint)
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists; choose a new report name")
    weights = mx.load(str(args.checkpoint))
    report = {"scope": "held-out cached-decode single-layer FFN outputs; not e2e image quality or ANE residency",
              "source_manifest_sha256": file_sha256(args.manifest),
              "checkpoint_sha256": manifest["source"]["checkpoint_sha256"],
              "compute_units": "CPU_AND_NE", "layers": {}}
    for block in blocks:
        sample_path = args.heldout_dir / f"block{block}" / "step1.npy"
        if sample_path.is_symlink() or not sample_path.is_file():
            raise ValueError(f"missing real held-out input for block {block}")
        sample = np.load(sample_path, mmap_mode="r", allow_pickle=False)
        if sample.shape != (1024, 4096) or sample.dtype != np.float32 or not np.isfinite(sample).all():
            raise ValueError(f"invalid held-out input for block {block}")
        digest = hashlib.sha256(b"step1.npy\0")
        with sample_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() == manifest["activation_quantization"][str(block)]["calibration_sha256"]:
            raise ValueError(f"block {block} held-out activation is a calibration sample")
        supplied = np.ascontiguousarray(sample.astype(np.float16).T[None, :, None, :])
        reference = bf16_branch(weights, block, sample)
        model_path = args.manifest.parent / manifest["artifacts"][str(block)]["int8_pc"]
        start = time.perf_counter()
        model = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        load_seconds = time.perf_counter() - start
        durations, actual = [], None
        for iteration in range(args.iterations + 2):
            start = time.perf_counter()
            actual = model.predict({"x": supplied})["y"]
            if iteration >= 2:
                durations.append(time.perf_counter() - start)
        actual = np.ascontiguousarray(actual.transpose(0, 3, 2, 1).reshape(1, 1024, 4096))
        report["layers"][str(block)] = {
            "input_sha256": file_sha256(sample_path),
            "ane_branch": metrics(actual, reference),
            "prediction_median_seconds": statistics.median(durations),
            "load_seconds": load_seconds,
        }
        print(f"block {block}: rRMSE={report['layers'][str(block)]['ane_branch']['relative_rmse']:.5f}, "
              f"Core ML={1000*statistics.median(durations):.2f} ms", flush=True)
        del model, actual, reference, supplied, sample
        gc.collect()
        mx.clear_cache()
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
