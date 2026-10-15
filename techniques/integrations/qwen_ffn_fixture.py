#!/usr/bin/env python3
"""Export one real Qwen FFN layer and captured activations for a generic lab.

Local source weights and original BF16 captures only. No LoRA, download, model
rewrite or installed App changes. All exported arrays belong in ignored output.
The BF16 reference mirrors the compiled fused GPU graph, not a request oracle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def header(path):
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("short safetensors header")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= 16 * 1024 * 1024 or length + 8 > path.stat().st_size:
            raise ValueError("unbounded safetensors header")
        value = json.loads(stream.read(length))
    return value, length + 8


def bf16_array(path, key, expected_shape, np):
    info, offset = header(path)
    item = info[key]
    begin, end = item["data_offsets"]
    count = 1
    for dimension in expected_shape:
        count *= dimension
    if (item["dtype"] != "BF16" or item["shape"] != list(expected_shape)
            or type(begin) is not int or type(end) is not int or begin < 0
            or end - begin != count * 2 or offset + end > path.stat().st_size):
        raise ValueError("source BF16 tensor geometry or byte range differs: " + key)
    bits = np.memmap(path, mode="r", dtype="<u2", offset=offset + begin, shape=expected_shape)
    source_sha = hashlib.sha256(bits.tobytes()).hexdigest()
    value = (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)
    if not np.isfinite(value).all():
        raise ValueError("source contains nonfinite values")
    return value, source_sha, info.get("__metadata__", {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True, help="completed capture_qwen_ffn.py output")
    parser.add_argument("--layer", type=int, choices=(0, 15, 31), default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import mlx.core as mx
    import numpy as np
    before = args.checkpoint.stat()
    capture = json.loads((args.capture_root / "receipt.json").read_text())
    if capture.get("status") != "captured" or capture.get("loras") != [] or capture.get("steps") != 20:
        parser.error("requires a completed base 20-step capture with no LoRA")
    if any(getattr(before, key) != value for key, value in capture["checkpoint_stat_before"].items()):
        parser.error("capture and checkpoint stat identities differ")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prefix = f"transformer_blocks.{args.layer}.img_mlp."
    gu, gu_sha, _ = bf16_array(args.checkpoint, prefix + "gate_up.weight", (24576, 4096), np)
    down, down_sha, _ = bf16_array(args.checkpoint, prefix + "out.weight", (4096, 12288), np)
    gpu_gu, gpu_down = mx.array(gu, dtype=mx.bfloat16), mx.array(down, dtype=mx.bfloat16)
    gu_half, down_half = gu.astype(np.float16), down.astype(np.float16)
    if not np.isfinite(gu_half).all() or not np.isfinite(down_half).all():
        raise ValueError("FP16 weight conversion overflow")
    np.save(output / "gate_up.npy", gu_half, allow_pickle=False)
    np.save(output / "down.npy", down_half, allow_pickle=False)
    mx.eval(gpu_gu, gpu_down)
    del gu, down, gu_half, down_half

    @mx.compile
    def original(x):
        gate, up = mx.split(x @ gpu_gu.T, 2, axis=-1)
        return ((gate * mx.sigmoid(gate)) * up) @ gpu_down.T

    points = []
    for step in (1, 10, 19):
        filename = f"layer-{args.layer}-step-{step}.safetensors"
        candidates = [record for record in capture["records"] if Path(record["path"]).name == filename]
        if len(candidates) != 1:
            raise ValueError("missing unique captured point: " + filename)
        path = (args.capture_root / candidates[0]["path"]).resolve()
        if not path.is_relative_to(args.capture_root.resolve()) or digest(path) != candidates[0]["sha256"]:
            raise ValueError("capture escaped its root or content changed")
        x, source_sha, metadata = bf16_array(path, "input", (1, 1024, 4096), np)
        if (metadata.get("capture_recipe") != "qwen21-real-ffn-bf16-v1"
                or metadata.get("trajectory") != "forced-gpu" or metadata.get("layer") != str(args.layer)
                or metadata.get("step") != str(step) or metadata.get("total_steps") != "20"):
            raise ValueError("captured point provenance differs")
        x = x.reshape(1024, 4096)
        half = x.astype(np.float16)
        if not np.isfinite(half).all():
            raise ValueError("FP16 activation conversion overflow")
        baseline = original(mx.array(x, dtype=mx.bfloat16))
        mx.eval(baseline)
        reference = np.array(baseline.astype(mx.float32))
        if not np.isfinite(reference).all():
            raise ValueError("original BF16 replica produced nonfinite output")
        np.save(output / f"input-{step}.npy", half, allow_pickle=False)
        np.save(output / f"original-reference-{step}.npy", reference, allow_pickle=False)
        points.append({"id": f"layer-{args.layer}-step-{step}", "step": step,
                       "input": f"input-{step}.npy", "original_reference": f"original-reference-{step}.npy",
                       "original_input_tensor_sha256": source_sha, "capture_file_sha256": candidates[0]["sha256"]})
        print("exported " + points[-1]["id"], flush=True)
    mx.synchronize()
    after = args.checkpoint.stat()
    if any(getattr(before, key) != getattr(after, key) for key in capture["checkpoint_stat_before"]):
        raise ValueError("checkpoint changed during export")
    report = {"schema": "tc-real-ffn-fixture-v1", "synthetic": False, "shape_mhf": [1024, 4096, 12288],
              "source_layer": args.layer, "loras": [], "dtype": "float16", "gate_up": "gate_up.npy", "down": "down.npy",
              "source_encoding": "original BF16 checkpoint and original BF16 forced-GPU activations, converted to FP16",
              "original_gate_up_tensor_sha256": gu_sha, "original_down_tensor_sha256": down_sha,
              "capture_receipt_sha256": digest(args.capture_root / "receipt.json"),
              "native_capture_runtime_build_id": capture["runtime_build_id"], "exporter_sha256": digest(Path(__file__)),
              "original_reference_scope": "compiled MLX BF16 full fused gate/up, SiLU*up, down replica; not full request parity",
              "points": points, "files": {path.name: digest(path) for path in sorted(output.glob("*.npy"))},
              "production_qualified": False}
    (output / "fixture.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
