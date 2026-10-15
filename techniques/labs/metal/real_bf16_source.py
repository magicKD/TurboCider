"""Bounded explicit BF16 source reader; no product imports, writes or downloads."""
import hashlib
import json
from pathlib import Path
import struct


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            result.update(part)
    return result.hexdigest()


def tensor(path, key, shape, expected_sha, np):
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError("short safetensors prefix")
        size = struct.unpack("<Q", raw)[0]
        if not 2 <= size <= 16 * 1024 * 1024 or size + 8 > path.stat().st_size:
            raise ValueError("unbounded safetensors header")
        header = json.loads(stream.read(size))
    item = header[key]
    begin, end = item["data_offsets"]
    elements = 1
    for dimension in shape:
        elements *= dimension
    if (item["dtype"] != "BF16" or item["shape"] != list(shape)
            or type(begin) is not int or type(end) is not int or begin < 0
            or end - begin != elements * 2 or size + 8 + end > path.stat().st_size):
        raise ValueError("BF16 source shape or byte bounds differ")
    bits = np.memmap(path, mode="r", dtype="<u2", offset=size + 8 + begin, shape=shape)
    if hashlib.sha256(bits.tobytes()).hexdigest() != expected_sha:
        raise ValueError("original BF16 tensor identity differs")
    value = (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)
    if not np.isfinite(value).all():
        raise ValueError("nonfinite original BF16 source")
    return value, header.get("__metadata__", {})


def load(checkpoint, capture_root, fixture, points, np):
    layer = fixture["source_layer"]
    if type(layer) is not int or not 0 <= layer < 32:
        raise ValueError("bounded model layer required")
    m, h, f = fixture["shape_mhf"]
    stem = f"transformer_blocks.{layer}.img_mlp."
    gu, _ = tensor(checkpoint, stem + "gate_up.weight", (2 * f, h), fixture["original_gate_up_tensor_sha256"], np)
    down, _ = tensor(checkpoint, stem + "out.weight", (h, f), fixture["original_down_tensor_sha256"], np)
    receipt_path = capture_root / "receipt.json"
    if digest(receipt_path) != fixture["capture_receipt_sha256"]:
        raise ValueError("original capture receipt differs")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "captured" or receipt.get("loras") != []:
        raise ValueError("completed base capture required")
    for point in points:
        info = next(item for item in fixture["points"] if item["id"] == point["id"])
        basename = f'layer-{layer}-step-{info["step"]}.safetensors'
        candidates = [item for item in receipt["records"] if Path(item["path"]).name == basename]
        if len(candidates) != 1:
            raise ValueError("missing unique original activation")
        path = (capture_root / candidates[0]["path"]).resolve()
        if not path.is_relative_to(capture_root.resolve()) or digest(path) != info["capture_file_sha256"]:
            raise ValueError("original activation escaped or changed")
        value, metadata = tensor(path, "input", (1, m, h), info["original_input_tensor_sha256"], np)
        if metadata.get("trajectory") != "forced-gpu" or metadata.get("capture_recipe") != "qwen21-real-ffn-bf16-v1":
            raise ValueError("original capture recipe differs")
        point["x_bf16_source"] = value.reshape(m, h)
        if not np.array_equal(point["x_bf16_source"].astype(np.float16).view(np.uint16), point["x"].view(np.uint16)):
            raise ValueError("FP16 fixture is not the original BF16 activation conversion")
    return [gu, down]
