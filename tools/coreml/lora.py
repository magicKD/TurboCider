"""Small repository-local LoRA reader used by the optional Core ML exporters.

The native runtime already applies LoRA deltas to MLX tensors in memory.  This
module mirrors the same A/B, up/down, alpha and fused-projection name rules so
an ANE artifact can be built for exactly the same independent adapter without
writing a second merged checkpoint.
"""

import hashlib
import json
import math
import struct
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(paths, strengths, roles):
    if len(paths) != len(strengths) or len(paths) != len(roles):
        raise ValueError("LoRA paths, strengths and roles must have equal lengths")
    result = []
    for raw_path, strength, role in zip(paths, strengths, roles):
        strength = float(strength)
        if not math.isfinite(strength) or not -8.0 <= strength <= 8.0:
            raise ValueError("LoRA strength must be finite and in -8...8")
        if role != "transformer":
            raise ValueError("Core ML FFN export only accepts transformer LoRA assets")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"LoRA must be a regular file: {path}")
        result.append({
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "role": str(role),
            "strength": strength,
        })
    return result


def _load_safetensors(path: Path, np):
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"invalid LoRA header: {path}")
        header_size = struct.unpack("<Q", raw)[0]
        if header_size > 100_000_000:
            raise ValueError(f"LoRA header is unreasonably large: {path}")
        header = json.loads(stream.read(header_size))
        data_offset = 8 + header_size
        memory = stream.read()
    if not isinstance(header, dict):
        raise ValueError(f"invalid LoRA header object: {path}")
    result = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict) or meta.get("dtype") not in {"BF16", "F16", "F32"}:
            continue
        shape = tuple(int(value) for value in meta.get("shape", []))
        start, end = meta.get("data_offsets", [-1, -1])
        count = int(np.prod(shape))
        itemsize = 4 if meta["dtype"] == "F32" else 2
        if start < 0 or end - start != count * itemsize or data_offset + end > data_offset + len(memory):
            raise ValueError(f"invalid LoRA tensor offsets: {name}")
        offset = start
        if meta["dtype"] == "BF16":
            words = np.frombuffer(memory, dtype="<u2", count=count, offset=offset)
            value = (words.astype(np.uint32) << 16).view(np.float32).astype(np.float16)
        else:
            dtype = "<f4" if meta["dtype"] == "F32" else "<f2"
            value = np.frombuffer(memory, dtype=dtype, count=count, offset=offset).copy()
        result[name] = value.reshape(shape)
    return result


def load(paths, strengths, roles, np):
    bundles = []
    for path, strength, role in zip(paths, strengths, roles):
        tensors = _load_safetensors(Path(path).expanduser().resolve(), np)
        pairs = {}
        for raw_name, value in tensors.items():
            stem = raw_name
            suffix = None
            for candidate in (
                ".lora_A.default.weight", ".lora_A.weight", ".lora_down.weight",
                ".lora_B.default.weight", ".lora_B.weight", ".lora_up.weight",
                ".alpha", ".lora_alpha",
            ):
                if stem.endswith(candidate):
                    stem = stem[:-len(candidate)]
                    suffix = candidate
                    break
            if suffix is None:
                continue
            pair = pairs.setdefault(stem, {})
            if "alpha" in suffix:
                pair["alpha"] = value
            elif "A" in suffix or "down" in suffix:
                pair["down"] = value
            else:
                pair["up"] = value
        bundles.append({"pairs": pairs, "strength": float(strength), "role": str(role),
                        "path": str(Path(path).resolve()), "applied": 0})
    return bundles


def _strip_prefix(stem):
    for prefix in ("base_model.model.", "transformer.", "diffusion_model.", "model."):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def targets(stem):
    stem = _strip_prefix(stem)
    result = [stem]
    parts = stem.split(".")
    if len(parts) >= 3 and parts[0] == "double_blocks" and parts[2] in {"img_attn", "txt_attn"}:
        prefix = "transformer_blocks." + parts[1] + ".attn."
        tail = ".".join(parts[3:])
        if parts[2] == "img_attn" and tail == "qkv":
            return [prefix + "to_q", prefix + "to_k", prefix + "to_v"]
        if parts[2] == "txt_attn" and tail == "qkv":
            return [prefix + "add_q_proj", prefix + "add_k_proj", prefix + "add_v_proj"]
    if len(parts) == 3 and parts[0] == "single_blocks":
        prefix = "single_transformer_blocks." + parts[1] + ".attn."
        if parts[2] == "linear1":
            return [prefix + "to_qkv_mlp_proj"]
        if parts[2] == "linear2":
            return [prefix + "to_out"]
    return result


def apply(base_name, base, bundles, np):
    """Apply all matching adapter deltas to a NumPy matrix and return it.

    `base_name` is the native un-fused weight key without ambiguity, for
    example `single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight` or
    `layers.0.feed_forward.w1.weight`.
    """
    bare = base_name[:-len(".weight")] if base_name.endswith(".weight") else base_name
    value = base.astype(np.float32, copy=True)
    applied = 0
    for bundle in bundles:
        if bundle["role"] != "transformer":
            continue
        for stem, pair in bundle["pairs"].items():
            down, up = pair.get("down"), pair.get("up")
            if down is None or up is None:
                continue
            for target in targets(stem):
                if target != bare and not (target + ".weight" == base_name):
                    continue
                if down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1] or down.shape[1] != value.shape[1]:
                    raise ValueError(f"LoRA input geometry does not match {base_name}")
                if up.shape[0] != value.shape[0]:
                    raise ValueError(f"LoRA output geometry does not match {base_name}")
                scale = bundle["strength"]
                alpha = pair.get("alpha")
                if alpha is not None:
                    if alpha.size != 1:
                        raise ValueError(f"LoRA alpha must be scalar: {stem}")
                    scale *= float(alpha.reshape(-1)[0]) / float(down.shape[0])
                value += scale * (up.astype(np.float32) @ down.astype(np.float32))
                bundle["applied"] += 1
                applied += 1
    return value.astype(base.dtype), applied
