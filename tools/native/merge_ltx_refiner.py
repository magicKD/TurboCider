#!/usr/bin/env python3
"""Merge Sana's pinned LTX-2.5 refiner LoRA into the official INT8 model.

The source Transformer is a single 20+ GiB safetensors file.  Loading and
saving the complete state dictionary would temporarily require several copies
of the model, which defeats the 48--60 GiB target.  This tool instead copies
the source checkpoint once and replaces only the 1,660 patched tensor ranges
in-place.  One weight (and, for INT8, its scale) is resident at a time.

The arithmetic intentionally mirrors ComfyUI's runtime patch path:

* a quantized base weight is dequantized to the MPS LoRA compute dtype FP16;
* ``B @ A`` is evaluated in FP32;
* ``0.8 * delta`` is cast to FP16 and added in FP16;
* BF16 weights are cast back to BF16;
* INT8 weights are ConvRot-rotated and row-wise requantized with ComfyUI's
  key-derived stochastic-rounding seed.

The output keeps the source safetensors header and tensor layout byte-for-byte,
and receives an adjacent provenance manifest.  Interrupted merges resume from
an identity-bound state file without loading already committed weights.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import time
from typing import Any, Iterable, Mapping


OFFICIAL_REPOSITORY = "Lightricks/LTX-2.5"
OFFICIAL_REVISION = "bf86adedf518142442575d1ce2e767b7d01c8c76"
OFFICIAL_BASE_NAME = (
    "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors"
)
OFFICIAL_BASE_BYTES = 21_504_034_224
OFFICIAL_BASE_SHA256 = (
    "2edbdb4465cd6c3b532cd67a31ddb38a63e97dcad20be3729675e2a4e8caf92b"
)
OFFICIAL_LORA_NAME = "ltx-2.5-22b-distilled-lora-450-bf16.safetensors"
OFFICIAL_LORA_BYTES = 8_899_889_568
OFFICIAL_LORA_SHA256 = (
    "86370bbf79a9eb4edaa158907e2b48a5188fe4c5dc8ce30c7eb8f2f131a9bbf5"
)
# The published distilled adapter is normally consumed at 0.8, but the
# merger is also used by the identity-bound runtime cache.  Keep the default
# compatible with the audited artifact while making the requested strength an
# explicit part of the merge state and manifest instead of silently baking a
# different value than the caller asked for.
OFFICIAL_STRENGTH = 0.8
OFFICIAL_RAW_TENSORS = 3_320
OFFICIAL_PATCHES = 1_660
OFFICIAL_INT8_PATCHES = 1_344
OFFICIAL_BF16_PATCHES = 316
MERGED_NAME = (
    "ltx-2.5-22b-dev-refiner-lora-0.8-comfy-int8-convrot.safetensors"
)
MANIFEST_SCHEMA = "h3-super-merged-refiner-v1"
STATE_SCHEMA = "h3-super-merged-refiner-state-v1"
ALGORITHM = "comfy-runtime-equivalent-fp16-lora-fp32-mm-convrot-int8-v1"
CHECKPOINT_INTERVAL = 8


@dataclass(frozen=True)
class TensorDescriptor:
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class MergeUpdate:
    target: str
    a_key: str
    b_key: str
    dtype: str
    shape: tuple[int, ...]
    scale_key: str | None
    convrot_groupsize: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path, help="official dev INT8 Transformer")
    parser.add_argument("lora", type=Path, help="official distilled refiner LoRA")
    parser.add_argument("output", type=Path, help="derived INT8 checkpoint path")
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cpu"),
        default="auto",
        help="merge device; auto prefers MPS",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=OFFICIAL_STRENGTH,
        help="LoRA multiplier recorded in the manifest (default: 0.8)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="restart and replace this tool's exact output/partial/state files",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify identity, tensor mapping, quantization metadata, and shapes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="diagnostic: merge only the first N tensors and keep partial state",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def string_to_seed(data: str | bytes) -> int:
    """Bit-for-bit copy of ``comfy.utils.string_to_seed`` (CRC-32)."""

    crc = 0xFFFFFFFF
    for value in data:
        byte = ord(value) if isinstance(value, str) else value
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xEDB88320 if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


def read_safetensors_header(
    path: Path,
) -> tuple[int, dict[str, TensorDescriptor], Mapping[str, str]]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"truncated safetensors header: {path}")
        header_bytes = struct.unpack("<Q", raw_length)[0]
        raw_header = handle.read(header_bytes)
    if len(raw_header) != header_bytes:
        raise ValueError(f"truncated safetensors JSON header: {path}")
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exception:
        raise ValueError(f"invalid safetensors JSON header: {path}") from exception
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not an object: {path}")
    tensors: dict[str, TensorDescriptor] = {}
    for key, value in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(value, dict):
            raise ValueError(f"invalid tensor descriptor for {key}")
        offsets = value.get("data_offsets")
        shape = value.get("shape")
        dtype = value.get("dtype")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(type(item) is int for item in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or not isinstance(shape, list)
            or not all(type(item) is int and item >= 0 for item in shape)
            or not isinstance(dtype, str)
        ):
            raise ValueError(f"invalid tensor descriptor for {key}")
        tensors[key] = TensorDescriptor(
            dtype=dtype,
            shape=tuple(shape),
            start=offsets[0],
            end=offsets[1],
        )
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError("safetensors __metadata__ is not an object")
    return 8 + header_bytes, tensors, metadata


def _pair_stems(keys: Iterable[str]) -> tuple[str, ...]:
    raw = set(keys)
    suffix_a = ".lora_A.weight"
    suffix_b = ".lora_B.weight"
    unsupported = sorted(
        key for key in raw if not key.endswith((suffix_a, suffix_b))
    )
    if unsupported:
        raise ValueError(f"unsupported LoRA tensors: {unsupported[:8]}")
    stems_a = {key[: -len(suffix_a)] for key in raw if key.endswith(suffix_a)}
    stems_b = {key[: -len(suffix_b)] for key in raw if key.endswith(suffix_b)}
    if stems_a != stems_b or not stems_a:
        missing_a = sorted(stems_b - stems_a)
        missing_b = sorted(stems_a - stems_b)
        raise ValueError(
            f"incomplete LoRA A/B pairs: missing_A={missing_a[:8]} "
            f"missing_B={missing_b[:8]}"
        )
    return tuple(sorted(stems_a))


def build_plan(
    base_tensors: Mapping[str, TensorDescriptor],
    lora_tensors: Mapping[str, TensorDescriptor],
    quant_metadata: Mapping[str, Mapping[str, Any]],
) -> tuple[MergeUpdate, ...]:
    if len(lora_tensors) != OFFICIAL_RAW_TENSORS:
        raise ValueError(
            f"official refiner has {len(lora_tensors)} tensors; "
            f"expected {OFFICIAL_RAW_TENSORS}"
        )
    updates: list[MergeUpdate] = []
    for stem in _pair_stems(lora_tensors):
        target = f"model.{stem}.weight"
        a_key = stem + ".lora_A.weight"
        b_key = stem + ".lora_B.weight"
        if target not in base_tensors:
            raise ValueError(f"LoRA target is absent from base model: {target}")
        target_desc = base_tensors[target]
        a_desc = lora_tensors[a_key]
        b_desc = lora_tensors[b_key]
        if len(target_desc.shape) != 2 or len(a_desc.shape) != 2 or len(b_desc.shape) != 2:
            raise ValueError(f"only 2-D LTX refiner weights are supported: {target}")
        output_features, input_features = target_desc.shape
        rank, a_input = a_desc.shape
        b_output, b_rank = b_desc.shape
        if (a_input, b_output, b_rank) != (input_features, output_features, rank):
            raise ValueError(
                f"LoRA shape mismatch for {target}: base={target_desc.shape} "
                f"A={a_desc.shape} B={b_desc.shape}"
            )
        scale_key = None
        groupsize = None
        if target_desc.dtype == "I8":
            layer = target[: -len(".weight")]
            scale_key = layer + ".weight_scale"
            quant_key = layer + ".comfy_quant"
            if scale_key not in base_tensors or quant_key not in base_tensors:
                raise ValueError(f"INT8 target lacks scale/metadata: {target}")
            scale_desc = base_tensors[scale_key]
            if scale_desc.dtype != "F32" or scale_desc.shape != (output_features, 1):
                raise ValueError(
                    f"unexpected INT8 scale for {target}: "
                    f"{scale_desc.dtype} {scale_desc.shape}"
                )
            layer_quant = quant_metadata.get(layer)
            if not isinstance(layer_quant, Mapping):
                raise ValueError(f"missing decoded quantization metadata: {layer}")
            if (
                layer_quant.get("format") != "int8_tensorwise"
                or layer_quant.get("convrot") is not True
                or int(layer_quant.get("convrot_groupsize", 0)) != 256
                or input_features % 256
            ):
                raise ValueError(f"unsupported INT8 ConvRot contract: {layer_quant}")
            groupsize = 256
        elif target_desc.dtype != "BF16":
            raise ValueError(
                f"official refiner target {target} has unsupported dtype "
                f"{target_desc.dtype}"
            )
        updates.append(
            MergeUpdate(
                target=target,
                a_key=a_key,
                b_key=b_key,
                dtype=target_desc.dtype,
                shape=target_desc.shape,
                scale_key=scale_key,
                convrot_groupsize=groupsize,
            )
        )
    updates.sort(key=lambda update: base_tensors[update.target].start)
    counts = {
        "total": len(updates),
        "int8": sum(update.dtype == "I8" for update in updates),
        "bf16": sum(update.dtype == "BF16" for update in updates),
    }
    expected = {
        "total": OFFICIAL_PATCHES,
        "int8": OFFICIAL_INT8_PATCHES,
        "bf16": OFFICIAL_BF16_PATCHES,
    }
    if counts != expected:
        raise ValueError(f"official mapping counts changed: {counts} != {expected}")
    return tuple(updates)


def decode_quant_metadata(
    base_path: Path, base_tensors: Mapping[str, TensorDescriptor]
) -> dict[str, Mapping[str, Any]]:
    from safetensors import safe_open

    decoded: dict[str, Mapping[str, Any]] = {}
    with safe_open(base_path, framework="pt", device="cpu") as checkpoint:
        for key in base_tensors:
            if not key.endswith(".comfy_quant"):
                continue
            raw = bytes(checkpoint.get_tensor(key).tolist())
            try:
                value = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exception:
                raise ValueError(f"invalid comfy_quant tensor: {key}") from exception
            if not isinstance(value, dict):
                raise ValueError(f"comfy_quant is not an object: {key}")
            decoded[key[: -len(".comfy_quant")]] = value
    return decoded


def verify_official_file(
    path: Path, expected_name: str, expected_bytes: int, expected_sha256: str
) -> str:
    path = path.resolve()
    if path.name != expected_name:
        raise ValueError(f"expected {expected_name}, got {path.name}")
    if not path.is_file() or path.stat().st_size != expected_bytes:
        actual = path.stat().st_size if path.is_file() else -1
        raise ValueError(f"official artifact size mismatch: {actual} != {expected_bytes}")
    started = time.perf_counter()
    digest = sha256_file(path)
    print(f"verified {path.name} sha256={digest} in {time.perf_counter() - started:.2f}s")
    if digest != expected_sha256:
        raise ValueError(
            f"official artifact SHA-256 mismatch: {digest} != {expected_sha256}"
        )
    return digest


def _state_identity(
    base: Path, lora: Path, output: Path, strength: float
) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "algorithm": ALGORITHM,
        "base_path": str(base.resolve()),
        "base_sha256": OFFICIAL_BASE_SHA256,
        "lora_path": str(lora.resolve()),
        "lora_sha256": OFFICIAL_LORA_SHA256,
        "output_path": str(output.resolve()),
        "strength": float(strength),
    }


def _load_or_create_state(
    state_path: Path,
    identity: Mapping[str, Any],
    partial_path: Path,
    base_path: Path,
) -> dict[str, Any]:
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for key, value in identity.items():
            if state.get(key) != value:
                raise ValueError(f"merge state identity mismatch for {key}")
        if not partial_path.is_file() or partial_path.stat().st_size != OFFICIAL_BASE_BYTES:
            raise ValueError("merge state exists but its partial checkpoint is invalid")
        completed = state.get("completed")
        if not isinstance(completed, list) or not all(
            isinstance(item, str) for item in completed
        ):
            raise ValueError("merge state has an invalid completed list")
        return state
    if partial_path.exists():
        raise ValueError(
            f"orphan partial checkpoint exists without state: {partial_path}"
        )
    started = time.perf_counter()
    shutil.copyfile(base_path, partial_path)
    if partial_path.stat().st_size != OFFICIAL_BASE_BYTES:
        raise RuntimeError("partial checkpoint copy has the wrong size")
    state = dict(identity)
    state.update({"completed": [], "copied_base_seconds": time.perf_counter() - started})
    write_json_atomic(state_path, state)
    print(
        f"copied base checkpoint to {partial_path.name} in "
        f"{state['copied_base_seconds']:.2f}s"
    )
    return state


def _tensor_bytes(tensor: Any) -> bytes:
    import torch

    value = tensor.detach().contiguous().cpu()
    return value.view(torch.uint8).numpy().tobytes(order="C")


def _write_tensor(
    descriptor: int,
    data_start: int,
    tensor_desc: TensorDescriptor,
    tensor: Any,
    label: str,
) -> None:
    raw = _tensor_bytes(tensor)
    if len(raw) != tensor_desc.nbytes:
        raise RuntimeError(
            f"serialized size mismatch for {label}: {len(raw)} != {tensor_desc.nbytes}"
        )
    written = os.pwrite(descriptor, raw, data_start + tensor_desc.start)
    if written != len(raw):
        raise OSError(f"short pwrite for {label}: {written} != {len(raw)}")


def _select_device(requested: str) -> str:
    import torch

    available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "mps" and not available:
        raise RuntimeError("MPS was requested but is unavailable in this process")
    if requested == "auto":
        return "mps" if available else "cpu"
    return requested


def merge_checkpoint(
    base_path: Path,
    lora_path: Path,
    output_path: Path,
    device_name: str,
    overwrite: bool,
    limit: int | None,
    strength: float,
) -> None:
    import torch
    from safetensors import safe_open
    import comfy_kitchen  # noqa: F401 - registers quantization custom ops
    from comfy_kitchen.backends.eager.quantization import (
        quantize_int8_convrot_weight,
    )

    if output_path.suffix != ".safetensors":
        raise ValueError("LTX merged output must use a .safetensors filename")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    partial_path = output_path.with_name(f".{output_path.name}.partial")
    state_path = output_path.with_name(f".{output_path.name}.merge-state.json")
    if overwrite:
        for path in (output_path, manifest_path, partial_path, state_path):
            path.unlink(missing_ok=True)
    elif output_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"completed output already exists; use --overwrite to rebuild: {output_path}"
        )

    base_data_start, base_tensors, _ = read_safetensors_header(base_path)
    _, lora_tensors, _ = read_safetensors_header(lora_path)
    quant_metadata = decode_quant_metadata(base_path, base_tensors)
    plan = build_plan(base_tensors, lora_tensors, quant_metadata)
    identity = _state_identity(base_path, lora_path, output_path, strength)
    state = _load_or_create_state(
        state_path, identity, partial_path, base_path
    )
    completed = set(state["completed"])
    unknown = completed - {update.target for update in plan}
    if unknown:
        raise ValueError(f"merge state contains unknown targets: {sorted(unknown)[:8]}")

    device_name = _select_device(device_name)
    device = torch.device(device_name)
    print(
        f"merge device={device} updates={len(plan)} resume={len(completed)} "
        f"strength={strength}"
    )
    descriptor = os.open(partial_path, os.O_RDWR)
    run_started = time.perf_counter()
    processed = 0
    try:
        with (
            safe_open(base_path, framework="pt", device="cpu") as base,
            safe_open(lora_path, framework="pt", device="cpu") as lora,
            torch.inference_mode(),
        ):
            for index, update in enumerate(plan, start=1):
                if update.target in completed:
                    continue
                if limit is not None and processed >= limit:
                    break
                started = time.perf_counter()
                seed = string_to_seed(update.target)
                a = lora.get_tensor(update.a_key).to(
                    device=device, dtype=torch.float32
                )
                b = lora.get_tensor(update.b_key).to(
                    device=device, dtype=torch.float32
                )
                delta = torch.mm(b, a)
                del a, b
                if tuple(delta.shape) != update.shape:
                    raise RuntimeError(
                        f"computed LoRA shape mismatch for {update.target}: "
                        f"{tuple(delta.shape)} != {update.shape}"
                    )
                delta = (delta * strength).to(torch.float16)

                if update.dtype == "BF16":
                    weight = base.get_tensor(update.target).to(
                        device=device, dtype=torch.float16
                    )
                    weight.add_(delta)
                    merged = weight.to(torch.bfloat16)
                    _write_tensor(
                        descriptor,
                        base_data_start,
                        base_tensors[update.target],
                        merged,
                        update.target,
                    )
                    del weight, merged
                else:
                    assert update.scale_key is not None
                    assert update.convrot_groupsize is not None
                    qdata = base.get_tensor(update.target).to(device=device)
                    scale = base.get_tensor(update.scale_key).to(device=device)
                    # Comfy first changes the QuantizedTensor logical dtype to
                    # FP16, then dequantizes.  The eager op performs FP32 scale
                    # and inverse ConvRot before the final FP16 cast.
                    weight = torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(
                        qdata, scale, update.convrot_groupsize, 1
                    )
                    del qdata, scale
                    weight.add_(delta)
                    merged_q, merged_scale = quantize_int8_convrot_weight(
                        weight,
                        update.convrot_groupsize,
                        stochastic_rounding=seed,
                    )
                    _write_tensor(
                        descriptor,
                        base_data_start,
                        base_tensors[update.target],
                        merged_q,
                        update.target,
                    )
                    _write_tensor(
                        descriptor,
                        base_data_start,
                        base_tensors[update.scale_key],
                        merged_scale,
                        update.scale_key,
                    )
                    del weight, merged_q, merged_scale
                del delta
                if device.type == "mps":
                    torch.mps.synchronize()
                completed.add(update.target)
                state["completed"] = sorted(completed)
                state["last_target"] = update.target
                state["merge_seconds"] = time.perf_counter() - run_started
                processed += 1
                if processed % CHECKPOINT_INTERVAL == 0:
                    os.fsync(descriptor)
                    write_json_atomic(state_path, state)
                    if device.type == "mps":
                        torch.mps.empty_cache()
                elapsed = time.perf_counter() - started
                total_elapsed = time.perf_counter() - run_started
                remaining = len(plan) - len(completed)
                rate = processed / total_elapsed if total_elapsed > 0 else 0.0
                eta = remaining / rate if rate > 0 else math.inf
                print(
                    f"[{len(completed):4d}/{len(plan)}] {update.dtype:4s} "
                    f"{elapsed:7.2f}s eta={eta / 60:7.1f}m {update.target}",
                    flush=True,
                )
    finally:
        os.fsync(descriptor)
        os.close(descriptor)
        state["completed"] = sorted(completed)
        state["merge_seconds"] = time.perf_counter() - run_started
        write_json_atomic(state_path, state)

    if len(completed) != len(plan):
        print(
            f"partial merge retained: completed={len(completed)}/{len(plan)} "
            f"state={state_path}"
        )
        return

    os.replace(partial_path, output_path)
    output_bytes = output_path.stat().st_size
    if output_bytes != OFFICIAL_BASE_BYTES:
        raise RuntimeError(f"merged checkpoint size changed: {output_bytes}")
    hash_started = time.perf_counter()
    output_sha256 = sha256_file(output_path)
    hash_seconds = time.perf_counter() - hash_started
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "algorithm": ALGORITHM,
        "repository": OFFICIAL_REPOSITORY,
        "revision": OFFICIAL_REVISION,
        "base": {
            "filename": OFFICIAL_BASE_NAME,
            "bytes": OFFICIAL_BASE_BYTES,
            "sha256": OFFICIAL_BASE_SHA256,
        },
        "lora": {
            "filename": OFFICIAL_LORA_NAME,
            "bytes": OFFICIAL_LORA_BYTES,
            "sha256": OFFICIAL_LORA_SHA256,
            "strength": float(strength),
            "raw_tensors": OFFICIAL_RAW_TENSORS,
            "patches": OFFICIAL_PATCHES,
        },
        "mapping": {
            "total": OFFICIAL_PATCHES,
            "int8_convrot": OFFICIAL_INT8_PATCHES,
            "bf16": OFFICIAL_BF16_PATCHES,
            "missing": 0,
        },
        "arithmetic": {
            "lora_matmul": "float32",
            "base_compute": "float16",
            "delta_add": "float16",
            "int8_convrot_groupsize": 256,
            "int8_scale": "per-output-channel float32",
            "int8_rounding": "comfy key-seeded stochastic rounding",
            "bf16_store": "round-to-bfloat16",
        },
        "output": {
            "filename": output_path.name,
            "bytes": output_bytes,
            "sha256": output_sha256,
        },
        "timing": {
            "base_copy_seconds": state.get("copied_base_seconds", 0.0),
            "merge_seconds": state.get("merge_seconds", 0.0),
            "output_hash_seconds": hash_seconds,
        },
    }
    write_json_atomic(manifest_path, manifest)
    state_path.unlink(missing_ok=True)
    print(
        f"merged checkpoint complete: {output_path}\n"
        f"sha256={output_sha256} manifest={manifest_path}"
    )


def main() -> int:
    args = parse_args()
    base = args.base.resolve()
    lora = args.lora.resolve()
    output = args.output.resolve()
    if not math.isfinite(args.strength) or args.strength <= 0.0 or args.strength > 4.0:
        raise ValueError("--strength must be finite and in (0, 4]")
    verify_official_file(
        base, OFFICIAL_BASE_NAME, OFFICIAL_BASE_BYTES, OFFICIAL_BASE_SHA256
    )
    verify_official_file(
        lora, OFFICIAL_LORA_NAME, OFFICIAL_LORA_BYTES, OFFICIAL_LORA_SHA256
    )
    _, base_tensors, _ = read_safetensors_header(base)
    _, lora_tensors, _ = read_safetensors_header(lora)
    quant_metadata = decode_quant_metadata(base, base_tensors)
    plan = build_plan(base_tensors, lora_tensors, quant_metadata)
    print(
        f"validated official mapping: total={len(plan)} "
        f"int8={sum(update.dtype == 'I8' for update in plan)} "
        f"bf16={sum(update.dtype == 'BF16' for update in plan)}"
    )
    if args.check_only:
        return 0
    merge_checkpoint(
        base,
        lora,
        output,
        args.device,
        args.overwrite,
        args.limit,
        args.strength,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
