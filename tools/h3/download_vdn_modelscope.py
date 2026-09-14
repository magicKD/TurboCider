#!/usr/bin/env python3
"""Download the minimal ModelScope VDN-H3 stage-DMD attachment.

The MiniMax H3 base model is intentionally not downloaded here.  TurboCider
already keeps the base checkpoint outside the repository; this tool fetches
only the VDN hybrid-attention branch, its stage metadata, and the optional
Turbo adapter from ModelScope.  Each large file is written directly to the
final tree through a resumable ``.part`` sibling so a second content cache is
not created on a nearly-full Apple Silicon SSD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


MODEL_ID = "OpenVDN/vdn-minimax-h3"
REVISION = "master"
ENDPOINT = "https://modelscope.cn"
STAGE = "stage-dmd-step-250"

FILES = {
    f"{STAGE}/linear_branch/config.json": {
        "bytes": 465,
        "sha256": "decb06ac7e664610f677fb445318502b3c51f9c1b8603a2cdd00b16042be5bc8",
    },
    f"{STAGE}/linear_branch/model.safetensors": {
        "bytes": 4_279_428_112,
        "sha256": "dec6981c7874f5b3bc92d1a02e256b673a3b3499dc1a124714bb3b19da602855",
    },
    f"{STAGE}/adapters/turbo/adapter_spec.json": {
        "bytes": 22_264,
        "sha256": "627968f670747c29cd7a0d3f8c75166e501d70f9e829b5cd3242a8f14cefbc18",
    },
    f"{STAGE}/adapters/turbo/adapter_model.safetensors": {
        "bytes": 851_452_696,
        "sha256": "24fc93c82fe84dc45d0627f4e72c637bc387d282ba18f60ed3b7f8c81089392c",
    },
    f"{STAGE}/model_spec.json": {
        "bytes": 25_705,
        "sha256": "4171f4384e952f1f73467981a893440c03298af4956b947e2c8a857ba9f5a62b",
    },
    f"{STAGE}/metadata.json": {
        "bytes": 463,
        "sha256": "54054ceb1c91b3fdf7fa0278e4a8841c127e8cf666b5e240d69613661f9d3e9e",
    },
}

BRANCH_TENSORS = {
    "attn.linear_attention.alpha.A_log": [56],
    "attn.linear_attention.alpha.down.weight": [128, 5376],
    "attn.linear_attention.alpha.dt_bias": [7168],
    "attn.linear_attention.alpha.up.weight": [7168, 128],
    "attn.linear_attention.beta_proj.weight": [56, 5376],
    "attn.linear_attention.norm.weight": [128],
    "attn.linear_attention.output_gate.down.weight": [128, 5376],
    "attn.linear_attention.output_gate.up.bias": [7168],
    "attn.linear_attention.output_gate.up.weight": [7168, 128],
    "attn.linear_attention.short_conv.k_sp.weight": [7168, 1, 5, 5],
    "attn.linear_attention.short_conv.k_tm.weight": [7168, 1, 5],
    "attn.linear_attention.short_conv.v_sp.weight": [7168, 1, 5, 5],
    "attn.linear_attention.short_conv.v_tm.weight": [7168, 1, 5],
    "attn.softmax_gate.up.bias": [56],
    "attn.softmax_gate.up.weight": [56, 5376],
    "attn.to_out_linear.weight": [5376, 7168],
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot parse JSON object {path}: {error}") from error
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read and structurally validate a safetensors header without loading data."""
    try:
        with path.open("rb") as stream:
            raw_length = stream.read(8)
            require(len(raw_length) == 8, f"truncated safetensors prefix: {path}")
            header_length = struct.unpack("<Q", raw_length)[0]
            require(2 <= header_length <= 64 * 1024 * 1024,
                    f"unsafe safetensors header length in {path}: {header_length}")
            raw_header = stream.read(header_length)
    except OSError as error:
        raise RuntimeError(f"cannot read safetensors header {path}: {error}") from error
    require(len(raw_header) == header_length, f"truncated safetensors header: {path}")
    try:
        header = json.loads(raw_header)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid safetensors JSON header {path}: {error}") from error
    require(isinstance(header, dict), f"safetensors header must be an object: {path}")
    data_bytes = path.stat().st_size - 8 - header_length
    intervals: list[tuple[int, int, str]] = []
    for name, record in header.items():
        if name == "__metadata__":
            continue
        require(isinstance(name, str) and name, f"invalid safetensors key in {path}")
        require(isinstance(record, dict), f"invalid tensor record {name} in {path}")
        require(isinstance(record.get("dtype"), str), f"missing dtype for {name}")
        require(isinstance(record.get("shape"), list) and
                all(isinstance(dim, int) and dim >= 0 for dim in record["shape"]),
                f"invalid shape for {name}")
        offsets = record.get("data_offsets")
        require(isinstance(offsets, list) and len(offsets) == 2 and
                all(isinstance(offset, int) for offset in offsets),
                f"invalid data offsets for {name}")
        start, end = offsets
        require(0 <= start <= end <= data_bytes,
                f"out-of-range data offsets for {name}")
        intervals.append((start, end, name))
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        require(previous[1] <= current[0],
                f"overlapping safetensors data for {previous[2]} and {current[2]}")
    return header


def tensor_records(header: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {name: record for name, record in header.items() if name != "__metadata__"}


def validate_tensor(record: dict[str, Any], name: str, shape: list[int]) -> None:
    require(record.get("dtype") == "BF16", f"VDN tensor must be BF16: {name}")
    require(record.get("shape") == shape,
            f"VDN tensor shape mismatch for {name}: {record.get('shape')} != {shape}")


def adapter_module_shapes(name: str) -> tuple[list[int], list[int]]:
    if name == "norm_out.linear":
        return [16, 2688], [10752, 16]
    if name.endswith(".adaln_proj.linear"):
        return [16, 2688], [96768, 16]
    rank = 64
    if name.endswith(("attn.to_q", "attn.to_k", "attn.to_v",
                      "attn.orig.to_q", "attn.orig.to_k", "attn.orig.to_v")):
        return [rank, 5376], [7168, rank]
    if name.endswith(("attn.to_out.0", "attn.orig.to_out.0")):
        return [rank, 7168], [5376, rank]
    if name.endswith("ff.net.0.proj"):
        return [rank, 5376], [28672, rank]
    if name.endswith("ff.net.2"):
        return [rank, 14336], [5376, rank]
    raise RuntimeError(f"unsupported VDN Turbo adapter target: {name}")


def validate_branch_header(path: Path) -> None:
    tensors = tensor_records(read_safetensors_header(path))
    expected = {
        f"transformer_blocks.{block}.{suffix}": shape
        for block in range(50)
        for suffix, shape in BRANCH_TENSORS.items()
    }
    require(len(expected) == 800, "internal VDN branch tensor contract is inconsistent")
    require(set(tensors) == set(expected),
            f"VDN branch keys differ: missing={sorted(set(expected) - set(tensors))[:5]} "
            f"extra={sorted(set(tensors) - set(expected))[:5]}")
    for name, shape in expected.items():
        validate_tensor(tensors[name], name, shape)


def validate_adapter_header(path: Path, adapter_spec: dict[str, Any]) -> None:
    config = adapter_spec.get("config")
    require(isinstance(config, dict), "VDN Turbo adapter spec has no config object")
    targets = config.get("targets")
    require(isinstance(targets, list) and len(targets) == 363 and
            all(isinstance(target, str) and target for target in targets),
            "VDN Turbo adapter must declare 363 exact targets")
    require(len(set(targets)) == len(targets), "VDN Turbo adapter targets are duplicated")
    tensors = tensor_records(read_safetensors_header(path))
    expected: dict[str, list[int]] = {}
    for module in targets:
        a_shape, b_shape = adapter_module_shapes(module)
        expected[f"{module}.lora_A.turbo.weight"] = a_shape
        expected[f"{module}.lora_B.turbo.weight"] = b_shape
    require(len(expected) == 726, "internal VDN Turbo tensor contract is inconsistent")
    require(set(tensors) == set(expected),
            f"VDN Turbo keys differ: missing={sorted(set(expected) - set(tensors))[:5]} "
            f"extra={sorted(set(tensors) - set(expected))[:5]}")
    for name, shape in expected.items():
        validate_tensor(tensors[name], name, shape)


def validate_stage_contract(root: Path) -> None:
    stage = root / STAGE
    branch_config = read_json_object(stage / "linear_branch/config.json")
    expected_transform = {
        "type": "hybrid_attention",
        "version": 2,
        "config": {
            "anchor_frames": "both",
            "enable_softmax_gate": True,
            "linear_attention": {
                "a_fp32": True,
                "bridge": "alpha",
                "delta_rule": "vdn_solve",
                "enable_text_state": True,
                "linear_head_dim": 128,
                "short_conv": {"targets": ["k", "v"]},
            },
            "softmax_attention": {"chunk": 5, "radius": 1},
        },
    }
    require(branch_config == expected_transform,
            "VDN linear branch config is not the supported stage-DMD v2 contract")

    model_spec = read_json_object(stage / "model_spec.json")
    require(model_spec.get("format_version") == 2, "VDN model_spec format must be 2")
    require(model_spec.get("transforms") == [expected_transform],
            "VDN model_spec transform differs from linear_branch/config.json")
    base = model_spec.get("base")
    require(isinstance(base, dict) and
            base.get("class_name") == "MiniMaxH3Transformer3DModel" and
            base.get("library") == "diffusers" and
            base.get("config_hash") ==
                "fa5c3b3b3bc9f2e604f7a790d5a67369b975daeea3f0db7a6ceadbd5315cd3d0",
            "VDN stage is not attached to the pinned MiniMax H3 FL2VA base contract")

    adapter_spec = read_json_object(stage / "adapters/turbo/adapter_spec.json")
    adapter = adapter_spec.get("config")
    require(adapter_spec.get("type") == "lora" and adapter_spec.get("version") == 1 and
            isinstance(adapter, dict) and adapter.get("name") == "turbo" and
            adapter.get("family") == "larryvrh_v4_step600_ema" and
            adapter.get("rank") == 64 and adapter.get("alpha") == 64 and
            adapter.get("exact_targets") is True,
            "VDN stage Turbo adapter is not larryvrh_v4_step600_ema")
    require(isinstance(adapter.get("rank_pattern"), dict) and
            isinstance(adapter.get("alpha_pattern"), dict) and
            len(adapter["rank_pattern"]) == 51 and
            adapter["rank_pattern"] == adapter["alpha_pattern"] and
            set(adapter["rank_pattern"].values()) == {16},
            "VDN Turbo AdaLN/norm rank and alpha patterns are unsupported")

    metadata = read_json_object(stage / "metadata.json")
    details = metadata.get("metadata")
    require(metadata.get("checkpoint_format_version") == 2 and
            metadata.get("kind") == "weights" and
            metadata.get("weights_dtype") == "bfloat16" and
            isinstance(details, dict) and details.get("stage") == "dmd" and
            details.get("step") == 250 and details.get("real_score") == "vdn" and
            details.get("video_shift") == 12.0 and details.get("audio_shift") == 3.0,
            "VDN stage-DMD metadata is unsupported")

    validate_branch_header(stage / "linear_branch/model.safetensors")
    validate_adapter_header(stage / "adapters/turbo/adapter_model.safetensors",
                            adapter_spec)


def validate_download_manifest(root: Path) -> None:
    manifest = read_json_object(root / "modelscope_download.json")
    require(manifest.get("schema") == "turbocider-modelscope-vdn-h3-v1" and
            manifest.get("repository") == MODEL_ID and
            manifest.get("revision") == REVISION and
            manifest.get("stage") == STAGE and
            manifest.get("endpoint") == ENDPOINT and
            manifest.get("files") == FILES,
            "VDN ModelScope download manifest is inconsistent")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, target: Path, *, retries: int = 4) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    for attempt in range(1, retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", response.getcode())
                if offset and status != 206:
                    offset = 0
                    part.unlink(missing_ok=True)
                mode = "ab" if offset else "wb"
                length = int(response.headers.get("Content-Length", "0"))
                total = offset + length if length else 0
                downloaded = offset
                last_report = time.monotonic()
                with part.open(mode) as stream:
                    while True:
                        chunk = response.read(16 * 1024 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 10:
                            suffix = f"/{total / 2**30:.2f} GiB" if total else ""
                            print(f"  {target.name}: {downloaded / 2**30:.2f} GiB{suffix}", flush=True)
                            last_report = now
                if total and downloaded != total:
                    raise IOError(f"short response ({downloaded} != {total} bytes)")
            part.replace(target)
            return
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            print(f"[retry {attempt}/{retries}] {target.name}: {error}", flush=True)
            if attempt == retries:
                raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="final stage root, e.g. models/VDN-H3-ModelScope")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--min-free-gib", type=float, default=4.0)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.revision != REVISION:
        raise SystemExit(f"this pinned VDN asset only accepts revision {REVISION!r}")
    root = args.root.expanduser().resolve()
    needed = sum(int(item["bytes"]) for path, item in FILES.items()
                 if not (root / path).is_file())
    free = shutil.disk_usage(root.parent).free / 2**30
    if not args.verify_only and free < needed / 2**30 + args.min_free_gib:
        raise RuntimeError(
            f"VDN attachment needs {needed / 2**30:.2f} GiB and only {free:.2f} GiB is free")
    base = f"{ENDPOINT}/models/{MODEL_ID}/resolve/{args.revision}"
    for relative, record in FILES.items():
        target = root / relative
        if target.is_file() and target.stat().st_size == int(record["bytes"]):
            print(f"[skip] {relative}", flush=True)
        elif args.verify_only:
            raise RuntimeError(f"missing or wrong-sized VDN file: {relative}")
        else:
            print(f"[download] {relative}", flush=True)
            download_file(f"{base}/{relative}?download=true", target)
        if target.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"size mismatch for {relative}")
        actual_sha256 = sha256_file(target)
        print(f"[verify] {relative} sha256={actual_sha256}", flush=True)
        if actual_sha256 != record["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {relative}")
    manifest = {
        "schema": "turbocider-modelscope-vdn-h3-v1",
        "repository": MODEL_ID,
        "revision": args.revision,
        "stage": STAGE,
        "base_model_required": "MiniMax H3 FL2VA compatible checkpoint",
        "files": FILES,
        "endpoint": ENDPOINT,
    }
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "modelscope_download.json"
    if not args.verify_only:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validate_download_manifest(root)
    validate_stage_contract(root)
    print(f"[ready] {root}", flush=True)


if __name__ == "__main__":
    main()
