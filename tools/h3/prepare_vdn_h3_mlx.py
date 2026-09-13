#!/usr/bin/env python3
"""Prepare the ModelScope VDN-H3 base/branch artifact for TurboCider.

The ordinary FL2VA transformer is converted with the same checked-in MLX
affine INT6/g64 converter used by FastH3, but with the six-step VDN AdaLN
ladder.  The OpenVDN stage-DMD branch and its ``larryvrh_v4_step600_ema``
adapter are kept as separate, provenance-bound assets.  By default they are
linked rather than copied so preparing an artifact never creates another 5 GB
branch copy on a nearly-full SSD.

This script deliberately refuses the merged LightX2V/Turbo checkpoint and the
pruned Comfy ConvRot checkpoint.  Those checkpoints do not contain the full
2688-dimensional time embedding expected by the VDN adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
VDN_DOWNLOADER = ROOT / "tools/h3/download_vdn_modelscope.py"
BASE_DOWNLOADER = ROOT / "tools/h3/download_h3_fl2va_modelscope.py"
FASTVIDEO_BOOTSTRAP = ROOT / "tools/h3/run_fastvideo_mlx_converter.py"
H3_WEIGHTS_FILENAME = "mlx_h3_dit.safetensors"
H3_MANIFEST_FILENAME = "mlx_h3_dit.json"
STAGE = "stage-dmd-step-250"
STEPS = 6


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def normalized_h3_config(config: dict[str, Any]) -> dict[str, Any]:
    """Translate MiniMaxAI's release field names to the MLX manifest schema."""
    result = dict(config)
    aliases = {
        "num_refiner_layers": "token_refiner_num_layers",
        "ffn_dim": "ffn_hidden_size",
        "in_channels": "latents_dim",
        "audio_in_channels": "audio_latents_dim",
        "freq_dim": "timestep_input_dim",
        "rope_freq_dim": "rope_inv_freq_len",
    }
    for target, source in aliases.items():
        if target not in result and source in result:
            result[target] = result[source]
    result.setdefault("rope_theta", 10_000.0)
    required = {
        "hidden_size", "num_layers", "num_refiner_layers",
        "num_attention_heads", "attention_head_dim", "ffn_dim",
        "in_channels", "audio_in_channels", "patch_size", "text_dim",
        "freq_dim", "time_embed_dim", "rope_freq_dim", "rope_theta",
        "norm_eps", "qk_norm_eps", "final_norm_eps",
    }
    require(required <= result.keys(),
            f"MiniMax H3 config is missing {sorted(required - result.keys())}")
    return result


def rewritten_h3_key(raw_key: str) -> str:
    key = raw_key
    prefix_rewrites = (
        ("token_refiner.blocks.", "token_refiner.refiner_blocks."),
        ("blocks.", "transformer_blocks."),
        ("final_layer.adaln_proj.linear", "norm_out.linear"),
        ("final_layer.video_out", "proj_out"),
        ("final_layer.audio_out", "audio_proj_out"),
        ("final_layer.norm", "norm_out.norm"),
        ("video_patch_proj", "proj_in"),
        ("audio_patch_proj", "audio_proj_in"),
        ("condition_proj", "context_embedder"),
        ("time_embedder.proj_in", "time_embedder.linear_1"),
        ("time_embedder.proj_out", "time_embedder.linear_2"),
    )
    for old, new in prefix_rewrites:
        if old.endswith("."):
            matches = key.startswith(old)
        else:
            matches = key == old or key.startswith(old + ".")
        if matches:
            key = new + key[len(old):]
            break
    key = key.replace(".attn.q_norm", ".attn.norm_q")
    key = key.replace(".attn.k_norm", ".attn.norm_k")
    key = key.replace(".attn.out_proj", ".attn.to_out.0")
    key = key.replace(".mlp.fc2", ".ff.net.2")
    return key


def mapped_h3_key_names(raw_key: str) -> tuple[str, ...]:
    """Return the exact MLX names produced for one MiniMaxAI source key."""
    if raw_key == "rope.inv_freq":
        return ()
    key = rewritten_h3_key(raw_key)
    if key.endswith(".attn.qkv_proj.weight"):
        stem = key[:-len("qkv_proj.weight")]
        return (stem + "to_q.weight", stem + "to_k.weight", stem + "to_v.weight")
    if key.endswith(".mlp.fc1.weight"):
        stem = key[:-len(".mlp.fc1.weight")]
        return (stem + ".ff.net.0.proj.weight",)
    return (key,)


def mapped_h3_tensor_entries(raw_key: str, source: Any, *, num_heads: int,
                             head_dim: int, concatenate: Any):
    """Map a source tensor using operations shared by NumPy and MLX arrays."""
    names = mapped_h3_key_names(raw_key)
    if not names:
        return []
    key = rewritten_h3_key(raw_key)
    if key.endswith(".attn.qkv_proj.weight"):
        require(source.ndim == 2 and source.shape[0] == 3 * num_heads * head_dim,
                f"unexpected H3 fused QKV shape for {raw_key}: {source.shape}")
        grouped = source.reshape((num_heads, 3, head_dim, source.shape[1]))
        return [
            (names[0], grouped[:, 0, :, :].reshape(
                (num_heads * head_dim, source.shape[1]))),
            (names[1], grouped[:, 1, :, :].reshape(
                (num_heads * head_dim, source.shape[1]))),
            (names[2], grouped[:, 2, :, :].reshape(
                (num_heads * head_dim, source.shape[1]))),
        ]
    if key.endswith(".mlp.fc1.weight"):
        require(source.ndim == 2 and source.shape[0] % 2 == 0,
                f"unexpected H3 fused FFN shape for {raw_key}: {source.shape}")
        half = source.shape[0] // 2
        return [(names[0], concatenate([source[half:], source[:half]], axis=0))]
    return [(names[0], source)]


def expected_mlx_base_keys(config: dict[str, Any]) -> set[str]:
    expected = {
        "proj_in.weight", "proj_in.bias",
        "audio_proj_in.weight", "audio_proj_in.bias",
        "context_embedder.weight", "context_embedder.bias",
        "time_embedder.linear_1.weight", "time_embedder.linear_1.bias",
        "time_embedder.linear_2.weight", "time_embedder.linear_2.bias",
        "token_refiner.final_norm.weight",
        "norm_out.linear.weight", "norm_out.linear.bias",
        "norm_out.norm.weight",
        "proj_out.weight", "proj_out.bias",
        "audio_proj_out.weight", "audio_proj_out.bias",
    }
    common = {
        "norm1.weight", "norm2.weight",
        "attn.norm_q.weight", "attn.norm_k.weight",
        "attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight",
        "attn.to_out.0.weight",
        "ff.net.0.proj.weight", "ff.net.2.weight",
    }
    for index in range(int(config["num_refiner_layers"])):
        expected.update(
            f"token_refiner.refiner_blocks.{index}.{suffix}" for suffix in common
        )
    block = common | {"adaln_proj.linear.weight", "adaln_proj.linear.bias"}
    for index in range(int(config["num_layers"])):
        expected.update(f"transformer_blocks.{index}.{suffix}" for suffix in block)
    return expected


def validate_source_mapping(weight_map: dict[str, Any], config: dict[str, Any]) -> set[str]:
    mapped = [name for raw_key in weight_map for name in mapped_h3_key_names(raw_key)]
    require(len(mapped) == len(set(mapped)), "ordinary FL2VA source mapping has key collisions")
    actual = set(mapped)
    expected = expected_mlx_base_keys(config)
    require(actual == expected,
            f"ordinary FL2VA source mapping differs: "
            f"missing={sorted(expected - actual)[:8]} extra={sorted(actual - expected)[:8]}")
    return actual


def adapter_target_for_key(key: str, targets: set[str]) -> str | None:
    if not key.endswith(".weight"):
        return None
    module = key[:-len(".weight")]
    if module.startswith("transformer_blocks.") and ".attn." in module:
        candidate = module.replace(".attn.", ".attn.orig.", 1)
        if candidate in targets:
            module = candidate
    return module if module in targets else None


def validate_adapter_target_coverage(mapped_keys: set[str], targets: list[str]) -> None:
    target_set = set(targets)
    require(len(target_set) == len(targets), "VDN adapter targets are duplicated")
    mapped_targets = [adapter_target_for_key(key, target_set) for key in mapped_keys]
    mapped_targets = [target for target in mapped_targets if target is not None]
    require(len(mapped_targets) == len(set(mapped_targets)),
            "multiple ordinary FL2VA weights map to one VDN adapter target")
    actual = set(mapped_targets)
    require(actual == target_set,
            f"VDN adapter targets differ: missing={sorted(target_set - actual)[:8]} "
            f"extra={sorted(actual - target_set)[:8]}")


def safetensors_header(path: Path) -> dict[str, Any]:
    import struct

    with path.open("rb") as stream:
        raw = stream.read(8)
        require(len(raw) == 8, f"truncated safetensors prefix: {path}")
        length = struct.unpack("<Q", raw)[0]
        require(0 < length <= 64 * 1024 * 1024,
                f"unsafe safetensors header length in {path}: {length}")
        payload = stream.read(length)
    require(len(payload) == length, f"truncated safetensors header: {path}")
    value = json.loads(payload)
    require(isinstance(value, dict), f"safetensors header must be an object: {path}")
    return value


def validate_base(base: Path) -> dict[str, Any]:
    """Validate an ordinary ModelScope FL2VA transformer without loading data."""
    require(base.is_absolute() and base.is_dir(),
            f"VDN base must be an absolute transformer directory: {base}")
    require((base / "config.json").is_file() and
            (base / "model.safetensors.index.json").is_file(),
            "VDN base is missing config.json or model.safetensors.index.json")
    config = normalized_h3_config(read_json(base / "config.json"))
    require(config.get("_class_name") == "MiniMaxH3DiTModel" and
            int(config.get("num_layers", 0)) == 50 and
            int(config.get("time_embed_dim", 0)) == 2688,
            "VDN requires the ordinary 50-block, 2688-dimensional FL2VA base")
    index = read_json(base / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    require(isinstance(weight_map, dict) and weight_map,
            "VDN base index has no weight_map")
    mapped_keys = validate_source_mapping(weight_map, config)
    shards = sorted(set(weight_map.values()))
    require(len(shards) == 13 and all((base / name).is_file() for name in shards),
            "VDN base must contain all 13 ordinary FL2VA shards")
    # The two unsupported local formats have distinctive markers.  Rejecting
    # them by content, not just by path spelling, prevents accidental misuse.
    first = base / shards[0]
    header = safetensors_header(first)
    keys = set(header) - {"__metadata__"}
    require("adaln_t_table" not in keys,
            "pruned ConvRot H3 checkpoint is not a VDN base")
    require(not (base / "h3-turbo-merge-manifest.json").exists(),
            "merged LightX2V/Turbo checkpoint is not a VDN base")
    require(any(key.startswith("time_embedder.") for key in keys),
            "VDN base must retain time_embedder tensors for adapter baking")
    provenance = base.parent.parent / "modelscope_download.json"
    require(provenance.is_file(),
            "VDN base must be produced by download_h3_fl2va_modelscope.py")
    base_manifest = read_json(provenance)
    require(base_manifest.get("repository") == "MiniMax/MiniMax-H3" and
            base_manifest.get("partition") == "FL2VA",
            "VDN base provenance is not the pinned ModelScope FL2VA snapshot")
    return {
        "config_sha256": sha256_file(base / "config.json"),
        "index_sha256": sha256_file(base / "model.safetensors.index.json"),
        "provenance_sha256": sha256_file(provenance),
        "repository": base_manifest["repository"],
        "revision": base_manifest["revision"],
        "mapped_keys": mapped_keys,
    }


def validate_stage(root: Path) -> dict[str, Any]:
    vdn = load_module(VDN_DOWNLOADER, "download_vdn_modelscope_for_prepare")
    vdn.validate_download_manifest(root)
    vdn.validate_stage_contract(root)
    stage = root / STAGE
    return {
        "repository": vdn.MODEL_ID,
        "revision": vdn.REVISION,
        "stage": STAGE,
        "branch_sha256": vdn.FILES[f"{STAGE}/linear_branch/model.safetensors"]["sha256"],
        "adapter_sha256": vdn.FILES[f"{STAGE}/adapters/turbo/adapter_model.safetensors"]["sha256"],
        "branch_bytes": vdn.FILES[f"{STAGE}/linear_branch/model.safetensors"]["bytes"],
        "adapter_bytes": vdn.FILES[f"{STAGE}/adapters/turbo/adapter_model.safetensors"]["bytes"],
        "stage_manifest_sha256": sha256_file(root / "modelscope_download.json"),
        "branch_config_sha256": sha256_file(stage / "linear_branch/config.json"),
        "adapter_spec_sha256": sha256_file(stage / "adapters/turbo/adapter_spec.json"),
    }


def six_step_adaln_timesteps() -> Any:
    """Use FastVideo's own scheduler arithmetic, but with six steps."""
    if not FASTVIDEO_BOOTSTRAP.is_file():
        raise FileNotFoundError(f"FastVideo bootstrap is missing: {FASTVIDEO_BOOTSTRAP}")
    bootstrap = load_module(FASTVIDEO_BOOTSTRAP, "turbocider_fastvideo_bootstrap")
    bootstrap.install_fastvideo_namespace()
    import numpy as np
    from fastvideo.mlx_runtime.minimax_h3 import minimax_h3_sigmas

    video = 1.0 - minimax_h3_sigmas(12.0, STEPS)[:-1]
    audio = 1.0 - minimax_h3_sigmas(3.0, STEPS)[:-1]
    return np.unique(np.concatenate([video, audio, [1.0]])).astype(np.float32)


def link_asset(source: Path, target: Path, copy_assets: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and target.resolve() == source.resolve():
            return
        raise RuntimeError(f"refusing to overwrite existing VDN asset: {target}")
    if copy_assets:
        shutil.copy2(source, target)
    else:
        target.symlink_to(os.path.relpath(source, target.parent))


def stage_assets(stage_root: Path, output_root: Path, copy_assets: bool) -> dict[str, str]:
    source_branch = stage_root / STAGE / "linear_branch/model.safetensors"
    source_branch_config = stage_root / STAGE / "linear_branch/config.json"
    source_adapter = stage_root / STAGE / "adapters/turbo/adapter_model.safetensors"
    source_adapter_spec = stage_root / STAGE / "adapters/turbo/adapter_spec.json"
    asset_root = output_root / "vdn-assets"
    link_asset(source_branch, asset_root / "linear_branch/model.safetensors", copy_assets)
    link_asset(source_branch_config, asset_root / "linear_branch/config.json", copy_assets)
    link_asset(source_adapter, asset_root / "adapters/turbo/adapter_model.safetensors", copy_assets)
    link_asset(source_adapter_spec, asset_root / "adapters/turbo/adapter_spec.json", copy_assets)
    return {
        "linear_branch": "../vdn-assets/linear_branch/model.safetensors",
        "linear_branch_config": "../vdn-assets/linear_branch/config.json",
        "turbo_adapter": "../vdn-assets/adapters/turbo/adapter_model.safetensors",
        "turbo_adapter_spec": "../vdn-assets/adapters/turbo/adapter_spec.json",
    }


def load_merged_vdn_dit(base: Path, adapter_path: Path, adapter_spec_path: Path,
                        quantization: Any, cache_timesteps: Any):
    """Stream the ordinary base, merge the stage adapter, and bake AdaLN.

    This is a small build-time specialization of FastVideo's public H3 MLX
    converter.  It deliberately uses that module's tensor layout, quantizer,
    and AdaLN functions so the resulting base artifact stays byte-contract
    compatible with TurboCider's existing C++ loader.  The runtime has no
    dependency on FastVideo or on this function.
    """
    import numpy as np
    import mlx.core as mx
    from fastvideo.mlx_runtime import minimax_h3 as h3

    config = normalized_h3_config(read_json(base / "config.json"))
    num_blocks = int(config["num_layers"])
    num_refiner = int(config["num_refiner_layers"])
    cast_dtype = mx.bfloat16
    weights: dict[str, Any] = {}
    blocks: list[dict[str, Any] | None] = [None] * num_blocks
    refiner: list[dict[str, Any] | None] = [None] * num_refiner
    adapter_spec = read_json(adapter_spec_path).get("config")
    require(isinstance(adapter_spec, dict), "VDN adapter spec has no config object")
    targets = adapter_spec.get("targets")
    require(isinstance(targets, list) and len(targets) == 363,
            "VDN adapter must declare 363 exact targets")
    target_set = set(targets)
    require(len(target_set) == len(targets), "VDN adapter targets are duplicated")
    rank_pattern = adapter_spec.get("rank_pattern") or {}
    alpha_pattern = adapter_spec.get("alpha_pattern") or {}
    default_rank = int(adapter_spec.get("rank", 0))
    default_alpha = float(adapter_spec.get("alpha", 0))
    require(default_rank > 0 and default_alpha > 0,
            "VDN adapter rank/alpha must be positive")
    adapter_arrays = mx.load(str(adapter_path))
    require(isinstance(adapter_arrays, dict), "VDN adapter did not load as a tensor map")
    consumed_targets: set[str] = set()

    def mapped_entries(raw_key: str, source: Any):
        """Yield FastVideo/diffusers-layout entries for one ModelScope tensor."""
        return mapped_h3_tensor_entries(
            raw_key, source,
            num_heads=int(config["num_attention_heads"]),
            head_dim=int(config["attention_head_dim"]),
            concatenate=mx.concatenate,
        )

    def assign(key: str, value: Any) -> None:
        if key.startswith("transformer_blocks."):
            _, index_string, sub = key.split(".", 2)
            index = int(index_string)
            if index < num_blocks:
                if blocks[index] is None:
                    blocks[index] = {}
                assert blocks[index] is not None
                blocks[index][sub] = value
        elif key.startswith("token_refiner.refiner_blocks."):
            _, _, index_string, sub = key.split(".", 3)
            index = int(index_string)
            if refiner[index] is None:
                refiner[index] = {}
            assert refiner[index] is not None
            refiner[index][sub] = value
        else:
            weights[key] = value

    def load_array(source: Any, dtype: Any):
        value = source.astype(dtype) if source.dtype != dtype else source
        mx.eval(value)
        return value

    def merge_lora(key: str, base_value: Any):
        if not key.endswith(".weight"):
            return base_value
        adapter_module = adapter_target_for_key(key, target_set)
        if adapter_module is None:
            return base_value
        a_key = f"{adapter_module}.lora_A.turbo.weight"
        b_key = f"{adapter_module}.lora_B.turbo.weight"
        require(a_key in adapter_arrays and b_key in adapter_arrays,
                f"missing VDN adapter pair for {adapter_module}")
        rank = int(rank_pattern.get(adapter_module, default_rank))
        alpha = float(alpha_pattern.get(adapter_module, default_alpha))
        a = adapter_arrays[a_key]
        b = adapter_arrays[b_key]
        require(a.ndim == 2 and b.ndim == 2 and
                int(a.shape[0]) == rank and int(b.shape[1]) == rank and
                int(b.shape[0]) == int(base_value.shape[0]) and
                int(a.shape[1]) == int(base_value.shape[1]),
                f"VDN adapter shape does not match {adapter_module}")
        # The released adapter currently has alpha == rank everywhere.  Keep
        # the general factor explicit so a changed upstream spec cannot be
        # interpreted as an unscaled update.
        factor = np.float32(alpha / rank)
        merged = base_value.astype(mx.float32) + factor * mx.matmul(
            b.astype(mx.float32), a.astype(mx.float32))
        merged = merged.astype(base_value.dtype)
        mx.eval(merged)
        consumed_targets.add(adapter_module)
        return merged

    cache_timesteps = np.unique(np.asarray(cache_timesteps, dtype=np.float32))
    pending_adaln: dict[int, dict[str, Any]] = {}
    cached_block_tables: list[tuple[Any, ...] | None] = [None] * num_blocks

    # The time embedder is small and needed before the first huge AdaLN
    # projection can be merged and immediately collapsed into schedule tables.
    for shard in h3._safetensors_shards(base):
        shard_arrays = mx.load(str(shard))
        for raw_key, source in shard_arrays.items():
            for key, value in mapped_entries(raw_key, source):
                if key.startswith("time_embedder."):
                    assign(key, load_array(value, mx.float32))
        del shard_arrays
    required_time_keys = {
        "time_embedder.linear_1.weight", "time_embedder.linear_1.bias",
        "time_embedder.linear_2.weight", "time_embedder.linear_2.bias",
    }
    require(required_time_keys <= weights.keys(),
            "ordinary FL2VA base is missing time-embedder weights")
    timestep_rows = mx.array(cache_timesteps)
    time_frequency = h3.timestep_embedding(
        timestep_rows, int(config["freq_dim"])).astype(
            h3.weight_dtype(weights["time_embedder.linear_1.weight"]))
    temb = h3.linear(
        time_frequency,
        weights["time_embedder.linear_1.weight"],
        weights["time_embedder.linear_1.bias"],
    )
    temb = h3.linear(
        h3.silu(temb),
        weights["time_embedder.linear_2.weight"],
        weights["time_embedder.linear_2.bias"],
    )
    mx.eval(temb)

    for shard in h3._safetensors_shards(base):
        shard_arrays = mx.load(str(shard))
        for raw_key, source in shard_arrays.items():
            if raw_key.startswith("time_embedder.") or raw_key.startswith("rope."):
                continue
            for key, mapped in mapped_entries(raw_key, source):
                if key.startswith("transformer_blocks."):
                    index = int(key.split(".")[1])
                    if index >= num_blocks:
                        continue
                keep_fp32 = key.split(".", 1)[0] in h3.FP32_MODULE_PREFIXES
                target_dtype = mx.float32 if keep_fp32 else cast_dtype
                array = merge_lora(key, load_array(mapped, target_dtype))
                if ".adaln_proj.linear." in key:
                    _, index_string, sub = key.split(".", 2)
                    index = int(index_string)
                    pending = pending_adaln.setdefault(index, {})
                    pending[sub] = array
                    if {"adaln_proj.linear.weight", "adaln_proj.linear.bias"} <= pending.keys():
                        tables = h3._adaln_tables(pending, temb)
                        mx.eval(tables)
                        cached_block_tables[index] = tables
                        if blocks[index] is None:
                            blocks[index] = {}
                        assert blocks[index] is not None
                        blocks[index]["adaln_proj.linear.weight"] = None
                        blocks[index]["adaln_proj.linear.bias"] = None
                        del pending_adaln[index]
                    continue
                if quantization is not None and h3._is_quantizable(key) and not keep_fp32:
                    value = h3.quantize_matrix(array, quantization)
                    del array
                else:
                    value = array
                h3._eval_value(value)
                assign(key, value)
        del shard_arrays
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

    require(not pending_adaln,
            f"incomplete VDN AdaLN pairs for blocks {sorted(pending_adaln)}")
    missing_tables = [index for index, table in enumerate(cached_block_tables)
                      if table is None]
    require(not missing_tables, f"missing VDN AdaLN tables for blocks {missing_tables}")
    require(consumed_targets == target_set,
            f"VDN adapter targets differ: missing={sorted(target_set - consumed_targets)[:8]} "
            f"extra={sorted(consumed_targets - target_set)[:8]}")
    del adapter_arrays

    shift_scale = h3.linear(
        h3.silu(temb).astype(h3.weight_dtype(weights["norm_out.linear.weight"])),
        weights["norm_out.linear.weight"],
        weights["norm_out.linear.bias"],
    )
    norm_out_shift, norm_out_scale = mx.split(shift_scale, 2, axis=-1)
    mx.eval(norm_out_shift, norm_out_scale)
    require(all(block is not None for block in blocks),
            "VDN conversion did not load all transformer blocks")
    require(all(block is not None for block in refiner),
            "VDN conversion did not load both token refiner blocks")
    loaded_blocks = [block for block in blocks if block is not None]
    loaded_refiner = [block for block in refiner if block is not None]
    dit = h3.MLXMiniMaxH3DiT(weights, loaded_blocks, loaded_refiner, config)
    dit._adaln_cache = h3.MiniMaxH3StepCache(
        timesteps=cache_timesteps,
        block_tables=[table for table in cached_block_tables if table is not None],
        norm_out_shift=norm_out_shift,
        norm_out_scale=norm_out_scale,
    )
    return dit


def run_conversion(base: Path, stage_root: Path, output_root: Path) -> None:
    bootstrap = load_module(FASTVIDEO_BOOTSTRAP, "turbocider_fastvideo_converter")
    reference = bootstrap.install_fastvideo_namespace()
    # Importing the official converter module is safe here: it is a build-time
    # utility, never a TurboCider runtime dependency.
    import mlx.core as mx
    from fastvideo.mlx_runtime.fastwan import MLXQuantizationSpec, ensure_quantization_supported
    from fastvideo.mlx_runtime.minimax_h3 import save_mlx_h3_checkpoint

    spec = MLXQuantizationSpec.from_name("int6")
    require(spec is not None, "FastVideo MLX build does not support INT6")
    ensure_quantization_supported(spec)
    output = output_root / "int6"
    if (output / H3_MANIFEST_FILENAME).is_file() and (output / H3_WEIGHTS_FILENAME).is_file():
        print(f"[skip] existing base artifact: {output}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    timesteps = six_step_adaln_timesteps()
    print(f"[convert] ordinary FL2VA + stage-DMD adapter -> VDN INT6 "
          f"({len(timesteps)} AdaLN rows)", flush=True)
    dit = load_merged_vdn_dit(
        base,
        stage_root / STAGE / "adapters/turbo/adapter_model.safetensors",
        stage_root / STAGE / "adapters/turbo/adapter_spec.json",
        spec,
        timesteps,
    )
    save_mlx_h3_checkpoint(dit, output)
    del dit
    mx.clear_cache()
    del reference


def write_manifest(output_root: Path, base_info: dict[str, Any], stage_info: dict[str, Any],
                   asset_paths: dict[str, str]) -> None:
    manifest_path = output_root / "int6" / H3_MANIFEST_FILENAME
    manifest = read_json(manifest_path)
    manifest["profile"] = "minimax-h3-vdn"
    manifest["steps"] = STEPS
    manifest["source"] = {
        "repository": base_info["repository"],
        "revision": base_info["revision"],
        "config_sha256": base_info["config_sha256"],
        "transformer_index_sha256": base_info["index_sha256"],
        "provenance_sha256": base_info["provenance_sha256"],
    }
    manifest["vdn"] = {
        "stage": stage_info["stage"],
        "repository": stage_info["repository"],
        "revision": stage_info["revision"],
        "branch_sha256": stage_info["branch_sha256"],
        "branch_bytes": stage_info["branch_bytes"],
        "branch_config_sha256": stage_info["branch_config_sha256"],
        "turbo_adapter_family": "larryvrh_v4_step600_ema",
        "turbo_adapter_sha256": stage_info["adapter_sha256"],
        "turbo_adapter_bytes": stage_info["adapter_bytes"],
        "adapter_spec_sha256": stage_info["adapter_spec_sha256"],
        "modelscope_download_sha256": stage_info["stage_manifest_sha256"],
        "assets": asset_paths,
        "attention": {
            "version": 2,
            "anchor_frames": "both",
            "softmax_chunk": 5,
            "softmax_radius": 1,
            "delta_rule": "vdn_solve",
            "bridge": "alpha",
            "linear_head_dim": 128,
            "a_fp32": True,
            "enable_text_state": True,
            "short_conv_targets": ["k", "v"],
            "enable_softmax_gate": True,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def validate_output(output_root: Path, base_info: dict[str, Any], stage_info: dict[str, Any]) -> None:
    output = output_root / "int6"
    manifest_path = output / H3_MANIFEST_FILENAME
    weights_path = output / H3_WEIGHTS_FILENAME
    require(manifest_path.is_file() and weights_path.is_file(),
            "VDN INT6 output is incomplete")
    manifest = read_json(manifest_path)
    require(manifest.get("profile") == "minimax-h3-vdn" and
            manifest.get("steps") == STEPS,
            "VDN output manifest is not bound to the six-step profile")
    quant = manifest.get("quantization") or {}
    require(quant == {"mode": "affine", "bits": 6, "group_size": 64},
            f"VDN output is not affine INT6/g64: {quant}")
    require(manifest.get("adaln_cache", {}).get("timesteps") and
            len(manifest["adaln_cache"]["timesteps"]) == 11,
            "VDN output does not contain the six-step AdaLN union")
    require(manifest.get("source", {}).get("repository") == base_info["repository"] and
            manifest["source"].get("transformer_index_sha256") == base_info["index_sha256"],
            "VDN output base provenance mismatch")
    vdn = manifest.get("vdn") or {}
    require(vdn.get("branch_sha256") == stage_info["branch_sha256"] and
            vdn.get("turbo_adapter_sha256") == stage_info["adapter_sha256"] and
            vdn.get("turbo_adapter_family") == "larryvrh_v4_step600_ema",
            "VDN output branch/adapter identity mismatch")
    for key in ["linear_branch", "turbo_adapter"]:
        relative = vdn.get("assets", {}).get(key)
        require(isinstance(relative, str) and (output / relative).exists(),
                f"VDN asset link is missing: {key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True,
                        help="ordinary ModelScope FL2VA/transformer directory")
    parser.add_argument("--vdn-root", type=Path, required=True,
                        help="models/VDN-H3-ModelScope root")
    parser.add_argument("--out", type=Path, required=True,
                        help="output root, e.g. models/VDN-H3-MLX")
    parser.add_argument("--copy-assets", action="store_true",
                        help="copy the 5 GB branch/adapter instead of linking it")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--min-free-gib", type=float, default=12.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = args.base.expanduser().resolve()
    vdn_root = args.vdn_root.expanduser().resolve()
    output_root = args.out.expanduser().resolve()
    base_info = validate_base(base)
    stage_info = validate_stage(vdn_root)
    adapter_spec = read_json(
        vdn_root / STAGE / "adapters/turbo/adapter_spec.json"
    ).get("config")
    require(isinstance(adapter_spec, dict), "VDN adapter spec has no config object")
    targets = adapter_spec.get("targets")
    require(isinstance(targets, list) and len(targets) == 363,
            "VDN adapter must declare 363 exact targets")
    validate_adapter_target_coverage(base_info["mapped_keys"], targets)
    if args.validate_only:
        if (output_root / "int6" / H3_MANIFEST_FILENAME).is_file():
            validate_output(output_root, base_info, stage_info)
        print(f"[ready] base={base} stage={vdn_root}", flush=True)
        return
    free = shutil.disk_usage(output_root.parent).free / 2**30
    require(free >= args.min_free_gib,
            f"only {free:.1f} GiB free; refusing VDN conversion below {args.min_free_gib:.1f} GiB")
    output_root.mkdir(parents=True, exist_ok=True)
    run_conversion(base, vdn_root, output_root)
    paths = stage_assets(vdn_root, output_root, args.copy_assets)
    write_manifest(output_root, base_info, stage_info, paths)
    validate_output(output_root, base_info, stage_info)
    print(f"[ready] {output_root / 'int6'}", flush=True)


if __name__ == "__main__":
    main()
