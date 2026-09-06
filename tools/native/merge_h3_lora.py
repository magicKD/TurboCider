#!/usr/bin/env python3
"""Merge a MiniMax-H3 Turbo LoRA into the released sharded transformer.

Two source layouts are supported:

* ``native``: already uses h3.c/SGLang names such as
  ``blocks.0.attn.qkv_proj.lora_A.weight``;
* ``lightx2v-4step``: the legacy CLI name for a provenance-pinned LightX2V
  Diffusers adapter layout, including the official four- and eight-step files. Its
  Q/K/V updates are merged into the native fused QKV weight and its
  ``.default`` adapter namespace is normalized.  The Sana-v2 544p v0.1 file
  omits alpha and therefore uses ``8 / 128 == 0.0625``; the newer 768p files
  declare alpha 128 for rank 128 and therefore use strength 1.0.

Every completed output gets a provenance manifest containing the input LoRA
digest, base/output shard digests, mapping counts, and merge strength. A
partial merge is resumable only through an identity-bound state file; existing
shards are never trusted merely because their names match.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Iterable, Mapping, Optional


LIGHTX2V_PROFILE = "lightx2v-4step"
NATIVE_PROFILE = "native"
LIGHTX2V_REPOSITORY = "lightx2v/Minimax-h3-Turbo"
LIGHTX2V_REVISION = "050494d5fe05bd1b1140b8565ea51dc33a5085a5"
LIGHTX2V_FILENAME = "minimax_h3_fl2v_turbo_4step_v0.1.safetensors"
LIGHTX2V_BYTES = 1_383_677_888
LIGHTX2V_SHA256 = "5ff4a12c8b4599fec716e1b15a45e504e0d1129111896bdcde5ac4a15e395b29"
LIGHTX2V_STRENGTH = 8.0 / 128.0
LIGHTX2V_RAW_TENSORS = 624
LIGHTX2V_RAW_PAIRS = 312
LIGHTX2V_MAPPED_WEIGHTS = 208
MANIFEST_NAME = "h3-turbo-merge-manifest.json"
STATE_NAME = ".h3-turbo-merge-state.json"


@dataclass(frozen=True)
class LoraUpdate:
    target: str
    a_key: str
    b_key: str
    source: str
    output_group: Optional[int] = None
    output_groups: Optional[int] = None


@dataclass(frozen=True)
class LoraPlan:
    profile: str
    updates: Mapping[str, tuple[LoraUpdate, ...]]
    raw_tensors: int
    raw_pairs: int
    mapped_weights: int


@dataclass(frozen=True)
class LightX2VArtifact:
    variant: str
    filename: str
    bytes: int
    sha256: str
    revision: str
    strength: float
    video_flow_shift: float
    audio_flow_shift: float
    training_resolution: str
    recommended_steps: int


LIGHTX2V_ARTIFACTS = {
    LIGHTX2V_FILENAME: LightX2VArtifact(
        variant="v0.1-544p",
        filename=LIGHTX2V_FILENAME,
        bytes=LIGHTX2V_BYTES,
        sha256=LIGHTX2V_SHA256,
        revision=LIGHTX2V_REVISION,
        strength=LIGHTX2V_STRENGTH,
        video_flow_shift=12.0,
        audio_flow_shift=3.0,
        training_resolution="544p_mixed_aspect_ratio",
        recommended_steps=4,
    ),
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors":
        LightX2VArtifact(
            variant="v1.0-768p",
            filename="minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
            bytes=1_383_677_808,
            sha256="1bdabc2e9fce20b1db563b96bcf6e46adcad4c1964f423676436bf266cc7416c",
            revision="ae75cbf19a9df1e24af09981b9aeb681ea371e23",
            strength=1.0,
            video_flow_shift=6.0,
            audio_flow_shift=3.0,
            training_resolution="768p",
            recommended_steps=4,
        ),
    "minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16.safetensors":
        LightX2VArtifact(
            variant="v1.1-768p",
            filename="minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16.safetensors",
            bytes=1_383_677_808,
            sha256="b5e25a59292d51bca3fc02b9a0b2284e11b4eb20921a9c5adc2db785956b8966",
            revision="2f8ea0dc0a7e2b26c9a43124eb89673787189b4e",
            strength=1.0,
            video_flow_shift=6.0,
            audio_flow_shift=3.0,
            training_resolution="768p",
            recommended_steps=4,
        ),
    "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors":
        LightX2VArtifact(
            variant="v1.0-544p-8step",
            filename="minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors",
            bytes=1_383_677_768,
            sha256="e16ac20824d6e6649b193806f8fb095639bd9946c97b1bb84b4248eab1cc807f",
            revision="5d1d4829fe614c1b93fcfd9cc7718e9ba71f73e1",
            strength=8.0 / 128.0,
            video_flow_shift=12.0,
            audio_flow_shift=3.0,
            training_resolution="544p_mixed_aspect_ratio",
            recommended_steps=8,
        ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path, help="base FL2VA/transformer directory")
    parser.add_argument("lora", type=Path, help="Turbo LoRA safetensors file")
    parser.add_argument("output", type=Path, help="output transformer directory")
    parser.add_argument(
        "--profile",
        choices=("auto", NATIVE_PROFILE, LIGHTX2V_PROFILE),
        default="auto",
        help="LoRA source layout; auto detects from tensor names",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=None,
        help=(
            "merge multiplier; pinned LightX2V artifacts require their "
            "published alpha/rank value"
        ),
    )
    parser.add_argument("--device", default="mps", choices=("mps", "cpu"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace output shards and restart the identity-bound merge state",
    )
    parser.add_argument(
        "--source-revision",
        default=None,
        help="optional source revision recorded in the provenance manifest",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate file identity, key mapping, and every base/LoRA shape",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def normalize_lora_key(name: str) -> str:
    for kind in ("lora_A", "lora_B"):
        suffix = f".{kind}.default.weight"
        if name.endswith(suffix):
            return name[: -len(suffix)] + f".{kind}.weight"
    return name


def _normalized_key_map(keys: Iterable[str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw in keys:
        target = normalize_lora_key(raw)
        if ".default." in target:
            raise ValueError(f"unhandled LoRA adapter namespace in {raw!r}")
        if target in normalized:
            raise ValueError(f"duplicate normalized LoRA key {target!r}")
        normalized[target] = raw
    return normalized


def detect_profile(keys: Iterable[str]) -> str:
    normalized = tuple(normalize_lora_key(key) for key in keys)
    if any(
        key.startswith("transformer_blocks.")
        or key.startswith("token_refiner.refiner_blocks.")
        for key in normalized
    ):
        return LIGHTX2V_PROFILE
    if any(
        key.startswith("blocks.") or key.startswith("token_refiner.blocks.")
        for key in normalized
    ):
        return NATIVE_PROFILE
    raise ValueError("could not detect a supported MiniMax-H3 LoRA layout")


_LIGHTX2V_PATTERN = re.compile(
    r"^(transformer_blocks\.(?P<main>\d+)|"
    r"token_refiner\.refiner_blocks\.(?P<refiner>\d+))\."
    r"(?P<module>attn\.to_q|attn\.to_k|attn\.to_v|attn\.to_out\.0|"
    r"ff\.net\.0\.proj|ff\.net\.2)\.lora_(?P<kind>[AB])\.weight$"
)


def _lightx2v_target(
    match: re.Match[str],
) -> tuple[str, Optional[int], Optional[int]]:
    if match.group("main") is not None:
        prefix = f"blocks.{int(match.group('main'))}"
    else:
        prefix = f"token_refiner.blocks.{int(match.group('refiner'))}"
    module = match.group("module")
    if module == "attn.to_q":
        return f"{prefix}.attn.qkv_proj.weight", 0, 3
    if module == "attn.to_k":
        return f"{prefix}.attn.qkv_proj.weight", 1, 3
    if module == "attn.to_v":
        return f"{prefix}.attn.qkv_proj.weight", 2, 3
    if module == "attn.to_out.0":
        return f"{prefix}.attn.out_proj.weight", None, None
    if module == "ff.net.0.proj":
        return f"{prefix}.mlp.fc1.weight", None, None
    if module == "ff.net.2":
        return f"{prefix}.mlp.fc2.weight", None, None
    raise AssertionError(f"unhandled LightX2V module {module}")


def build_lightx2v_plan(keys: Iterable[str]) -> LoraPlan:
    raw_keys = tuple(keys)
    normalized = _normalized_key_map(raw_keys)
    if len(raw_keys) != LIGHTX2V_RAW_TENSORS:
        raise ValueError(
            f"LightX2V checkpoint has {len(raw_keys)} tensors; "
            f"expected {LIGHTX2V_RAW_TENSORS}"
        )
    pairs: dict[str, dict[str, str]] = {}
    main_blocks: set[int] = set()
    refiner_blocks: set[int] = set()
    for name, raw in normalized.items():
        match = _LIGHTX2V_PATTERN.fullmatch(name)
        if match is None:
            raise ValueError(f"unsupported LightX2V tensor {name!r}")
        pair_name = name.rsplit(".lora_", 1)[0]
        kind = match.group("kind")
        pair = pairs.setdefault(pair_name, {})
        if kind in pair:
            raise ValueError(f"duplicate LightX2V {kind} tensor for {pair_name}")
        pair[kind] = raw
        if match.group("main") is not None:
            main_blocks.add(int(match.group("main")))
        else:
            refiner_blocks.add(int(match.group("refiner")))
    if len(pairs) != LIGHTX2V_RAW_PAIRS or any(
        set(pair) != {"A", "B"} for pair in pairs.values()
    ):
        raise ValueError(
            f"LightX2V A/B coverage is not {LIGHTX2V_RAW_PAIRS} complete pairs"
        )
    if main_blocks != set(range(50)) or refiner_blocks != set(range(2)):
        raise ValueError(
            "LightX2V block coverage mismatch: "
            f"main={sorted(main_blocks)} refiner={sorted(refiner_blocks)}"
        )

    updates: dict[str, list[LoraUpdate]] = {}
    for pair_name, pair in pairs.items():
        match = _LIGHTX2V_PATTERN.fullmatch(pair_name + ".lora_A.weight")
        assert match is not None
        target, group, groups = _lightx2v_target(match)
        updates.setdefault(target, []).append(
            LoraUpdate(
                target=target,
                a_key=pair["A"],
                b_key=pair["B"],
                source=pair_name,
                output_group=group,
                output_groups=groups,
            )
        )
    frozen = {
        target: tuple(
            sorted(
                values,
                key=lambda value: (
                    -1 if value.output_group is None else value.output_group
                ),
            )
        )
        for target, values in updates.items()
    }
    if len(frozen) != LIGHTX2V_MAPPED_WEIGHTS:
        raise ValueError(
            f"LightX2V maps to {len(frozen)} weights; "
            f"expected {LIGHTX2V_MAPPED_WEIGHTS}"
        )
    for target, values in frozen.items():
        if target.endswith("attn.qkv_proj.weight"):
            if [value.output_group for value in values] != [0, 1, 2]:
                raise ValueError(f"incomplete Q/K/V mapping for {target}")
        elif len(values) != 1 or values[0].output_group is not None:
            raise ValueError(f"unexpected fused mapping for {target}")
    return LoraPlan(
        profile=LIGHTX2V_PROFILE,
        updates=frozen,
        raw_tensors=len(raw_keys),
        raw_pairs=len(pairs),
        mapped_weights=len(frozen),
    )


def build_native_plan(keys: Iterable[str]) -> LoraPlan:
    raw_keys = tuple(keys)
    normalized = _normalized_key_map(raw_keys)
    suffix_a = ".lora_A.weight"
    suffix_b = ".lora_B.weight"
    a_names = {
        name[: -len(suffix_a)] for name in normalized if name.endswith(suffix_a)
    }
    b_names = {
        name[: -len(suffix_b)] for name in normalized if name.endswith(suffix_b)
    }
    unsupported = [
        name for name in normalized if not name.endswith((suffix_a, suffix_b))
    ]
    if unsupported or a_names != b_names or not a_names:
        raise ValueError(
            "native LoRA must contain only complete .lora_A/.lora_B weight pairs"
        )
    updates = {
        name + ".weight": (
            LoraUpdate(
                target=name + ".weight",
                a_key=normalized[name + suffix_a],
                b_key=normalized[name + suffix_b],
                source=name,
            ),
        )
        for name in sorted(a_names)
    }
    return LoraPlan(
        profile=NATIVE_PROFILE,
        updates=updates,
        raw_tensors=len(raw_keys),
        raw_pairs=len(a_names),
        mapped_weights=len(updates),
    )


def build_plan(keys: Iterable[str], requested: str) -> LoraPlan:
    raw_keys = tuple(keys)
    profile = detect_profile(raw_keys) if requested == "auto" else requested
    if profile == LIGHTX2V_PROFILE:
        return build_lightx2v_plan(raw_keys)
    if profile == NATIVE_PROFILE:
        return build_native_plan(raw_keys)
    raise ValueError(f"unsupported LoRA profile {profile!r}")


def resolve_strength(
    plan: LoraPlan,
    requested: Optional[float],
    canonical_strength: Optional[float] = None,
) -> float:
    expected = (
        LIGHTX2V_STRENGTH
        if canonical_strength is None and plan.profile == LIGHTX2V_PROFILE
        else canonical_strength
    )
    strength = (
        expected
        if requested is None and expected is not None
        else 1.0 if requested is None else float(requested)
    )
    if not math.isfinite(strength):
        raise ValueError("LoRA strength must be finite")
    if expected is not None and not math.isclose(
        strength, expected, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "pinned LightX2V adapter requires effective alpha/rank strength "
            f"{expected}; got {strength}"
        )
    return strength


def validate_lightx2v_file(path: Path, digest: str) -> LightX2VArtifact:
    artifact = LIGHTX2V_ARTIFACTS.get(path.name)
    if artifact is None:
        raise ValueError(
            "unsupported LightX2V artifact; expected one of "
            f"{sorted(LIGHTX2V_ARTIFACTS)}"
        )
    if path.stat().st_size != artifact.bytes:
        raise ValueError(
            f"canonical LightX2V file has {path.stat().st_size} bytes; "
            f"expected {artifact.bytes}"
        )
    if digest != artifact.sha256:
        raise ValueError(
            f"canonical LightX2V SHA-256 is {digest}; expected {artifact.sha256}"
        )
    return artifact


def _validate_update_shapes(
    weight: Any, a: Any, b: Any, update: LoraUpdate
) -> tuple[int, int]:
    if weight.ndim != 2 or a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"LoRA merge requires 2-D tensors for {update.source}")
    return validate_update_shape_values(
        tuple(weight.shape), tuple(a.shape), tuple(b.shape), update
    )


def validate_update_shape_values(
    weight_shape: tuple[int, ...],
    a_shape: tuple[int, ...],
    b_shape: tuple[int, ...],
    update: LoraUpdate,
) -> tuple[int, int]:
    if len(weight_shape) != 2 or len(a_shape) != 2 or len(b_shape) != 2:
        raise ValueError(f"LoRA merge requires 2-D tensors for {update.source}")
    if int(a_shape[0]) != int(b_shape[1]):
        raise ValueError(
            f"LoRA rank mismatch for {update.source}: "
            f"A={a_shape} B={b_shape}"
        )
    if int(a_shape[1]) != int(weight_shape[1]):
        raise ValueError(
            f"LoRA input mismatch for {update.source}: "
            f"W={weight_shape} A={a_shape}"
        )
    if update.output_group is None:
        start, stop = 0, int(weight_shape[0])
    else:
        assert update.output_groups is not None
        if int(weight_shape[0]) % update.output_groups:
            raise ValueError(f"fused output width is not divisible for {update.target}")
        width = int(weight_shape[0]) // update.output_groups
        start = update.output_group * width
        stop = start + width
    if int(b_shape[0]) != stop - start:
        raise ValueError(
            f"LoRA output mismatch for {update.source}: "
            f"slice={stop-start} B={b_shape}"
        )
    return start, stop


def merge_tensor(
    torch: Any,
    weight: Any,
    updates: tuple[LoraUpdate, ...],
    lora_handle: Any,
    strength: float,
    device: Any,
) -> Any:
    with torch.inference_mode():
        base = weight.to(device=device, dtype=torch.float32)
        for update in updates:
            a = lora_handle.get_tensor(update.a_key)
            b = lora_handle.get_tensor(update.b_key)
            start, stop = _validate_update_shapes(weight, a, b, update)
            delta = torch.mm(
                b.to(device=device, dtype=torch.float32),
                a.to(device=device, dtype=torch.float32),
            )
            base[start:stop].add_(delta, alpha=strength)
            del a, b, delta
        merged = base.to(dtype=weight.dtype).cpu().contiguous()
        del base
        return merged


def _identity(
    *,
    base: Path,
    index_path: Path,
    lora: Path,
    lora_sha256: str,
    plan: LoraPlan,
    strength: float,
    revision: Optional[str],
) -> dict[str, Any]:
    return {
        "schema": "h3-turbo-merge-state-v2",
        "base": str(base),
        "base_index_sha256": sha256_file(index_path),
        "lora": str(lora),
        "lora_bytes": lora.stat().st_size,
        "lora_sha256": lora_sha256,
        "profile": plan.profile,
        "raw_tensors": plan.raw_tensors,
        "raw_pairs": plan.raw_pairs,
        "mapped_weights": plan.mapped_weights,
        "strength": strength,
        "source_revision": revision,
    }


def _load_or_create_state(
    output: Path,
    identity: Mapping[str, Any],
    shards: list[str],
    overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    state_path = output / STATE_NAME
    existing_shards = [name for name in shards if (output / name).exists()]
    if overwrite:
        state = {"identity": dict(identity), "completed": {}}
        write_json_atomic(state_path, state)
        return state_path, state
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        if state.get("identity") != dict(identity):
            raise ValueError("partial merge state identity does not match this request")
        return state_path, state
    manifest = output / MANIFEST_NAME
    if existing_shards or manifest.exists():
        raise ValueError(
            "output already contains checkpoint data without a matching partial state; "
            "use a new directory or --overwrite"
        )
    state = {"identity": dict(identity), "completed": {}}
    write_json_atomic(state_path, state)
    return state_path, state


def main() -> None:
    args = parse_args()
    base = args.base.resolve()
    lora = args.lora.resolve()
    output = args.output.resolve()
    index_path = base / "model.safetensors.index.json"
    if not index_path.is_file() or not lora.is_file():
        raise SystemExit("base index or LoRA file is missing")
    if output == base or output in lora.parents:
        raise SystemExit("output must be separate from the base and LoRA paths")

    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise SystemExit("merge requires torch and safetensors") from exc
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable; use --device cpu")
    device = torch.device(args.device)
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads(index_path.read_text())
    shards = sorted(set(index["weight_map"].values()))

    lora_sha256 = sha256_file(lora)
    with safe_open(lora, framework="pt", device="cpu") as lora_handle:
        plan = build_plan(lora_handle.keys(), args.profile)
        lora_metadata = dict(lora_handle.metadata() or {})
    artifact = None
    revision = args.source_revision
    if plan.profile == LIGHTX2V_PROFILE:
        artifact = validate_lightx2v_file(lora, lora_sha256)
        if revision is not None and revision != artifact.revision:
            raise SystemExit(
                f"LightX2V revision must be {artifact.revision}; got {revision}"
            )
        revision = artifact.revision
    strength = resolve_strength(
        plan, args.strength,
        None if artifact is None else artifact.strength,
    )

    if args.check_only:
        missing_from_index = sorted(set(plan.updates) - set(index["weight_map"]))
        if missing_from_index:
            raise SystemExit(
                f"LoRA targets missing from base index: {missing_from_index[:5]}"
            )
        checked: set[str] = set()
        with safe_open(lora, framework="pt", device="cpu") as lora_handle:
            for shard_name in shards:
                source = base / shard_name
                with safe_open(source, framework="pt", device="cpu") as base_handle:
                    for name in base_handle.keys():
                        updates = plan.updates.get(name)
                        if not updates:
                            continue
                        weight_shape = tuple(base_handle.get_slice(name).get_shape())
                        for update in updates:
                            validate_update_shape_values(
                                weight_shape,
                                tuple(lora_handle.get_slice(update.a_key).get_shape()),
                                tuple(lora_handle.get_slice(update.b_key).get_shape()),
                                update,
                            )
                        checked.add(name)
        missing = sorted(set(plan.updates) - checked)
        if missing:
            raise SystemExit(f"LoRA targets not found in base shards: {missing[:5]}")
        print(
            json.dumps(
                {
                    "profile": plan.profile,
                    "lora_sha256": lora_sha256,
                    "raw_tensors": plan.raw_tensors,
                    "raw_pairs": plan.raw_pairs,
                    "mapped_weights": plan.mapped_weights,
                    "strength": strength,
                    "source_revision": revision,
                    "shape_checks": len(checked),
                    "variant": None if artifact is None else artifact.variant,
                    "video_flow_shift": (
                        None if artifact is None else artifact.video_flow_shift
                    ),
                    "audio_flow_shift": (
                        None if artifact is None else artifact.audio_flow_shift
                    ),
                    "training_resolution": (
                        None if artifact is None else artifact.training_resolution
                    ),
                    "recommended_steps": (
                        None if artifact is None else artifact.recommended_steps
                    ),
                },
                sort_keys=True,
            )
        )
        return

    identity = _identity(
        base=base,
        index_path=index_path,
        lora=lora,
        lora_sha256=lora_sha256,
        plan=plan,
        strength=strength,
        revision=revision,
    )
    if artifact is not None:
        identity.update({
            "variant": artifact.variant,
            "video_flow_shift": artifact.video_flow_shift,
            "audio_flow_shift": artifact.audio_flow_shift,
            "training_resolution": artifact.training_resolution,
            "recommended_steps": artifact.recommended_steps,
        })
    state_path, state = _load_or_create_state(
        output, identity, shards, args.overwrite
    )
    completed = state.setdefault("completed", {})
    merged_targets: set[str] = set()
    started = time.monotonic()

    with safe_open(lora, framework="pt", device="cpu") as lora_handle:
        for shard_index, shard_name in enumerate(shards, 1):
            source = base / shard_name
            target = output / shard_name
            record = completed.get(shard_name)
            if target.exists() and not args.overwrite:
                if not isinstance(record, dict):
                    raise SystemExit(f"untrusted existing output shard: {target}")
                actual = sha256_file(target)
                if actual != record.get("output_sha256"):
                    raise SystemExit(f"resumed output shard SHA mismatch: {target}")
                if sha256_file(source) != record.get("base_sha256"):
                    raise SystemExit(f"resumed base shard SHA mismatch: {source}")
                with safe_open(source, framework="pt", device="cpu") as base_handle:
                    merged_targets.update(
                        name for name in base_handle.keys() if name in plan.updates
                    )
                print(
                    f"[{shard_index:02d}/{len(shards):02d}] verified {target.name}",
                    flush=True,
                )
                continue

            shard_started = time.monotonic()
            base_sha256 = sha256_file(source)
            tensors: dict[str, Any] = {}
            shard_targets: list[str] = []
            with safe_open(source, framework="pt", device="cpu") as base_handle:
                metadata = dict(base_handle.metadata() or {})
                for name in base_handle.keys():
                    weight = base_handle.get_tensor(name)
                    updates = plan.updates.get(name)
                    if updates:
                        tensor_started = time.monotonic()
                        tensors[name] = merge_tensor(
                            torch, weight, updates, lora_handle, strength, device
                        )
                        merged_targets.add(name)
                        shard_targets.append(name)
                        print(
                            f"  merged {name} {tuple(weight.shape)} from "
                            f"{len(updates)} update(s) in "
                            f"{time.monotonic() - tensor_started:.2f}s",
                            flush=True,
                        )
                    else:
                        tensors[name] = weight.contiguous()

            temporary = target.with_suffix(target.suffix + ".tmp")
            if temporary.exists():
                temporary.unlink()
            save_file(tensors, temporary, metadata=metadata or None)
            os.replace(temporary, target)
            del tensors
            if device.type == "mps":
                torch.mps.empty_cache()
            output_sha256 = sha256_file(target)
            completed[shard_name] = {
                "base_bytes": source.stat().st_size,
                "base_sha256": base_sha256,
                "output_bytes": target.stat().st_size,
                "output_sha256": output_sha256,
                "merged_targets": sorted(shard_targets),
            }
            write_json_atomic(state_path, state)
            print(
                f"[{shard_index:02d}/{len(shards):02d}] wrote {target.name} "
                f"in {time.monotonic() - shard_started:.2f}s",
                flush=True,
            )

    missing = sorted(set(plan.updates) - merged_targets)
    unexpected = sorted(merged_targets - set(plan.updates))
    if missing or unexpected:
        raise SystemExit(
            f"LoRA/base mapping mismatch: missing={missing[:5]} "
            f"unexpected={unexpected[:5]}"
        )
    shutil.copy2(index_path, output / index_path.name)
    config = base / "config.json"
    if config.is_file():
        shutil.copy2(config, output / config.name)

    manifest = {
        "schema": "h3-turbo-merge-manifest-v2",
        "algorithm": "W_bf16 = round_bf16(W_f32 + strength * (B_f32 @ A_f32))",
        "identity": identity,
        "source": {
            "repository": (
                LIGHTX2V_REPOSITORY if plan.profile == LIGHTX2V_PROFILE else None
            ),
            "revision": revision,
            "lora_metadata": lora_metadata,
            "artifact": None if artifact is None else artifact.variant,
        },
        "mapping": {
            "raw_tensors": plan.raw_tensors,
            "raw_pairs": plan.raw_pairs,
            "mapped_weights": plan.mapped_weights,
            "mapped_weight_names": sorted(plan.updates),
            "qkv_policy": "separate Q/K/V deltas applied to fused output slices",
        },
        "shards": completed,
        "output_index_sha256": sha256_file(output / index_path.name),
        "output_config_sha256": (
            sha256_file(output / config.name) if config.is_file() else None
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json_atomic(output / MANIFEST_NAME, manifest)
    state_path.unlink()
    print(
        f"merged {plan.raw_pairs} LoRA pairs into {plan.mapped_weights} weights "
        f"across {len(shards)} shards in {time.monotonic() - started:.1f}s; "
        f"manifest={output / MANIFEST_NAME}",
        flush=True,
    )


if __name__ == "__main__":
    main()
