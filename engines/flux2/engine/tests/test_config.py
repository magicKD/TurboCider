from pathlib import Path

import pytest

from flux2_engine.config import (
    AttentionMode,
    EngineConfig,
    ExecutionMode,
    GenerationProfile,
    GenerationRequest,
    ModelVariant,
)


def test_profiles_resolve_steps():
    assert GenerationProfile.QUALITY.default_steps == 4
    assert GenerationProfile.BALANCED.default_steps == 3
    assert GenerationProfile.PREVIEW.default_steps == 2


def test_request_uses_profile_steps():
    request = GenerationRequest(prompt="test", profile=GenerationProfile.BALANCED)
    assert request.resolved_steps == 3


def test_request_rejects_non_aligned_resolution():
    with pytest.raises(ValueError, match="multiples of 16"):
        GenerationRequest(prompt="test", width=513)


def test_explicit_ane_requires_manifest(tmp_path: Path):
    with pytest.raises(ValueError, match="requires at least one ANE manifest"):
        EngineConfig(
            model_path=tmp_path / "model",
            mflux_root=tmp_path / "mflux",
            mode=ExecutionMode.MLX_ANE,
        )


def test_public_config_accepts_string_enums(tmp_path: Path):
    config = EngineConfig(
        model_path=tmp_path / "model",
        mflux_root=tmp_path / "mflux",
        mode="mlx",
        attention="auto",
    )
    request = GenerationRequest(prompt="test", profile="preview")
    assert config.mode is ExecutionMode.MLX
    assert config.attention is AttentionMode.AUTO
    assert request.profile is GenerationProfile.PREVIEW


def _write_transformer_config(path: Path, **values) -> Path:
    transformer = path / "transformer"
    transformer.mkdir(parents=True)
    payload = {
        "num_layers": 5,
        "num_single_layers": 20,
        "num_attention_heads": 24,
        "joint_attention_dim": 7680,
    }
    payload.update(values)
    (transformer / "config.json").write_text(__import__("json").dumps(payload))
    return path


def test_model_variant_is_detected_from_transformer_config(tmp_path: Path):
    model = _write_transformer_config(
        tmp_path / "FLUX.2-klein-9B",
        num_layers=8,
        num_single_layers=24,
        num_attention_heads=32,
        joint_attention_dim=12288,
    )
    config = EngineConfig(model_path=model, mflux_root=tmp_path / "mflux")
    assert config.resolved_model_variant is ModelVariant.KLEIN_9B


def test_model_variant_falls_back_to_checkpoint_directory_name(tmp_path: Path):
    config = EngineConfig(
        model_path=tmp_path / "FLUX.2-klein-9B-kv",
        mflux_root=tmp_path / "mflux",
    )
    assert config.resolved_model_variant is ModelVariant.KLEIN_9B_KV


def test_explicit_model_variant_must_match_checkpoint(tmp_path: Path):
    model = _write_transformer_config(tmp_path / "FLUX.2-klein-4B")
    config = EngineConfig(
        model_path=model,
        mflux_root=tmp_path / "mflux",
        model_variant="flux2-klein-9b",
    )
    with pytest.raises(ValueError, match="does not match"):
        _ = config.resolved_model_variant


def test_9b_kv_uses_the_9b_architecture_signature(tmp_path: Path):
    model = _write_transformer_config(
        tmp_path / "FLUX.2-klein-9B-kv",
        num_layers=8,
        num_single_layers=24,
        num_attention_heads=32,
        joint_attention_dim=12288,
    )
    config = EngineConfig(
        model_path=model,
        mflux_root=tmp_path / "mflux",
        model_variant=ModelVariant.KLEIN_9B_KV,
    )
    assert config.resolved_model_variant is ModelVariant.KLEIN_9B_KV
