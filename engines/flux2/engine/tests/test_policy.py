from pathlib import Path

import pytest

from flux2_engine.config import AttentionMode, EngineConfig, ExecutionMode, PipelineMode
from flux2_engine.errors import AttentionPlanUnavailableError
from flux2_engine.policy import choose_backend


def _config(tmp_path: Path, **kwargs) -> EngineConfig:
    return EngineConfig(model_path=tmp_path / "model", mflux_root=tmp_path / "mflux", **kwargs)


def test_auto_one_shot_prefers_mlx(tmp_path: Path):
    decision = choose_backend(_config(tmp_path, ane_manifests=(tmp_path / "manifest.json",)))
    assert decision.mode is ExecutionMode.MLX


def test_auto_persistent_prefers_ane(tmp_path: Path):
    decision = choose_backend(_config(tmp_path, ane_manifests=(tmp_path / "manifest.json",), persistent=True))
    assert decision.mode is ExecutionMode.MLX_ANE


def test_experimental_attention_fails_closed(tmp_path: Path):
    with pytest.raises(AttentionPlanUnavailableError):
        choose_backend(_config(tmp_path, attention=AttentionMode.ANE_EXPERIMENTAL))


def test_edit_pipeline_uses_mlx_and_rejects_explicit_ane(tmp_path: Path):
    automatic = choose_backend(_config(
        tmp_path,
        pipeline=PipelineMode.EDIT,
        ane_manifests=(tmp_path / "manifest.json",),
        persistent=True,
    ))
    assert automatic.mode is ExecutionMode.MLX
    with pytest.raises(AttentionPlanUnavailableError, match="multi-reference edit"):
        choose_backend(_config(
            tmp_path,
            pipeline=PipelineMode.EDIT,
            mode=ExecutionMode.MLX_ANE,
            ane_manifests=(tmp_path / "manifest.json",),
        ))
