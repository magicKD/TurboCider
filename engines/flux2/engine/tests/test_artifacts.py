import json
from pathlib import Path

import pytest

from flux2_engine.artifacts import ANEArtifactCatalog
from flux2_engine.errors import ArtifactError


def _manifest(
    tmp_path: Path,
    buckets: list[int],
    name: str,
    *,
    k: int = 3072,
    n: int = 3072,
    blocks: list[int] | None = None,
    shape_extra: dict | None = None,
) -> Path:
    artifact = tmp_path / f"{name}.mlpackage"
    artifact.mkdir()
    path = tmp_path / f"{name}.json"
    shape = {"buckets": buckets, "K": k, "N": n}
    shape.update(shape_extra or {})
    path.write_text(
        json.dumps(
            {
                "source": {"blocks": blocks or [0]},
                "shape": shape,
                "functions": {str(bucket): f"main_{bucket}" for bucket in buckets},
                "artifacts": {
                    str(block): {"int8_pc": artifact.name}
                    for block in (blocks or [0])
                },
            }
        )
    )
    return path


def test_catalog_selects_smallest_bucket_across_manifests(tmp_path: Path):
    catalog = ANEArtifactCatalog.from_paths(
        (_manifest(tmp_path, [1536], "general"), _manifest(tmp_path, [1088], "fast"))
    )
    plan = catalog.resolve(sequence_length=1065, variant="int8_pc", requested_blocks=None)
    assert plan.bucket == 1088
    assert plan.function_name == "main_1088"


def test_catalog_rejects_unsupported_sequence(tmp_path: Path):
    catalog = ANEArtifactCatalog.from_paths((_manifest(tmp_path, [1088], "fast"),))
    with pytest.raises(ArtifactError, match="no ANE bucket"):
        catalog.resolve(sequence_length=4096, variant="int8_pc", requested_blocks=None)


def test_catalog_selects_manifest_matching_model_dimensions(tmp_path: Path):
    catalog = ANEArtifactCatalog.from_paths((
        _manifest(tmp_path, [1088], "four_b", k=3072, n=3072),
        _manifest(tmp_path, [1088], "nine_b", k=4096, n=4096),
    ))
    plan = catalog.resolve(
        sequence_length=1080,
        variant="int8_pc",
        requested_blocks=None,
        k=4096,
        n=4096,
        block_count=24,
    )
    assert plan.manifest.k == 4096


def test_catalog_rejects_manifest_blocks_outside_model(tmp_path: Path):
    catalog = ANEArtifactCatalog.from_paths((
        _manifest(tmp_path, [1088], "too_many", blocks=[24]),
    ))
    with pytest.raises(ArtifactError, match="model block range"):
        catalog.resolve(
            sequence_length=1080,
            variant="int8_pc",
            requested_blocks=None,
            k=3072,
            n=3072,
            block_count=24,
        )


def test_catalog_skips_smaller_bucket_with_incompatible_blocks(tmp_path: Path):
    catalog = ANEArtifactCatalog.from_paths((
        _manifest(tmp_path, [1088], "wrong_model", blocks=[24]),
        _manifest(tmp_path, [1152], "compatible", blocks=[0, 1]),
    ))
    plan = catalog.resolve(
        sequence_length=1080,
        variant="int8_pc",
        requested_blocks=None,
        k=3072,
        n=3072,
        block_count=24,
    )
    assert plan.bucket == 1152
    assert plan.blocks == (0, 1)


def test_manifest_exposes_gpu_ane_mlp_partition(tmp_path: Path):
    manifest = ANEArtifactCatalog.from_paths((
        _manifest(
            tmp_path,
            [1088],
            "split",
            shape_extra={
                "mlp_width": 9216,
                "ane_mlp_start": 0,
                "ane_mlp_end": 6144,
            },
        ),
    )).manifests[0]
    assert manifest.mlp_width == 9216
    assert manifest.ane_mlp_end == 6144


def test_manifest_rejects_non_prefix_mlp_partition(tmp_path: Path):
    path = _manifest(
        tmp_path,
        [1088],
        "non_prefix",
        shape_extra={
            "mlp_width": 9216,
            "ane_mlp_start": 3072,
            "ane_mlp_end": 9216,
        },
    )
    with pytest.raises(ArtifactError, match="requires an ANE MLP prefix"):
        ANEArtifactCatalog.from_paths((path,))
