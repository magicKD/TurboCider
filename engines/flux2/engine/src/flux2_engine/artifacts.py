"""Core ML artifact discovery and fixed-shape bucket selection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from flux2_engine.errors import ArtifactError


@dataclass(frozen=True, slots=True)
class ANEManifest:
    path: Path
    blocks: tuple[int, ...]
    buckets: tuple[int, ...]
    k: int
    n: int
    mlp_width: int | None
    ane_mlp_start: int
    ane_mlp_end: int | None
    functions: dict[int, str]
    artifacts: dict[int, dict[str, Path]]

    @classmethod
    def load(cls, path: Path) -> ANEManifest:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise ArtifactError(f"ANE manifest does not exist: {resolved}")
        raw = json.loads(resolved.read_text())
        try:
            shape = raw["shape"]
            buckets = tuple(sorted(set(shape.get("buckets") or [shape["M"]])))
            blocks = tuple(sorted(int(block) for block in raw["source"]["blocks"]))
            raw_functions = raw.get("functions") or {str(shape["M"]): raw["function_name"]}
            functions = {int(bucket): name for bucket, name in raw_functions.items()}
            artifacts = {
                int(block): {variant: (resolved.parent / filename).resolve() for variant, filename in variants.items()}
                for block, variants in raw["artifacts"].items()
            }
            mlp_width = (
                int(shape["mlp_width"])
                if shape.get("mlp_width") is not None
                else None
            )
            ane_mlp_start = int(shape.get("ane_mlp_start", 0))
            ane_mlp_end = (
                int(shape.get("ane_mlp_end", mlp_width))
                if mlp_width is not None
                else None
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactError(f"invalid ANE manifest {resolved}: {error}") from error
        missing_functions = sorted(set(buckets) - set(functions))
        if missing_functions:
            raise ArtifactError(f"manifest has no functions for buckets {missing_functions}")
        if mlp_width is not None and not (
            0 <= ane_mlp_start < int(ane_mlp_end) <= mlp_width
        ):
            raise ArtifactError(
                "invalid ANE MLP partition "
                f"[{ane_mlp_start},{ane_mlp_end}) for width {mlp_width}"
            )
        if ane_mlp_start != 0:
            raise ArtifactError(
                "the current GPU complement requires an ANE MLP prefix"
            )
        return cls(
            path=resolved,
            blocks=blocks,
            buckets=buckets,
            k=int(shape["K"]),
            n=int(shape["N"]),
            mlp_width=mlp_width,
            ane_mlp_start=ane_mlp_start,
            ane_mlp_end=ane_mlp_end,
            functions=functions,
            artifacts=artifacts,
        )

    def smallest_bucket(self, sequence_length: int) -> int | None:
        return next((bucket for bucket in self.buckets if bucket >= sequence_length), None)


@dataclass(frozen=True, slots=True)
class ResolvedANEPlan:
    manifest: ANEManifest
    bucket: int
    blocks: tuple[int, ...]
    variant: str

    def artifact_for(self, block: int) -> Path:
        try:
            path = self.manifest.artifacts[block][self.variant]
        except KeyError as error:
            raise ArtifactError(f"block {block} has no {self.variant} artifact") from error
        if not path.exists():
            raise ArtifactError(f"ANE artifact does not exist: {path}")
        return path

    @property
    def function_name(self) -> str:
        return self.manifest.functions[self.bucket]


class ANEArtifactCatalog:
    def __init__(self, manifests: tuple[ANEManifest, ...]):
        self.manifests = manifests

    @classmethod
    def from_paths(cls, paths: tuple[Path, ...]) -> ANEArtifactCatalog:
        return cls(tuple(ANEManifest.load(path) for path in paths))

    def resolve(
        self,
        *,
        sequence_length: int,
        variant: str,
        requested_blocks: tuple[int, ...] | None,
        k: int | None = None,
        n: int | None = None,
        block_count: int | None = None,
    ) -> ResolvedANEPlan:
        candidates: list[tuple[int, ANEManifest, tuple[int, ...]]] = []
        rejections: list[str] = []
        for manifest in self.manifests:
            if k is not None and manifest.k != k:
                continue
            if n is not None and manifest.n != n:
                continue
            bucket = manifest.smallest_bucket(sequence_length)
            if bucket is None:
                continue
            blocks = requested_blocks if requested_blocks is not None else manifest.blocks
            if block_count is not None:
                invalid = sorted(
                    block for block in blocks if block < 0 or block >= block_count
                )
                if invalid:
                    rejections.append(
                        f"{manifest.path}: blocks {invalid} exceed model block range "
                        f"0..{block_count - 1}"
                    )
                    continue
            missing = sorted(set(blocks) - set(manifest.blocks))
            if missing:
                rejections.append(
                    f"{manifest.path}: manifest does not contain blocks {missing}"
                )
                continue
            unavailable = sorted(
                block
                for block in blocks
                if variant not in manifest.artifacts.get(block, {})
            )
            if unavailable:
                rejections.append(
                    f"{manifest.path}: blocks {unavailable} have no {variant} artifact"
                )
                continue
            candidates.append((bucket, manifest, blocks))
        if not candidates:
            available = sorted(
                (manifest.k, manifest.n, bucket)
                for manifest in self.manifests
                for bucket in manifest.buckets
            )
            detail = f"; rejected candidates: {'; '.join(rejections)}" if rejections else ""
            raise ArtifactError(
                "no ANE bucket/artifact fits "
                f"sequence length {sequence_length}, K={k}, N={n}; "
                f"available (K, N, bucket): {available}{detail}"
            )
        bucket, manifest, blocks = min(candidates, key=lambda item: item[0])
        return ResolvedANEPlan(manifest=manifest, bucket=bucket, blocks=blocks, variant=variant)
