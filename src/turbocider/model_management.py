"""Pinned model download, conversion, and Core ML cache orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from turbocider.device import current_device
from turbocider.errors import ModelPreparationError
from turbocider.paths import PACKAGE_ROOT, model_root, state_root
from turbocider.registry import ModelRegistry


GIB = 1 << 30
DEFAULT_RESERVE_GIB = 5.0


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _parse_blocks(value: str, count: int) -> List[int]:
    blocks = set()
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if "-" in item:
            first_text, last_text = item.split("-", 1)
            first, last = int(first_text), int(last_text)
        else:
            first = last = int(item)
        if first < 0 or last < first or last >= count:
            raise ModelPreparationError(
                "invalid block range %s; expected values in [0, %d]"
                % (item, count - 1)
            )
        blocks.update(range(first, last + 1))
    if not blocks:
        raise ModelPreparationError("at least one block must be selected")
    return sorted(blocks)


def _directory_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    return sum(
        child.stat().st_size
        for child in path.rglob("*")
        if child.is_file() and not child.is_symlink()
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _command_text(command: Sequence[str]) -> str:
    # Commands never contain access tokens; Hugging Face authentication is
    # inherited through its normal environment/keychain mechanism.
    return " ".join(str(item) for item in command)


@dataclass
class PreparationAction:
    kind: str
    description: str
    target: Path
    estimated_bytes: int = 0
    command: List[str] = field(default_factory=list)
    cwd: Optional[Path] = None
    environment: Dict[str, str] = field(default_factory=dict)
    cached: bool = False
    requires: List[Path] = field(default_factory=list)
    internal: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "description": self.description,
            "target": str(self.target),
            "estimated_bytes": self.estimated_bytes,
            "command": list(self.command),
            "cwd": str(self.cwd) if self.cwd else None,
            "environment_keys": sorted(self.environment),
            "cached": self.cached,
            "requires": [str(path) for path in self.requires],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PreparationOptions:
    download: bool = False
    ane: bool = False
    cache: bool = False
    dry_run: bool = False
    force: bool = False
    model_directory: Optional[Path] = None
    source_model: Optional[Path] = None
    lora: Optional[Path] = None
    ane_output: Optional[Path] = None
    cache_directory: Optional[Path] = None
    h3_rows: Optional[int] = None
    fastmetal_rows: int = 32760
    fastmetal_ane_width: int = 4096
    blocks: Optional[str] = None
    ltx_text_rows: int = 1024
    buckets: Sequence[int] = (1088,)
    workers: int = 1
    minimum_free_gib: float = DEFAULT_RESERVE_GIB
    python: Optional[Path] = None


class ModelPreparer:
    """Build and execute a reproducible preparation plan for a model pack."""

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        state_directory: Optional[Path] = None,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ):
        self.registry = registry or ModelRegistry()
        self.state_directory = (state_directory or state_root()).resolve()
        self.runner = runner

    def prepare(self, model_id: str, options: PreparationOptions) -> Dict[str, Any]:
        if options.workers < 1:
            raise ModelPreparationError("workers must be at least 1")
        if options.minimum_free_gib < 0:
            raise ModelPreparationError("minimum-free-gib must be nonnegative")
        if options.ltx_text_rows < 1:
            raise ModelPreparationError("ltx-text-rows must be positive")
        if options.fastmetal_rows < 1:
            raise ModelPreparationError("fastmetal-rows must be positive")
        if (
            options.fastmetal_ane_width <= 0
            or options.fastmetal_ane_width >= 8960
            or options.fastmetal_ane_width % 64
        ):
            raise ModelPreparationError(
                "fastmetal-ane-width must be a 64-aligned value in (0, 8960)"
            )
        if not options.buckets or any(int(value) <= 0 for value in options.buckets):
            raise ModelPreparationError("Core ML buckets must be positive")

        model = self.registry.get(model_id)
        if not model.preparation:
            raise ModelPreparationError(
                "model pack %s has no preparation recipe" % model_id
            )
        python = self._python(
            model.preparation, options.python, (), allow_unverified=True
        )
        actions, artifacts, sources = self._plan(model, options, python)
        required_modules = self._required_modules(actions)
        selected_python = self._python(
            model.preparation,
            options.python,
            required_modules,
            allow_unverified=options.dry_run,
        )
        if selected_python != python:
            python = selected_python
            actions, artifacts, sources = self._plan(model, options, python)
        write_bytes = sum(
            action.estimated_bytes for action in actions if not action.cached
        )
        capacity_target = next(
            (action.target for action in actions if not action.cached),
            options.model_directory or model_root(),
        )
        disk_root = _nearest_existing_parent(Path(capacity_target))
        free_bytes = shutil.disk_usage(disk_root).free
        reserve_bytes = int(options.minimum_free_gib * GIB)
        sufficient = write_bytes + reserve_bytes <= free_bytes
        report: Dict[str, Any] = {
            "schema": "turbocider-model-preparation-v1",
            "model": model.id,
            "engine": model.engine,
            "dry_run": options.dry_run,
            "pinned_sources": sources,
            "actions": [action.as_dict() for action in actions],
            "artifacts": artifacts,
            "disk": {
                "path": str(disk_root),
                "estimated_write_bytes": write_bytes,
                "free_bytes": free_bytes,
                "reserve_bytes": reserve_bytes,
                "sufficient": sufficient,
            },
        }
        if not sufficient:
            raise ModelPreparationError(
                "insufficient free space for %s: estimated writes need %.2f GiB "
                "plus %.2f GiB reserve, but %.2f GiB is free at %s"
                % (
                    model.id,
                    write_bytes / GIB,
                    reserve_bytes / GIB,
                    free_bytes / GIB,
                    disk_root,
                )
            )
        if options.dry_run:
            report["status"] = "planned"
            return report

        self._validate_python(python, actions)
        executed = []
        for action in actions:
            row = action.as_dict()
            started = time.perf_counter()
            if action.cached:
                row["status"] = "cached"
                row["duration_seconds"] = 0.0
                executed.append(row)
                continue
            missing = [path for path in action.requires if not path.exists()]
            if missing:
                raise ModelPreparationError(
                    "%s prerequisites are missing: %s"
                    % (action.kind, ", ".join(str(path) for path in missing))
                )
            print("+ " + _command_text(action.command or [action.internal or action.kind]), flush=True)
            try:
                if action.internal == "h3-layout":
                    self._prepare_h3_layout(action)
                else:
                    environment = os.environ.copy()
                    environment.update(action.environment)
                    action.target.parent.mkdir(parents=True, exist_ok=True)
                    self.runner(
                        action.command,
                        cwd=str(action.cwd or PACKAGE_ROOT),
                        env=environment,
                        check=True,
                    )
            except (OSError, subprocess.CalledProcessError) as error:
                raise ModelPreparationError(
                    "%s failed for %s: %s" % (action.kind, model.id, error)
                ) from error
            row["status"] = "completed"
            row["duration_seconds"] = time.perf_counter() - started
            executed.append(row)

        report["actions"] = executed
        report["status"] = "ready"
        report["prepared_at"] = datetime.now(timezone.utc).isoformat()
        report["device"] = current_device().as_dict()
        for name, value in list(report["artifacts"].items()):
            path = Path(value)
            report["artifacts"][name] = {
                "path": str(path),
                "exists": path.exists(),
                "bytes": _directory_bytes(path),
            }
        receipt = self.state_directory / "model-management" / (model.id + ".json")
        report["receipt"] = str(receipt)
        _atomic_json(receipt, report)
        return report

    def inspect(self, model_id: str) -> Dict[str, Any]:
        model = self.registry.get(model_id)
        receipt = self.state_directory / "model-management" / (model.id + ".json")
        return {
            "model": model.id,
            "engine": model.engine,
            "recipe": dict(model.preparation),
            "receipt": (
                json.loads(receipt.read_text(encoding="utf-8"))
                if receipt.is_file()
                else None
            ),
        }

    def _python(
        self,
        recipe: Mapping[str, Any],
        explicit: Optional[Path],
        modules: Iterable[str],
        *,
        allow_unverified: bool,
    ) -> Path:
        candidates: List[Path] = []
        if explicit is not None:
            candidates.append(explicit.expanduser())
        for value in recipe.get("python_candidates", []):
            if value:
                candidates.append(Path(str(value)).expanduser())
        candidates.append(Path(sys.executable))
        # Preserve a virtual environment's python symlink. Resolving it to the
        # host interpreter changes sys.prefix and silently drops that venv's
        # installed packages.
        existing = [path.absolute() for path in candidates if path.is_file()]
        if not existing:
            raise ModelPreparationError(
                "no preparation Python exists; pass --python with an interpreter "
                "that provides huggingface_hub, safetensors, torch, and coremltools"
            )
        module_list = sorted(set(modules))
        if not module_list:
            return existing[0]
        for candidate in existing:
            probe = self.runner(
                [str(candidate), "-c", "import " + ",".join(module_list)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if probe.returncode == 0:
                return candidate
            if explicit is not None:
                break
        if allow_unverified:
            return existing[0]
        raise ModelPreparationError(
            "no preparation Python provides required modules %s; pass --python "
            "with a compatible environment" % ", ".join(module_list)
        )

    def _validate_python(
        self, python: Path, actions: Sequence[PreparationAction]
    ) -> None:
        modules = self._required_modules(actions)
        if not modules:
            return
        probe = self.runner(
            [str(python), "-c", "import " + ",".join(sorted(modules))],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            raise ModelPreparationError(
                "preparation Python %s is missing required modules %s: %s"
                % (python, ", ".join(sorted(modules)), probe.stdout.strip())
            )

    @staticmethod
    def _required_modules(actions: Sequence[PreparationAction]) -> set[str]:
        modules: set[str] = set()
        if any(action.kind == "download" and not action.cached for action in actions):
            modules.add("huggingface_hub")
        if any(
            action.kind in {"ane_export", "coreml_cache"} and not action.cached
            for action in actions
        ):
            modules.update(("coremltools", "numpy", "safetensors"))
        if any(
            action.metadata.get("requires_mlx") and not action.cached
            for action in actions
        ):
            modules.add("mlx")
        if any(action.kind == "merge" and not action.cached for action in actions):
            modules.update(("safetensors", "torch"))
        return modules

    def _plan(self, model, options: PreparationOptions, python: Path):
        if model.engine == "h3":
            return self._plan_h3(model, options, python)
        if model.engine == "ltx25":
            return self._plan_ltx(model, options, python)
        if model.engine == "flux2":
            return self._plan_flux(model, options, python)
        if model.engine == "fastmetal":
            return self._plan_fastmetal(model, options, python)
        raise ModelPreparationError(
            "preparation is not implemented for engine %s" % model.engine
        )

    def _download_actions(
        self,
        recipe: Mapping[str, Any],
        python: Path,
        destination_override: Optional[Path],
        roles: Iterable[str],
    ):
        selected_roles = set(roles)
        actions: List[PreparationAction] = []
        sources: List[Dict[str, Any]] = []
        destinations: Dict[str, Path] = {}
        for source in recipe.get("sources", []):
            role = str(source["role"])
            if role not in selected_roles:
                continue
            destination = (
                destination_override.expanduser().resolve()
                if destination_override is not None and len(selected_roles) == 1
                else Path(str(source["destination"])).expanduser().resolve()
            )
            destinations[role] = destination
            expected = [destination / str(item) for item in source.get("required_paths", [])]
            cached = bool(expected) and all(path.exists() for path in expected)
            patterns = [str(item) for item in source.get("allow_patterns", [])]
            command = [
                str(python),
                "-c",
                (
                    "import json,sys; from huggingface_hub import snapshot_download; "
                    "snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], "
                    "local_dir=sys.argv[3], allow_patterns=json.loads(sys.argv[4]), "
                    "max_workers=int(sys.argv[5]))"
                ),
                str(source["repo_id"]),
                str(source["revision"]),
                str(destination),
                json.dumps(patterns),
                str(int(source.get("max_workers", 8))),
            ]
            actions.append(PreparationAction(
                kind="download",
                description="download pinned %s source" % role,
                target=destination,
                estimated_bytes=(
                    0 if cached else int(source.get("estimated_download_bytes", 0))
                ),
                command=command,
                cwd=PACKAGE_ROOT,
                cached=cached,
                metadata={"role": role},
            ))
            sources.append({
                "role": role,
                "repo_id": str(source["repo_id"]),
                "revision": str(source["revision"]),
                "destination": str(destination),
                "allow_patterns": patterns,
                "required_paths": [str(item) for item in source.get("required_paths", [])],
                "cached": cached,
            })
        return actions, sources, destinations

    def _plan_h3(self, model, options: PreparationOptions, python: Path):
        recipe = model.preparation
        actions: List[PreparationAction] = []
        sources: List[Dict[str, Any]] = []
        artifacts: Dict[str, str] = {}
        runtime_model = (
            options.model_directory.expanduser().resolve()
            if options.model_directory
            else Path(str(recipe["runtime_model_directory"])).resolve()
        )
        base = options.source_model.expanduser().resolve() if options.source_model else None
        lora = options.lora.expanduser().resolve() if options.lora else None

        if options.download:
            roles = []
            if base is None:
                roles.append("base")
            if lora is None:
                roles.append("adapter")
            download_actions, source_rows, destinations = self._download_actions(
                recipe, python, None, roles
            )
            actions.extend(download_actions)
            sources.extend(source_rows)
            base = base or destinations.get("base")
            adapter_root = destinations.get("adapter")
            if lora is None and adapter_root is not None:
                lora = adapter_root / str(recipe["merge"]["lora_subpath"])
            if base is None or lora is None:
                raise ModelPreparationError("H3 download recipe is incomplete")
            base_transformer, base_fl2va = self._h3_source_paths(base)
            merge = recipe["merge"]
            output_transformer = runtime_model / "FL2VA" / "transformer"
            cached = (output_transformer / "h3-turbo-merge-manifest.json").is_file()
            layout = PreparationAction(
                kind="layout",
                description="link immutable H3 tokenizer/VAE assets into the runtime model",
                target=runtime_model / "FL2VA",
                internal="h3-layout",
                cached=self._h3_layout_ready(base_fl2va, runtime_model / "FL2VA"),
                requires=[base_fl2va],
                metadata={"source_fl2va": str(base_fl2va)},
            )
            actions.append(layout)
            command = [
                str(python), str(merge["tool"]), str(base_transformer), str(lora),
                str(output_transformer), "--profile", str(merge["profile"]),
                "--source-revision", str(merge["source_revision"]),
            ]
            if options.force:
                command.append("--overwrite")
            actions.append(PreparationAction(
                kind="merge",
                description="merge the pinned LightX2V Turbo LoRA into H3",
                target=output_transformer,
                estimated_bytes=(
                    0 if cached else int(merge.get("estimated_output_bytes", 0))
                ),
                command=command,
                cwd=Path(str(merge["tool"])).resolve().parent.parent,
                cached=cached and not options.force,
                requires=[base_transformer, lora],
            ))

        ane_source = (
            options.source_model.expanduser().resolve()
            if options.source_model is not None and not options.download
            else runtime_model
        )
        if not ane_source.exists() and not options.download:
            configured = Path(str(model.config.get("model_path", "")))
            if configured.exists():
                ane_source = configured
        if options.ane or options.cache:
            if options.h3_rows is None or options.h3_rows <= 0:
                raise ModelPreparationError(
                    "H3 ANE artifacts are fixed-shape; pass the exact --h3-rows "
                    "for the target request instead of relying on a guessed value"
                )
            blocks = _parse_blocks(options.blocks or "0-49", 50)
            transformer, _ = self._h3_source_paths(ane_source)
            ane_output = (
                options.ane_output.expanduser().resolve()
                if options.ane_output
                else Path(str(recipe["ane"]["output_directory"])).resolve()
                    / ("rows-%d" % options.h3_rows)
            )
            command = [
                str(python), str(recipe["ane"]["tool"]), str(transformer),
                str(ane_output), "--blocks", self._range_text(blocks),
                "--rows", str(options.h3_rows), "--weight-bits", "8",
                "--weight-granularity", "per_channel",
                "--minimum-free-gib", str(options.minimum_free_gib),
            ]
            if options.cache:
                command.append("--compile")
            if options.force:
                command.append("--force")
            expected = [ane_output / ("block-%d" % block) / "manifest.json" for block in blocks]
            if options.cache:
                expected.extend(ane_output / ("block-%d.mlmodelc" % block) for block in blocks)
            cached = all(path.exists() for path in expected)
            actions.append(PreparationAction(
                kind="ane_export",
                description="export fixed-row H3 MLP partitions%s" % (
                    " and stable compiled Core ML models" if options.cache else ""
                ),
                target=ane_output,
                estimated_bytes=(0 if cached else (
                    int(recipe["ane"].get("estimated_bytes", 0))
                    * len(blocks) // 50
                )),
                command=command,
                cwd=Path(str(recipe["ane"]["tool"])).resolve().parent.parent,
                cached=cached and not options.force,
                requires=[transformer / "model.safetensors.index.json"],
                metadata={
                    "rows": options.h3_rows,
                    "blocks": blocks,
                    "activation_environment": {
                        "H3_COREML_ANE_DIR": str(ane_output),
                        "H3_COREML_ANE_BLOCKS": self._range_text(blocks),
                    },
                },
            ))
            artifacts["ane_directory"] = str(ane_output)
        artifacts["model_directory"] = str(runtime_model)
        return actions, artifacts, sources

    def _plan_ltx(self, model, options: PreparationOptions, python: Path):
        recipe = model.preparation
        actions: List[PreparationAction] = []
        sources: List[Dict[str, Any]] = []
        artifacts: Dict[str, str] = {}
        model_directory = (
            options.model_directory.expanduser().resolve()
            if options.model_directory
            else Path(str(recipe["runtime_model_directory"])).resolve()
        )
        if options.download:
            rows, source_rows, _ = self._download_actions(
                recipe, python, model_directory, ("model",)
            )
            actions.extend(rows)
            sources.extend(source_rows)
        transformer = (
            options.source_model.expanduser().resolve()
            if options.source_model
            else model_directory / str(recipe["ane"]["transformer_subpath"])
        )
        if not transformer.is_file() and not options.download:
            configured = Path(str(model.config.get("transformer_path", "")))
            if configured.is_file():
                transformer = configured
        if options.ane or options.cache:
            blocks = _parse_blocks(options.blocks or "0-47", 48)
            block_text = ",".join(str(block) for block in blocks)
            ane_root = (
                options.ane_output.expanduser().resolve()
                if options.ane_output
                else Path(str(recipe["ane"]["output_directory"])).resolve()
            )
            mlp = ane_root / "mlp"
            kv = ane_root / ("text-kv-r%d" % options.ltx_text_rows)
            mlp_expected = [
                mlp / stage / ("block-%d" % block) / "manifest.json"
                for stage in ("stage1", "stage2") for block in blocks
            ]
            kv_expected = [kv / ("block-%d" % block) / "manifest.json" for block in blocks]
            mlp_cached = all(path.exists() for path in mlp_expected)
            kv_cached = all(path.exists() for path in kv_expected)
            mlp_command = [
                str(python), str(recipe["ane"]["mlp_tool"]), str(transformer),
                str(mlp), "--blocks", block_text, "--ane-intermediate", "6912",
                "--variant", "int8_pc",
            ]
            kv_command = [
                str(python), str(recipe["ane"]["kv_tool"]), str(transformer),
                str(kv), "--blocks", block_text, "--variant", "int8_pc",
                "--text-rows", str(options.ltx_text_rows),
            ]
            if options.force:
                mlp_command.append("--force")
                kv_command.append("--force")
            actions.extend((
                PreparationAction(
                    kind="ane_export",
                    description="export and compile LTX Stage-1/Stage-2 MLP partitions",
                    target=mlp,
                    estimated_bytes=(0 if mlp_cached else (
                        int(recipe["ane"].get("mlp_estimated_bytes", 0))
                        * len(blocks) // 48
                    )),
                    command=mlp_command,
                    cwd=Path(str(recipe["ane"]["mlp_tool"])).resolve().parent.parent,
                    cached=mlp_cached and not options.force,
                    requires=[transformer],
                    metadata={"blocks": blocks},
                ),
                PreparationAction(
                    kind="ane_export",
                    description="export and compile LTX text K/V projections",
                    target=kv,
                    estimated_bytes=(0 if kv_cached else (
                        int(recipe["ane"].get("kv_estimated_bytes", 0))
                        * len(blocks) // 48
                    )),
                    command=kv_command,
                    cwd=Path(str(recipe["ane"]["kv_tool"])).resolve().parent.parent,
                    cached=kv_cached and not options.force,
                    requires=[transformer],
                    metadata={"blocks": blocks, "text_rows": options.ltx_text_rows},
                ),
            ))
            kv_artifact_key = (
                "ane_kv_r256_directory"
                if options.ltx_text_rows == 256
                else "ane_kv_directory"
            )
            artifacts.update({
                "ane_mlp_stage1_directory": str(mlp / "stage1"),
                "ane_mlp_stage2_directory": str(mlp / "stage2"),
                kv_artifact_key: str(kv),
            })
        artifacts["model_directory"] = str(model_directory)
        return actions, artifacts, sources

    def _plan_flux(self, model, options: PreparationOptions, python: Path):
        recipe = model.preparation
        actions: List[PreparationAction] = []
        sources: List[Dict[str, Any]] = []
        artifacts: Dict[str, str] = {}
        model_directory = (
            options.model_directory.expanduser().resolve()
            if options.model_directory
            else Path(str(recipe["runtime_model_directory"])).resolve()
        )
        if options.download:
            rows, source_rows, _ = self._download_actions(
                recipe, python, model_directory, ("model",)
            )
            actions.extend(rows)
            sources.extend(source_rows)
        source_model = (
            options.source_model.expanduser().resolve()
            if options.source_model
            else model_directory
        )
        if not source_model.exists() and not options.download:
            configured = Path(str(model.config.get("model_path", "")))
            if configured.exists():
                source_model = configured
        checkpoint = source_model / str(recipe["ane"]["checkpoint_subpath"])
        blocks = _parse_blocks(options.blocks or "0-19", 20)
        bucket_text = "-".join(str(value) for value in sorted(set(options.buckets)))
        ane_output = (
            options.ane_output.expanduser().resolve()
            if options.ane_output
                else Path(str(recipe["ane"]["output_directory"])).resolve()
                    / ("m" + bucket_text)
        )
        manifest = ane_output / "manifest.json"
        if options.cache and not options.ane and options.ane_output is None:
            configured_manifest = model.config.get("ane_manifests", "")
            if isinstance(configured_manifest, str) and configured_manifest:
                candidate = Path(configured_manifest).expanduser().resolve()
                if candidate.is_file():
                    manifest = candidate
                    ane_output = candidate.parent
        if options.ane:
            command = [
                str(python), str(recipe["ane"]["tool"]),
                "--checkpoint", str(checkpoint), "--out-dir", str(ane_output),
                "--buckets", *(str(value) for value in sorted(set(options.buckets))),
                "--blocks", *(str(block) for block in blocks),
                "--variants", "int8_pc",
            ]
            if options.force:
                command.append("--overwrite")
            environment = {
                "PYTHONPATH": os.pathsep.join(
                    part for part in (
                        str(Path(str(recipe["ane"]["research_root"])) / ".deps" / "coreml"),
                        os.environ.get("PYTHONPATH", ""),
                    ) if part
                )
            }
            cached = manifest.is_file()
            actions.append(PreparationAction(
                kind="ane_export",
                description="export FLUX.2 INT8 MLP branches for fixed sequence buckets",
                target=ane_output,
                estimated_bytes=(0 if cached else (
                    int(recipe["ane"].get("estimated_bytes", 0))
                    * len(blocks) // 20
                )),
                command=command,
                cwd=Path(str(recipe["ane"]["research_root"])).resolve(),
                environment=environment,
                cached=cached and not options.force,
                requires=[checkpoint],
                metadata={"blocks": blocks, "buckets": list(options.buckets)},
            ))
        if options.cache:
            cache_directory = (
                options.cache_directory.expanduser().resolve()
                if options.cache_directory
                else ane_output / "compiled"
            )
            runtime_manifest = cache_directory / "manifest.json"
            cached = runtime_manifest.is_file() and not options.force
            command = [
                str(python), str(recipe["cache"]["tool"]),
                "--manifest", str(manifest), "--variant", "int8_pc",
                "--output-dir", str(cache_directory),
                "--workers", str(options.workers),
            ]
            actions.append(PreparationAction(
                kind="coreml_cache",
                description="compile stable content-addressed FLUX.2 .mlmodelc assets",
                target=cache_directory,
                estimated_bytes=(0 if cached else (
                    int(recipe["cache"].get("estimated_bytes", 0))
                    * len(blocks) // 20
                )),
                command=command,
                cwd=Path(str(recipe["cache"]["tool"])).resolve().parent.parent,
                cached=cached,
                requires=[manifest],
                metadata={"workers": options.workers},
            ))
            artifacts["ane_runtime_manifest"] = str(runtime_manifest)
        elif options.ane:
            artifacts["ane_manifest"] = str(manifest)
        artifacts["model_directory"] = str(model_directory)
        return actions, artifacts, sources

    def _plan_fastmetal(self, model, options: PreparationOptions, python: Path):
        recipe = model.preparation
        actions: List[PreparationAction] = []
        sources: List[Dict[str, Any]] = []
        artifacts: Dict[str, str] = {}
        model_directory = (
            options.model_directory.expanduser().resolve()
            if options.model_directory
            else Path(str(recipe["runtime_model_directory"])).resolve()
        )
        if options.download:
            rows, source_rows, _ = self._download_actions(
                recipe, python, model_directory, ("model",)
            )
            actions.extend(rows)
            sources.extend(source_rows)
        source_model = (
            options.source_model.expanduser().resolve()
            if options.source_model
            else model_directory
        )
        if not source_model.exists() and not options.download:
            configured = Path(str(model.config.get("model_path", "")))
            if configured.exists():
                source_model = configured

        if options.ane or options.cache:
            block_count = int(recipe["ane"].get("blocks", 30))
            blocks = _parse_blocks(options.blocks or "0-%d" % (block_count - 1), block_count)
            if blocks != list(range(block_count)):
                raise ModelPreparationError(
                    "FastMetal GPU+ANE currently requires complete block coverage 0-%d"
                    % (block_count - 1)
                )
            ane_output = (
                options.ane_output.expanduser().resolve()
                if options.ane_output
                else Path(str(recipe["ane"]["output_directory"])).resolve()
            )
            manifest = ane_output / "manifest.json"
            cached = manifest.is_file() and not options.force
            command = [
                str(python), str(recipe["ane"]["tool"]),
                "--model-root", str(source_model),
                "--output-dir", str(ane_output),
                "--rows", str(options.fastmetal_rows),
                "--ane-width", str(options.fastmetal_ane_width),
                "--variant", str(recipe["ane"].get("variant", "int8_pc")),
                "--blocks", *(str(block) for block in blocks),
            ]
            if options.force:
                command.append("--force")
            actions.append(PreparationAction(
                kind="ane_export",
                description=(
                    "export and compile all FastMetal FFN prefixes for the "
                    "fixed 480x832x81 denoise shape"
                ),
                target=ane_output,
                estimated_bytes=(
                    0 if cached else int(recipe["ane"].get("estimated_bytes", 0))
                ),
                command=command,
                cwd=Path(str(recipe["ane"]["tool"])).resolve().parent.parent,
                cached=cached,
                requires=[
                    source_model / "mlx_dit.json",
                    source_model / "mlx_dit.safetensors",
                ],
                metadata={
                    "blocks": blocks,
                    "rows": options.fastmetal_rows,
                    "ane_intermediate": options.fastmetal_ane_width,
                    "variant": str(recipe["ane"].get("variant", "int8_pc")),
                    "requires_mlx": True,
                },
            ))
            artifacts["ane_manifest"] = str(manifest)
        artifacts["model_directory"] = str(model_directory)
        return actions, artifacts, sources

    @staticmethod
    def _range_text(blocks: Sequence[int]) -> str:
        if blocks and list(blocks) == list(range(blocks[0], blocks[-1] + 1)):
            return "%d-%d" % (blocks[0], blocks[-1])
        return ",".join(str(block) for block in blocks)

    @staticmethod
    def _h3_source_paths(source: Path):
        source = source.expanduser().resolve()
        if (source / "model.safetensors.index.json").is_file():
            transformer = source
            fl2va = source.parent
        elif (source / "transformer" / "model.safetensors.index.json").is_file():
            fl2va = source
            transformer = source / "transformer"
        else:
            fl2va = source / "FL2VA"
            transformer = fl2va / "transformer"
        return transformer, fl2va

    @staticmethod
    def _h3_layout_ready(source_fl2va: Path, destination_fl2va: Path) -> bool:
        names = ("model_index.json", "audio_vae", "processor", "text_encoder", "tokenizer", "video_vae")
        return all((destination_fl2va / name).exists() for name in names)

    @staticmethod
    def _prepare_h3_layout(action: PreparationAction) -> None:
        source = Path(str(action.metadata["source_fl2va"]))
        destination = action.target
        destination.mkdir(parents=True, exist_ok=True)
        for name in (
            "model_index.json", "audio_vae", "processor", "text_encoder",
            "tokenizer", "video_vae",
        ):
            source_path = source / name
            target = destination / name
            if not source_path.exists():
                raise ModelPreparationError("H3 base component is missing: %s" % source_path)
            if target.exists() or target.is_symlink():
                if target.resolve() != source_path.resolve():
                    raise ModelPreparationError(
                        "refusing to replace existing H3 component %s" % target
                    )
                continue
            target.symlink_to(source_path, target_is_directory=source_path.is_dir())


def preparation_options_from_namespace(args) -> PreparationOptions:
    return PreparationOptions(
        download=bool(args.download),
        ane=bool(args.ane),
        cache=bool(args.cache),
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        model_directory=args.model_dir,
        source_model=args.source_model,
        lora=args.lora,
        ane_output=args.ane_output,
        cache_directory=args.cache_dir,
        h3_rows=args.h3_rows,
        fastmetal_rows=args.fastmetal_rows,
        fastmetal_ane_width=args.fastmetal_ane_width,
        blocks=args.blocks,
        ltx_text_rows=args.ltx_text_rows,
        buckets=tuple(args.bucket or (1088,)),
        workers=args.workers,
        minimum_free_gib=args.minimum_free_gib,
        python=args.python,
    )
