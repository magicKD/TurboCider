"""Model pack loading and runtime path expansion."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from turbocider.errors import ModelNotFoundError, ValidationError
from turbocider.models import ExecutionPlan, ModelDescriptor
from turbocider.paths import PACKAGE_ROOT, model_root, workspace_root


_ENGINE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]+$")


def _expand_value(value: Any) -> Any:
    if isinstance(value, str):
        expanded = value.replace("${TURBOCIDER}", str(PACKAGE_ROOT))
        expanded = expanded.replace("${WORKSPACE}", str(workspace_root()))
        expanded = expanded.replace("${TURBOCIDER_MODELS}", str(model_root()))
        return os.path.expandvars(os.path.expanduser(expanded))
    if isinstance(value, list):
        return [_expand_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_value(item) for key, item in value.items()}
    return value


class ModelRegistry:
    def __init__(self, search_paths: Iterable[Path] = ()):
        default = PACKAGE_ROOT / "model-packs"
        self.search_paths = [default]
        self.search_paths.extend(Path(path).expanduser().resolve() for path in search_paths)
        env_paths = os.environ.get("TURBOCIDER_MODEL_PACKS", "")
        self.search_paths.extend(
            Path(item).expanduser().resolve()
            for item in env_paths.split(os.pathsep)
            if item
        )
        self._models: Dict[str, ModelDescriptor] = {}
        self.reload()

    def reload(self) -> None:
        models: Dict[str, ModelDescriptor] = {}
        for directory in self.search_paths:
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.json")):
                descriptor = self._load(path)
                models[descriptor.id] = descriptor
        self._models = models

    def _load(self, path: Path) -> ModelDescriptor:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            config = _expand_value(dict(raw.get("config", {})))
            for key, candidates in list(config.items()):
                if not key.endswith("_candidates") or not isinstance(candidates, list):
                    continue
                target = key[: -len("_candidates")]
                selected = next(
                    (str(candidate) for candidate in candidates if candidate and Path(str(candidate)).exists()),
                    "",
                )
                config[target] = selected
            engine = str(raw["engine"])
            if not _ENGINE_ID.fullmatch(engine):
                raise ValueError("invalid engine id: %s" % engine)
            plans = [ExecutionPlan.from_dict(_expand_value(item)) for item in raw["plans"]]
            return ModelDescriptor(
                id=str(raw["id"]),
                name=str(raw["name"]),
                engine=engine,
                version=str(raw.get("version", "1")),
                capabilities=dict(raw.get("capabilities", {})),
                config=config,
                plans=plans,
                preparation=_expand_value(dict(raw.get("preparation", {}))),
                source=path.resolve(),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValidationError("invalid model pack %s: %s" % (path, error)) from error

    def all(self) -> List[ModelDescriptor]:
        return sorted(self._models.values(), key=lambda item: item.id)

    def get(self, model_id: str) -> ModelDescriptor:
        try:
            return self._models[model_id]
        except KeyError as error:
            raise ModelNotFoundError("unknown model: %s" % model_id) from error

    def doctor(self) -> List[Dict[str, Any]]:
        reports = []
        for model in self.all():
            config_paths = {}
            for key, value in model.config.items():
                if key.endswith(("_path", "_root", "_executable", "_directory")):
                    candidate = Path(str(value))
                    config_paths[key] = {
                        "path": str(candidate),
                        "exists": candidate.exists(),
                    }
            reports.append({
                "id": model.id,
                "engine": model.engine,
                "paths": config_paths,
                "available": all(item["exists"] for item in config_paths.values()),
            })
        return reports
