"""Built-in and installed engine-adapter registry."""

from __future__ import annotations

import re
from importlib import metadata
from typing import Callable, Dict, Optional, Union

from turbocider.adapters.base import EngineAdapter
from turbocider.adapters.fastmetal import FastMetalAdapter
from turbocider.adapters.flux2 import Flux2Adapter
from turbocider.adapters.h3 import H3Adapter
from turbocider.adapters.ltx import LTXAdapter


AdapterFactory = Union[type[EngineAdapter], Callable[[], EngineAdapter]]
_ENGINE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]+$")
_ENTRY_POINT_GROUP = "turbocider.adapters"
_BUILTIN_ADAPTERS: Dict[str, AdapterFactory] = {
    "h3": H3Adapter,
    "ltx25": LTXAdapter,
    "flux2": Flux2Adapter,
    "fastmetal": FastMetalAdapter,
}
_REGISTERED_ADAPTERS: Dict[str, AdapterFactory] = {}


def register_adapter(
    engine: str,
    factory: AdapterFactory,
    *,
    replace: bool = False,
) -> None:
    """Register an adapter class/factory for embedding and tests.

    Distributions should normally publish the same factory through the
    ``turbocider.adapters`` Python entry-point group instead.
    """
    if not _ENGINE_ID.fullmatch(engine):
        raise ValueError("invalid engine adapter id: %s" % engine)
    if not callable(factory):
        raise TypeError("engine adapter factory must be callable")
    if not replace and (
        engine in _BUILTIN_ADAPTERS or engine in _REGISTERED_ADAPTERS
    ):
        raise ValueError("engine adapter is already registered: %s" % engine)
    _REGISTERED_ADAPTERS[engine] = factory


def _installed_factory(engine: str) -> Optional[AdapterFactory]:
    discovered = metadata.entry_points()
    if hasattr(discovered, "select"):
        matches = list(discovered.select(group=_ENTRY_POINT_GROUP, name=engine))
    else:  # Python 3.9 compatibility.
        matches = [
            entry
            for entry in discovered.get(_ENTRY_POINT_GROUP, ())
            if entry.name == engine
        ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(
            "multiple installed engine adapters use id %s: %s"
            % (engine, ", ".join(entry.value for entry in matches))
        )
    loaded = matches[0].load()
    if not callable(loaded):
        raise ValueError(
            "installed engine adapter %s must be a class or zero-argument factory"
            % engine
        )
    return loaded


def _instantiate(engine: str, factory: AdapterFactory) -> EngineAdapter:
    try:
        adapter = factory()
    except Exception as error:
        raise ValueError(
            "failed to initialize engine adapter %s: %s" % (engine, error)
        ) from error
    if not isinstance(adapter, EngineAdapter):
        raise ValueError(
            "engine adapter %s factory returned %s, expected EngineAdapter"
            % (engine, type(adapter).__name__)
        )
    if adapter.name != engine:
        raise ValueError(
            "engine adapter entry %s returned adapter named %s"
            % (engine, adapter.name)
        )
    return adapter


def adapter_for(engine: str) -> EngineAdapter:
    factory = _REGISTERED_ADAPTERS.get(engine) or _BUILTIN_ADAPTERS.get(engine)
    if factory is None:
        factory = _installed_factory(engine)
        if factory is not None:
            _REGISTERED_ADAPTERS[engine] = factory
    if factory is None:
        raise ValueError(
            "unsupported engine adapter: %s; install an adapter entry point "
            "in group %s" % (engine, _ENTRY_POINT_GROUP)
        )
    return _instantiate(engine, factory)


__all__ = ["AdapterFactory", "EngineAdapter", "adapter_for", "register_adapter"]
