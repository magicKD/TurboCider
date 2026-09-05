"""Engine-specific exceptions."""


class Flux2EngineError(RuntimeError):
    """Base error for actionable inference-engine failures."""


class BackendUnavailableError(Flux2EngineError):
    """Raised when a requested execution backend cannot be constructed."""


class ArtifactError(Flux2EngineError):
    """Raised when an ANE manifest or artifact does not satisfy a request."""


class AttentionPlanUnavailableError(Flux2EngineError):
    """Raised when an experimental attention plan has not passed its gate."""
