"""TurboCider error hierarchy."""


class TurboCiderError(RuntimeError):
    """Base user-facing framework error."""


class ValidationError(TurboCiderError):
    """A request or model pack is invalid."""


class ModelNotFoundError(TurboCiderError):
    """The requested model is not registered."""


class PlanUnavailableError(TurboCiderError):
    """No execution plan satisfies the request."""


class EngineUnavailableError(TurboCiderError):
    """The selected engine cannot run on this machine/configuration."""


class ModelPreparationError(TurboCiderError):
    """A model download, conversion, or compiled-cache step cannot run."""


class JobNotFoundError(TurboCiderError):
    """A job identifier is unknown."""
