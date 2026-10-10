"""The provider interface every model backend implements."""

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class ProviderError(RuntimeError):
    """A provider could not produce a response. The message says why."""


class ProviderConfigurationError(ProviderError):
    """The provider cannot be built from the current settings."""


class ProviderUnavailableError(ProviderError):
    """The model server cannot be reached, or the requested model is not installed."""


class ProviderTimeoutError(ProviderError):
    """The model did not finish replying within the configured time."""


class GenerationRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt: str = Field(min_length=1)
    system: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int | None = None
    # JSON Schema the reply text must conform to, for backends that can constrain output.
    response_schema: dict[str, Any] | None = None


class GenerationResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    provider: str
    model: str


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Return the model's reply, or raise ``ProviderError``. Never invent a reply."""
        ...
