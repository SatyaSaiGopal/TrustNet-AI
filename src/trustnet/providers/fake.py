"""A scripted provider for unit tests. Not selectable from configuration."""

from collections.abc import Mapping

from trustnet.providers.base import GenerationRequest, GenerationResponse
from trustnet.schemas import MOCK_PROVIDER_NAME


class DeterministicProvider:
    """Returns a fixed reply per prompt. Output is mock output, not model output.

    Its name is the mock provider name, so any ``AgentResponse`` built from its
    replies is labelled ``is_mock``.
    """

    name = MOCK_PROVIDER_NAME

    def __init__(
        self,
        responses: Mapping[str, str] | None = None,
        default: str = "MOCK RESPONSE",
        model: str = "deterministic-v0",
    ) -> None:
        self.model = model
        self._responses = dict(responses or {})
        self._default = default
        self.calls: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.calls.append(request)
        text = self._responses.get(request.prompt, self._default)
        return GenerationResponse(text=text, provider=self.name, model=self.model)
