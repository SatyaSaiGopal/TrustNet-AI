"""Local inference through Ollama, using LangChain's ``ChatOllama``."""

import time
from collections.abc import Callable
from typing import Any

import httpx
import ollama
from langchain_ollama import ChatOllama

from trustnet.providers.base import (
    GenerationRequest,
    GenerationResponse,
    ProviderConfigurationError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

# Builds the chat model from ChatOllama's keyword arguments. Injectable for tests.
ChatModelFactory = Callable[..., Any]


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 600.0,
        chat_model_factory: ChatModelFactory = ChatOllama,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not model or not model.strip():
            raise ProviderConfigurationError("an Ollama model name is required")
        if timeout_seconds <= 0:
            raise ProviderConfigurationError("the Ollama timeout must be greater than zero")
        self.model = model.strip()
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._chat_model_factory = chat_model_factory
        self._clock = clock

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        chat = self._chat_model_factory(
            model=self.model,
            base_url=self._base_url,
            temperature=request.temperature,
            seed=request.seed,
            format=request.response_schema,
            # Bounds each network wait; the deadline below bounds the whole reply.
            client_kwargs={"timeout": self._timeout},
        )
        messages = [("human", request.prompt)]
        if request.system is not None:
            messages.insert(0, ("system", request.system))

        deadline = self._clock() + self._timeout
        parts: list[str] = []
        try:
            for chunk in chat.stream(messages):
                if isinstance(chunk.content, str):
                    parts.append(chunk.content)
                if self._clock() > deadline:
                    raise ProviderTimeoutError(self._timeout_message())
        except ProviderError:
            raise
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(self._timeout_message()) from exc
        except (httpx.ConnectError, ConnectionError) as exc:
            raise ProviderUnavailableError(
                f"Cannot reach Ollama at {self._base_url}. Check that Ollama is running "
                f"and that TRUSTNET_OLLAMA_BASE_URL is correct. ({exc})"
            ) from exc
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise ProviderUnavailableError(
                    f"Ollama model '{self.model}' is not installed at {self._base_url}. "
                    f"Run `ollama pull {self.model}`, or set TRUSTNET_OLLAMA_MODEL to a "
                    "model shown by `ollama list`."
                ) from exc
            raise ProviderError(
                f"Ollama rejected the request for model '{self.model}': {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama request to {self._base_url} failed: {exc}") from exc

        text = "".join(parts)
        if not text.strip():
            raise ProviderError(f"Ollama model '{self.model}' returned an empty reply")
        return GenerationResponse(text=text, provider=self.name, model=self.model)

    def _timeout_message(self) -> str:
        return (
            f"Ollama model '{self.model}' did not finish replying within "
            f"{self._timeout:g} seconds. Raise TRUSTNET_OLLAMA_TIMEOUT_SECONDS or use a "
            "smaller model."
        )
