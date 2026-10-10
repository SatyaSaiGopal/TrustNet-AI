"""Build the configured provider. The only place that maps settings to a backend."""

from trustnet.config import Settings
from trustnet.providers.base import LLMProvider, ProviderConfigurationError
from trustnet.providers.ollama import OllamaProvider
from trustnet.providers.openai import OpenAIProvider


def create_provider(settings: Settings) -> LLMProvider:
    """Return the real provider named by the settings. Never returns a mock."""
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ProviderConfigurationError(
                "TRUSTNET_OPENAI_API_KEY is not set. Set your API key in .env or the UI."
            )
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model or "gpt-4o-mini",
            base_url=settings.openai_base_url or "https://api.openai.com/v1",
            timeout_seconds=settings.openai_timeout_seconds,
        )

    if settings.ollama_model is None:
        raise ProviderConfigurationError(
            "TRUSTNET_OLLAMA_MODEL is not set. Set it to a model you have pulled: "
            "`ollama list` shows them and `ollama pull <name>` downloads one."
        )
    return OllamaProvider(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
