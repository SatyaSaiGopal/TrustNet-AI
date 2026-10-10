from typing import Any

import httpx
import ollama
import pytest
from langchain_ollama import ChatOllama

from trustnet.config import Settings
from trustnet.providers import (
    DeterministicProvider,
    GenerationRequest,
    LLMProvider,
    OllamaProvider,
    OpenAIProvider,
    ProviderConfigurationError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    create_provider,
)


def provider_with(chat: Any, **kwargs: Any) -> OllamaProvider:
    return OllamaProvider(model="test-model", chat_model_factory=chat, **kwargs)


def test_deterministic_provider_is_repeatable_and_labelled_mock() -> None:
    provider = DeterministicProvider(responses={"ping": "pong"})
    request = GenerationRequest(prompt="ping")
    first, second = provider.generate(request), provider.generate(request)
    assert first == second
    assert first.text == "pong"
    assert first.provider == "mock"
    assert provider.calls == [request, request]


def test_deterministic_provider_falls_back_to_default() -> None:
    assert DeterministicProvider().generate(GenerationRequest(prompt="x")).text == "MOCK RESPONSE"


def test_ollama_provider_uses_chat_ollama_by_default() -> None:
    assert OllamaProvider(model="m")._chat_model_factory is ChatOllama


def test_ollama_provider_configures_the_chat_model_and_joins_the_reply(fake_chat: Any) -> None:
    chat = fake_chat(chunks=("hel", "lo"))
    provider = provider_with(chat, base_url="http://host:1234/", timeout_seconds=7)
    schema = {"type": "object"}

    response = provider.generate(
        GenerationRequest(prompt="hi", system="be brief", seed=11, response_schema=schema)
    )

    assert response.text == "hello"
    assert response.provider == "ollama"
    assert response.model == "test-model"
    assert chat.messages == [("system", "be brief"), ("human", "hi")]
    assert chat.init_kwargs == {
        "model": "test-model",
        "base_url": "http://host:1234",
        "temperature": 0.0,
        "seed": 11,
        "format": schema,
        "client_kwargs": {"timeout": 7},
    }


def test_ollama_provider_sends_no_system_message_when_none_is_given(fake_chat: Any) -> None:
    chat = fake_chat()
    provider_with(chat).generate(GenerationRequest(prompt="hi"))
    assert chat.messages == [("human", "hi")]


@pytest.mark.parametrize(
    "error",
    [httpx.ConnectError("actively refused"), ConnectionError("actively refused")],
    ids=["httpx", "builtin"],
)
def test_unreachable_server_is_reported_clearly(fake_chat: Any, error: Exception) -> None:
    provider = provider_with(fake_chat(error=error), base_url="http://host:1234")
    with pytest.raises(ProviderUnavailableError) as raised:
        provider.generate(GenerationRequest(prompt="hi"))
    message = str(raised.value)
    assert "Cannot reach Ollama at http://host:1234" in message
    assert "actively refused" in message


def test_missing_model_is_reported_with_the_fix(fake_chat: Any) -> None:
    error = ollama.ResponseError("model 'test-model' not found", 404)
    with pytest.raises(ProviderUnavailableError) as raised:
        provider_with(fake_chat(error=error)).generate(GenerationRequest(prompt="hi"))
    message = str(raised.value)
    assert "'test-model' is not installed" in message
    assert "ollama pull test-model" in message


def test_other_server_errors_are_reported(fake_chat: Any) -> None:
    error = ollama.ResponseError("out of memory", 500)
    with pytest.raises(ProviderError, match="out of memory") as raised:
        provider_with(fake_chat(error=error)).generate(GenerationRequest(prompt="hi"))
    assert not isinstance(raised.value, ProviderUnavailableError)


def test_network_timeout_is_reported(fake_chat: Any) -> None:
    provider = provider_with(fake_chat(error=httpx.ReadTimeout("timed out")), timeout_seconds=3)
    with pytest.raises(ProviderTimeoutError, match="within 3 seconds"):
        provider.generate(GenerationRequest(prompt="hi"))


def test_reply_that_overruns_the_deadline_is_cut_off(fake_chat: Any) -> None:
    ticks = iter([0.0, 1.0, 100.0])
    provider = provider_with(
        fake_chat(chunks=("slow", "er", "never read")),
        timeout_seconds=10,
        clock=lambda: next(ticks),
    )
    with pytest.raises(ProviderTimeoutError, match="within 10 seconds"):
        provider.generate(GenerationRequest(prompt="hi"))


@pytest.mark.parametrize("chunks", [(), ("",), ("  \n",)])
def test_empty_reply_is_an_error_not_an_answer(fake_chat: Any, chunks: tuple[str, ...]) -> None:
    with pytest.raises(ProviderError, match="empty reply"):
        provider_with(fake_chat(chunks=chunks)).generate(GenerationRequest(prompt="hi"))


@pytest.mark.parametrize("model", ["", "   "])
def test_ollama_provider_requires_a_model_name(model: str) -> None:
    with pytest.raises(ProviderConfigurationError, match="model name"):
        OllamaProvider(model=model)


def test_ollama_provider_requires_a_positive_timeout() -> None:
    with pytest.raises(ProviderConfigurationError, match="timeout"):
        OllamaProvider(model="m", timeout_seconds=0)


def test_generation_request_rejects_an_empty_prompt() -> None:
    with pytest.raises(ValueError):
        GenerationRequest(prompt="")


def test_factory_builds_the_configured_ollama_provider() -> None:
    provider = create_provider(Settings(_env_file=None, ollama_model="some-model:1b"))
    assert isinstance(provider, OllamaProvider)
    assert provider.model == "some-model:1b"


def test_factory_refuses_to_guess_a_model(clean_env: pytest.MonkeyPatch) -> None:
    with pytest.raises(ProviderConfigurationError, match="TRUSTNET_OLLAMA_MODEL is not set"):
        create_provider(Settings(_env_file=None))


@pytest.mark.parametrize(
    "provider",
    [
        DeterministicProvider(),
        OllamaProvider(model="m"),
        OpenAIProvider(api_key="sk-test", model="m"),
    ],
    ids=["mock", "ollama", "openai"],
)
def test_providers_satisfy_the_interface(provider: object) -> None:
    assert isinstance(provider, LLMProvider)


@pytest.mark.parametrize("key", ["", "   "])
def test_openai_provider_requires_an_api_key(key: str) -> None:
    with pytest.raises(ProviderConfigurationError, match="API key is required"):
        OpenAIProvider(api_key=key)


@pytest.mark.parametrize("model", ["", "   "])
def test_openai_provider_requires_a_model_name(model: str) -> None:
    with pytest.raises(ProviderConfigurationError, match="model name is required"):
        OpenAIProvider(api_key="sk-valid", model=model)


def test_openai_provider_requires_a_positive_timeout() -> None:
    with pytest.raises(ProviderConfigurationError, match="timeout"):
        OpenAIProvider(api_key="sk-valid", model="m", timeout_seconds=0)


def test_openai_provider_successful_generate() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer sk-test-key"
        assert "chat/completions" in str(request.url)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"recommendation": "A", "rationale": "valid", "confidence": 0.95, "evidence_references": [], "uncertainties": []}'
                        }
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(
        api_key="sk-test-key",
        model="gpt-4o-mini",
        http_client=client,
    )
    response = provider.generate(GenerationRequest(prompt="test question", system="system instructions"))
    assert response.provider == "openai"
    assert response.model == "gpt-4o-mini"
    assert "recommendation" in response.text


def test_openai_provider_reports_401_clearly() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Invalid API key"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="sk-bad-key", http_client=client)
    with pytest.raises(ProviderConfigurationError, match="Authentication failed"):
        provider.generate(GenerationRequest(prompt="hi"))


def test_openai_provider_reports_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="sk-valid", timeout_seconds=5, http_client=client)
    with pytest.raises(ProviderTimeoutError, match="within 5 seconds"):
        provider.generate(GenerationRequest(prompt="hi"))


def test_openai_provider_reports_server_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "internal error"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="sk-valid", http_client=client)
    with pytest.raises(ProviderError, match="internal error"):
        provider.generate(GenerationRequest(prompt="hi"))


def test_openai_provider_rejects_empty_reply() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "   "}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAIProvider(api_key="sk-valid", http_client=client)
    with pytest.raises(ProviderError, match="empty reply"):
        provider.generate(GenerationRequest(prompt="hi"))


def test_factory_builds_configured_openai_provider() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="openai",
        openai_api_key="sk-env-key",
        openai_model="gpt-4o",
    )
    provider = create_provider(settings)
    assert isinstance(provider, OpenAIProvider)
    assert provider.model == "gpt-4o"


def test_factory_requires_api_key_when_openai_selected() -> None:
    settings = Settings(_env_file=None, llm_provider="openai", openai_api_key=None)
    with pytest.raises(ProviderConfigurationError, match="TRUSTNET_OPENAI_API_KEY is not set"):
        create_provider(settings)

