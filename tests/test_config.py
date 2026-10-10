from pathlib import Path

import pytest
from pydantic import ValidationError

from trustnet.config import Settings


def load() -> Settings:
    # _env_file=None keeps a developer's local .env out of the tests.
    return Settings(_env_file=None)


def test_defaults(clean_env: pytest.MonkeyPatch) -> None:
    settings = load()
    assert settings.environment == "development"
    assert settings.llm_provider == "ollama"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.database_path == Path("data/trustnet.db")


def test_no_model_is_assumed_by_default(clean_env: pytest.MonkeyPatch) -> None:
    assert load().ollama_model is None


def test_environment_variables_override_defaults(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("TRUSTNET_OLLAMA_MODEL", "some-model:1b")
    clean_env.setenv("TRUSTNET_DATABASE_PATH", "custom/place.db")
    clean_env.setenv("TRUSTNET_OLLAMA_TIMEOUT_SECONDS", "5")
    settings = load()
    assert settings.ollama_model == "some-model:1b"
    assert settings.database_path == Path("custom/place.db")
    assert settings.ollama_timeout_seconds == 5.0


@pytest.mark.parametrize("value", ["", "   "])
def test_blank_model_counts_as_unset(clean_env: pytest.MonkeyPatch, value: str) -> None:
    clean_env.setenv("TRUSTNET_OLLAMA_MODEL", value)
    assert load().ollama_model is None


@pytest.mark.parametrize("provider", ["some-paid-api", "mock", "fake"])
def test_unknown_or_mock_provider_is_rejected(clean_env: pytest.MonkeyPatch, provider: str) -> None:
    clean_env.setenv("TRUSTNET_LLM_PROVIDER", provider)
    with pytest.raises(ValidationError):
        load()


def test_non_positive_timeout_is_rejected(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("TRUSTNET_OLLAMA_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValidationError):
        load()
