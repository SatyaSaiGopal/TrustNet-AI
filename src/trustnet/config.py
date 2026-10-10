"""Application configuration, loaded from environment variables.

Every setting is read from a ``TRUSTNET_``-prefixed environment variable, with an
optional ``.env`` file in the working directory as a fallback. Real environment
variables take precedence over ``.env``.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRUSTNET_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    database_path: Path = Path("data/trustnet.db")

    llm_provider: Literal["ollama", "openai"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    # No default: which models are pulled differs per machine, so it must be chosen.
    ollama_model: str | None = None
    # Per agent, and it includes time spent queued behind the other agents' requests.
    ollama_timeout_seconds: float = Field(default=600.0, gt=0)

    # Hosted API Key provider (OpenAI, Groq, OpenRouter, DeepSeek)
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_timeout_seconds: float = Field(default=60.0, gt=0)

    # Trust engine. Unset weights fall back to the defaults in trustnet.trust.TrustWeights.
    trust_min_samples: int = Field(default=5, ge=1)
    weight_historical_accuracy: float | None = Field(default=None, gt=0)
    weight_calibration: float | None = Field(default=None, ge=0)
    weight_evidence_quality: float | None = Field(default=None, ge=0)
    weight_consistency: float | None = Field(default=None, ge=0)
    weight_insufficient_information_behavior: float | None = Field(default=None, ge=0)

    # Decision engine. Unset values fall back to trustnet.decision.DecisionConfig.
    decision_min_support: float | None = None
    decision_min_trust: float | None = None
    decision_require_verified_evidence: bool | None = None

    # Evidence checking. Sources are text files in this directory, cited by file name.
    evidence_dir: Path | None = None
    evidence_llm_judge: bool = False

    @field_validator("ollama_model", "openai_api_key")
    @classmethod
    def _blank_means_unset(cls, value: str | None) -> str | None:
        return value.strip() or None if value else None


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings. Call ``get_settings.cache_clear()`` to reload."""
    return Settings()
