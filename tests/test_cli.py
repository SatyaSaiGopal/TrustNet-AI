import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

import trustnet.__main__ as cli
from trustnet.config import Settings
from trustnet.providers import DeterministicProvider, OllamaProvider


@pytest.fixture(autouse=True)
def isolated_settings(clean_env: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    # Keep a developer's .env and a real Ollama server out of these tests.
    clean_env.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    return clean_env


def test_unset_model_is_a_configuration_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["MOCK: a question"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Configuration error: TRUSTNET_OLLAMA_MODEL is not set" in captured.err


@pytest.mark.parametrize("question", ["", "   "])
def test_blank_question_is_rejected(capsys: pytest.CaptureFixture[str], question: str) -> None:
    assert cli.main([question]) == 2
    assert "Invalid question" in capsys.readouterr().err


def test_unreachable_model_exits_non_zero_and_prints_no_agent_output(
    isolated_settings: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fake_chat: Any
) -> None:
    down = OllamaProvider(
        model="test-model",
        chat_model_factory=fake_chat(error=httpx.ConnectError("actively refused")),
    )
    isolated_settings.setattr(cli, "create_provider", lambda settings: down)

    assert cli.main(["MOCK: a question"]) == 1

    captured = capsys.readouterr()
    printed = json.loads(captured.out)
    assert printed["responses"] == []
    assert printed["is_complete"] is False
    assert printed["missing_agent_ids"] == ["analyst", "skeptic", "researcher"]
    assert captured.err.count("AGENT FAILED") == 3
    assert "Cannot reach Ollama" in captured.err


def test_successful_run_prints_every_response(
    isolated_settings: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    draft_json: Callable[..., str],
) -> None:
    scripted = DeterministicProvider(default=draft_json())
    isolated_settings.setattr(cli, "create_provider", lambda settings: scripted)

    assert cli.main(["MOCK: a question"]) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed["is_complete"] is True
    assert [item["agent_id"] for item in printed["responses"]] == [
        "analyst",
        "skeptic",
        "researcher",
    ]
    assert all(item["is_mock"] for item in printed["responses"])
