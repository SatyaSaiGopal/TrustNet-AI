import json
import os
import sqlite3
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from trustnet.schemas import Task
from trustnet.storage import connect, initialize


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove every TRUSTNET_ variable so tests see only what they set."""
    for name in list(os.environ):
        if name.startswith("TRUSTNET_"):
            monkeypatch.delenv(name)
    return monkeypatch


@pytest.fixture
def db() -> Iterator[sqlite3.Connection]:
    connection = connect(":memory:")
    initialize(connection)
    yield connection
    connection.close()


@pytest.fixture
def task() -> Task:
    return Task(task_id="mock-task-1", question="MOCK: should we do X?", is_synthetic=True)


@pytest.fixture
def draft_json() -> Callable[..., str]:
    """Build the JSON text a model would return for an agent. Mock data for tests."""

    def build(**overrides: Any) -> str:
        fields: dict[str, Any] = {
            "recommendation": "MOCK do X",
            "rationale": "MOCK because of Y",
            "confidence": 0.6,
            "evidence_references": [{"claim": "MOCK claim", "reference": None}],
            "uncertainties": ["MOCK unknown"],
        }
        fields.update(overrides)
        return json.dumps({key: value for key, value in fields.items() if value is not ...})

    return build


class FakeChatModel:
    """Stands in for ChatOllama: yields scripted chunks or raises a scripted error."""

    def __init__(self, chunks: tuple[str, ...] = ("ok",), error: Exception | None = None) -> None:
        self.chunks = chunks
        self.error = error
        self.init_kwargs: dict[str, Any] = {}
        self.messages: list[tuple[str, str]] = []

    def __call__(self, **kwargs: Any) -> "FakeChatModel":
        self.init_kwargs = kwargs
        return self

    def stream(self, messages: list[tuple[str, str]]) -> Iterator[SimpleNamespace]:
        self.messages = messages
        if self.error is not None:
            raise self.error
        for chunk in self.chunks:
            yield SimpleNamespace(content=chunk)


@pytest.fixture
def fake_chat() -> type[FakeChatModel]:
    return FakeChatModel
