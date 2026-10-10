import importlib
from pathlib import Path

import pytest

import trustnet

MODULES = [
    "trustnet.config",
    "trustnet.schemas",
    "trustnet.providers",
    "trustnet.agents",
    "trustnet.agents.workflow",
    "trustnet.agents.mock",
    "trustnet.trust",
    "trustnet.evidence",
    "trustnet.decision",
    "trustnet.storage",
    "trustnet.pipeline",
    "trustnet.evaluation",
    "trustnet.evaluation.demo",
    "trustnet.api",
    "trustnet.ui",
    "trustnet.__main__",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    assert importlib.import_module(name) is not None


def test_version_is_set() -> None:
    assert trustnet.__version__ == "0.1.0"


def test_core_logic_does_not_depend_on_the_ui_or_api_frameworks() -> None:
    for name in MODULES:
        module = importlib.import_module(name)
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "import streamlit" not in source and "import fastapi" not in source, name


def test_mock_agents_are_not_part_of_the_public_agents_api() -> None:
    import trustnet.agents

    assert "MockAgent" not in trustnet.agents.__all__
    assert not hasattr(trustnet.agents, "MockAgent")
