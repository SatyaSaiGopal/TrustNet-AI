"""Headless smoke test of the Streamlit dashboard in its synthetic demo mode."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "src" / "trustnet" / "ui" / "app.py"


@pytest.fixture
def app(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> AppTest:
    clean_env.chdir(tmp_path)  # no developer .env, no project database
    return AppTest.from_file(str(APP), default_timeout=120).run()


def texts(elements) -> str:
    return " | ".join(str(element.value) for element in elements)


def test_dashboard_opens_in_synthetic_mode_and_says_so(app: AppTest) -> None:
    assert not app.exception
    assert "SYNTHETIC DATA" in texts(app.warning)
    assert len(app.dataframe) == 0


def test_running_agents_shows_decision_metrics_evidence_and_strategies(app: AppTest) -> None:
    app.button(key="run").click().run()

    assert not app.exception
    headings = texts(app.subheader)
    for heading in (
        "Final decision (trust-weighted)",
        "Agent recommendations",
        "Reliability metrics",
        "Evidence status",
        "Strategy comparison",
        "Verified outcome",
    ):
        assert heading in headings
    assert len(app.dataframe) >= 3
    strategies = app.dataframe[len(app.dataframe) - 1].value
    assert list(strategies["strategy"]) == [
        "trust_weighted",
        "majority_vote",
        "equal_weight_vote",
        "highest_confidence",
    ]


def test_an_outcome_can_be_recorded_only_once(app: AppTest) -> None:
    app.selectbox[0].select_index(3).run()
    app.button(key="run").click().run()
    app.button(key="record").click().run()
    assert "Reliability metrics were recomputed" in texts(app.success)

    app.button(key="record").click().run()
    assert not app.exception
    assert "already has a verified outcome" in texts(app.error)
